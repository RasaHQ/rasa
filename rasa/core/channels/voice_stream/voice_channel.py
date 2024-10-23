import asyncio
import logging
import copy
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sanic.exceptions import ServerError, WebsocketClosed

from rasa.core.channels import InputChannel, OutputChannel, UserMessage
from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.asr.asr_engine import ASREngine
from rasa.core.channels.voice_stream.asr.asr_event import ASREvent, NewTranscript
from sanic import Websocket  # type: ignore

from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR
from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.tts.azure import AzureTTS
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine, TTSError
from rasa.core.channels.voice_stream.tts.cartesia import CartesiaTTS
from rasa.core.channels.voice_stream.tts.tts_cache import TTSCache

logger = logging.getLogger(__name__)


@dataclass
class VoiceChannelAction:
    pass


@dataclass
class NewAudioAction(VoiceChannelAction):
    audio_bytes: RasaAudioBytes


@dataclass
class EndConversationAction(VoiceChannelAction):
    pass


@dataclass
class ContinueConversationAction(VoiceChannelAction):
    pass


def asr_engine_from_config(asr_config: Dict) -> ASREngine:
    name = str(asr_config["name"]).lower()
    asr_config = copy.copy(asr_config)
    asr_config.pop("name")
    if name == "deepgram":
        return DeepgramASR.from_config_dict(asr_config)
    else:
        raise NotImplementedError


def tts_engine_from_config(tts_config: Dict) -> TTSEngine:
    name = str(tts_config["name"]).lower()
    tts_config = copy.copy(tts_config)
    tts_config.pop("name")
    if name == "azure":
        return AzureTTS.from_config_dict(tts_config)
    elif name == "cartesia":
        return CartesiaTTS.from_config_dict(tts_config)
    else:
        raise NotImplementedError(f"TTS engine {name} is not implemented")


class VoiceOutputChannel(OutputChannel):
    def __init__(
        self,
        voice_websocket: Websocket,
        tts_engine: TTSEngine,
        tts_cache: TTSCache,
    ):
        self.voice_websocket = voice_websocket
        self.tts_engine = tts_engine
        self.tts_cache = tts_cache

        self.should_hangup = False
        self.latest_message_id: Optional[str] = None

    def rasa_audio_bytes_to_channel_bytes(
        self, rasa_audio_bytes: RasaAudioBytes
    ) -> bytes:
        raise NotImplementedError

    def channel_bytes_to_messages(
        self, recipient_id: str, channel_bytes: bytes
    ) -> List[Any]:
        raise NotImplementedError

    async def send_text_message(
        self, recipient_id: str, text: str, **kwargs: Any
    ) -> None:
        cached_audio_bytes = self.tts_cache.get(text)

        if cached_audio_bytes:
            await self.send_audio_bytes(recipient_id, cached_audio_bytes)
            return
        collected_audio_bytes = RasaAudioBytes(b"")
        # Todo: make kwargs compatible with engine config
        synth_config = self.tts_engine.config.__class__.from_dict({})
        try:
            audio_stream = self.tts_engine.synthesize(text, synth_config)
        except TTSError:
            # TODO: add message that works without tts, e.g. loading from disc
            pass
        async for audio_bytes in audio_stream:
            try:
                await self.send_audio_bytes(recipient_id, audio_bytes)
            except (WebsocketClosed, ServerError):
                # ignore sending error, and keep collecting and caching audio bytes
                self.should_hangup = True

            collected_audio_bytes = RasaAudioBytes(collected_audio_bytes + audio_bytes)

        self.tts_cache.put(text, collected_audio_bytes)

    async def send_audio_bytes(
        self, recipient_id: str, audio_bytes: RasaAudioBytes
    ) -> None:
        channel_bytes = self.rasa_audio_bytes_to_channel_bytes(audio_bytes)
        for message in self.channel_bytes_to_messages(recipient_id, channel_bytes):
            await self.voice_websocket.send(message)

    async def hangup(self, recipient_id: str, **kwargs: Any) -> None:
        self.should_hangup = True


class VoiceInputChannel(InputChannel):
    def __init__(self, server_url: str, asr_config: Dict, tts_config: Dict):
        self.server_url = server_url
        self.asr_config = asr_config
        self.tts_config = tts_config
        self.tts_cache = TTSCache(tts_config.get("cache_size", 1000))

        # if set to a value, call will be hungup after marker is reached
        self.hangup_after: Optional[str] = None

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[str, Any]]) -> InputChannel:
        credentials = credentials or {}
        return cls(credentials["server_url"], credentials["asr"], credentials["tts"])

    def channel_bytes_to_rasa_audio_bytes(self, input_bytes: bytes) -> RasaAudioBytes:
        raise NotImplementedError

    async def collect_call_parameters(
        self, channel_websocket: Websocket
    ) -> Optional[CallParameters]:
        raise NotImplementedError

    async def start_session(
        self,
        channel_websocket: Websocket,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        tts_engine: TTSEngine,
        call_parameters: CallParameters,
    ) -> None:
        output_channel = self.create_output_channel(channel_websocket, tts_engine)
        message = UserMessage(
            "/session_start",
            output_channel,
            call_parameters.stream_id,
            input_channel=self.name(),
            metadata=asdict(call_parameters),
        )
        await on_new_message(message)

    def map_input_message(
        self,
        message: Any,
    ) -> VoiceChannelAction:
        """Map a channel input message to a voice channel action."""
        raise NotImplementedError

    async def run_audio_streaming(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        channel_websocket: Websocket,
    ) -> None:
        """Pipe input audio to ASR and consume ASR events simultaneously."""
        asr_engine = asr_engine_from_config(self.asr_config)
        tts_engine = tts_engine_from_config(self.tts_config)
        await asr_engine.connect()

        call_parameters = await self.collect_call_parameters(channel_websocket)
        if call_parameters is None:
            raise ValueError("Failed to extract call parameters for call.")
        await self.start_session(
            channel_websocket, on_new_message, tts_engine, call_parameters
        )

        async def consume_audio_bytes() -> None:
            async for message in channel_websocket:
                channel_action = self.map_input_message(message)
                if isinstance(channel_action, NewAudioAction):
                    await asr_engine.send_audio_chunks(channel_action.audio_bytes)
                elif isinstance(channel_action, EndConversationAction):
                    # end stream event came from the other side
                    break

        async def consume_asr_events() -> None:
            async for event in asr_engine.stream_asr_events():
                await self.handle_asr_event(
                    event,
                    channel_websocket,
                    on_new_message,
                    tts_engine,
                    call_parameters,
                )

        await asyncio.wait(
            [consume_audio_bytes(), consume_asr_events()],
            return_when=asyncio.FIRST_COMPLETED,
        )
        await tts_engine.close_connection()
        await asr_engine.close_connection()

    def create_output_channel(
        self, voice_websocket: Websocket, tts_engine: TTSEngine
    ) -> VoiceOutputChannel:
        """Create a matching voice output channel for this voice input channel."""
        raise NotImplementedError

    async def handle_asr_event(
        self,
        e: ASREvent,
        voice_websocket: Websocket,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        tts_engine: TTSEngine,
        call_parameters: CallParameters,
    ) -> None:
        """Handle a new event from the ASR system."""
        if isinstance(e, NewTranscript) and e.text:
            logger.info(f"New transcript: {e.text}")
            output_channel = self.create_output_channel(voice_websocket, tts_engine)
            message = UserMessage(
                e.text,
                output_channel,
                call_parameters.stream_id,
                input_channel=self.name(),
                metadata=asdict(call_parameters),
            )
            await on_new_message(message)

            if output_channel.should_hangup:
                self.hangup_after = output_channel.latest_message_id
