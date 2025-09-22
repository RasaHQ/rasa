from __future__ import annotations

import asyncio
import copy
import string
import time
from dataclasses import asdict, dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

import structlog
from sanic import Websocket  # type: ignore
from sanic.exceptions import ServerError, WebsocketClosed

from rasa.core.channels import InputChannel, OutputChannel, UserMessage
from rasa.core.channels.constants import (
    USER_CONVERSATION_SESSION_END,
    USER_CONVERSATION_SESSION_START,
    USER_CONVERSATION_SILENCE_TIMEOUT,
)
from rasa.core.channels.voice_ready.utils import (
    CallParameters,
    validate_voice_license_scope,
)
from rasa.core.channels.voice_stream.asr.asr_engine import ASREngine
from rasa.core.channels.voice_stream.asr.asr_event import (
    ASREvent,
    NewTranscript,
    UserIsSpeaking,
    UserSilence,
)
from rasa.core.channels.voice_stream.asr.azure import AzureASR
from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR
from rasa.core.channels.voice_stream.audio_bytes import HERTZ, RasaAudioBytes
from rasa.core.channels.voice_stream.call_state import (
    CallState,
    _call_state,
    call_state,
)
from rasa.core.channels.voice_stream.tts.azure import AzureTTS
from rasa.core.channels.voice_stream.tts.cartesia import CartesiaTTS
from rasa.core.channels.voice_stream.tts.deepgram import DeepgramTTS
from rasa.core.channels.voice_stream.tts.tts_cache import TTSCache
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine, TTSError
from rasa.core.channels.voice_stream.util import (
    generate_silence,
)
from rasa.shared.core.constants import SILENCE_TIMEOUT_SLOT
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.utils.common import (
    class_from_module_path,
    mark_as_beta_feature,
)
from rasa.utils.io import remove_emojis

logger = structlog.get_logger(__name__)

# define constants for the voice channel
DEFAULT_INTERRUPTION_MIN_WORDS = 3


@dataclass
class InterruptionConfig:
    enabled: bool = False
    min_words: int = DEFAULT_INTERRUPTION_MIN_WORDS


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
    if not asr_config:
        raise ValueError("ASR configuration dictionary cannot be empty")

    if "name" not in asr_config:
        raise ValueError(
            "ASR configuration must contain 'name' key specifying the engine type"
        )

    name = str(asr_config["name"])
    asr_config = copy.copy(asr_config)
    asr_config.pop("name")
    if name.lower() == "deepgram":
        return DeepgramASR.from_config_dict(asr_config)
    if name == "azure":
        return AzureASR.from_config_dict(asr_config)
    else:
        mark_as_beta_feature("Custom ASR Engine")
        try:
            asr_engine_class = class_from_module_path(name)
            return asr_engine_class.from_config_dict(asr_config)
        except NameError:
            raise InvalidConfigException(
                f"Failed to initialize ASR Engine with type '{name}'. "
                f"Please make sure the method `from_config_dict`is implemented."
            )
        except TypeError as e:
            raise InvalidConfigException(
                f"Failed to initialize ASR Engine with type '{name}'. "
                f"Invalid configuration provided. "
                f"Error: {e}"
            )


def tts_engine_from_config(tts_config: Dict) -> TTSEngine:
    if not tts_config:
        raise ValueError("TTS configuration dictionary cannot be empty")

    if "name" not in tts_config:
        raise ValueError(
            "TTS configuration must contain 'name' key specifying the engine type"
        )

    name = str(tts_config["name"])
    tts_config = copy.copy(tts_config)
    tts_config.pop("name")
    if name.lower() == "azure":
        return AzureTTS.from_config_dict(tts_config)
    elif name.lower() == "cartesia":
        return CartesiaTTS.from_config_dict(tts_config)
    elif name.lower() == "deepgram":
        return DeepgramTTS.from_config_dict(tts_config)
    else:
        mark_as_beta_feature("Custom TTS Engine")
        try:
            tts_engine_class = class_from_module_path(name)
            return tts_engine_class.from_config_dict(tts_config)
        except NameError:
            raise InvalidConfigException(
                f"Failed to initialize TTS Engine with type '{name}'. "
                f"Please make sure the method `from_config_dict`is implemented."
            )
        except TypeError as e:
            raise InvalidConfigException(
                f"Failed to initialize TTS Engine with type '{name}'. "
                f"Invalid configuration provided. "
                f"Error: {e}"
            )


class VoiceOutputChannel(OutputChannel):
    def __init__(
        self,
        voice_websocket: Websocket,
        tts_engine: TTSEngine,
        tts_cache: TTSCache,
        min_buffer_size: int = 0,
    ):
        super().__init__()
        self.voice_websocket = voice_websocket
        self.tts_engine = tts_engine
        self.tts_cache = tts_cache
        self.latest_message_id: Optional[str] = None
        self.min_buffer_size = min_buffer_size

    def rasa_audio_bytes_to_channel_bytes(
        self, rasa_audio_bytes: RasaAudioBytes
    ) -> bytes:
        """Turn rasa's audio byte format into the format for the channel."""
        raise NotImplementedError

    def channel_bytes_to_message(self, recipient_id: str, channel_bytes: bytes) -> str:
        """Wrap the bytes for the channel in the proper format."""
        raise NotImplementedError

    def create_marker_message(self, recipient_id: str) -> Tuple[str, str]:
        """Create a marker message for a specific channel."""
        raise NotImplementedError

    async def send_marker_message(self, recipient_id: str) -> None:
        """Send a message that marks positions in the audio stream."""
        marker_message, mark_id = self.create_marker_message(recipient_id)
        await self.voice_websocket.send(marker_message)
        self.latest_message_id = mark_id

    async def send_start_marker(self, recipient_id: str) -> None:
        """Send a marker message before the first audio chunk."""
        # Default implementation uses the generic marker message
        await self.send_marker_message(recipient_id)

    async def send_intermediate_marker(self, recipient_id: str) -> None:
        """Send a marker message during audio streaming."""
        await self.send_marker_message(recipient_id)

    async def send_end_marker(self, recipient_id: str) -> None:
        """Send a marker message after the last audio chunk."""
        await self.send_marker_message(recipient_id)

    def update_silence_timeout(self) -> None:
        """Updates the silence timeout for the session."""
        if self.tracker_state:
            call_state.silence_timeout = self.tracker_state["slots"][
                SILENCE_TIMEOUT_SLOT
            ]
            logger.debug(
                "voice_channel.silence_timeout_updated",
                silence_timeout=call_state.silence_timeout,
            )

    async def send_text_with_buttons(
        self,
        recipient_id: str,
        text: str,
        buttons: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Uses the concise button output format for voice channels."""
        await self.send_text_with_buttons_concise(recipient_id, text, buttons, **kwargs)

    def _track_rasa_processing_latency(self) -> None:
        """Track and log Rasa processing completion latency."""
        if call_state.rasa_processing_start_time:
            call_state.rasa_processing_latency_ms = (
                time.time() - call_state.rasa_processing_start_time
            ) * 1000
            logger.debug(
                "voice_channel.rasa_processing_latency",
                latency_ms=call_state.rasa_processing_latency_ms,
            )

    def _track_tts_first_byte_latency(self) -> None:
        """Track and log TTS first byte latency."""
        if call_state.tts_start_time:
            call_state.tts_first_byte_latency_ms = (
                time.time() - call_state.tts_start_time
            ) * 1000
            logger.debug(
                "voice_channel.tts_first_byte_latency",
                latency_ms=call_state.tts_first_byte_latency_ms,
            )

    def _track_tts_complete_latency(self) -> None:
        """Track and log TTS completion latency."""
        if call_state.tts_start_time:
            call_state.tts_complete_latency_ms = (
                time.time() - call_state.tts_start_time
            ) * 1000
            logger.debug(
                "voice_channel.tts_complete_latency",
                latency_ms=call_state.tts_complete_latency_ms,
            )

    async def send_text_message(
        self, recipient_id: str, text: str, **kwargs: Any
    ) -> None:
        text = remove_emojis(text)
        self.update_silence_timeout()

        # Track Rasa processing completion
        self._track_rasa_processing_latency()

        # Track TTS start time
        call_state.tts_start_time = time.time()

        cached_audio_bytes = self.tts_cache.get(text)
        collected_audio_bytes = RasaAudioBytes(b"")
        seconds_marker = -1
        last_sent_offset = 0
        first_audio_sent = False
        logger.debug("voice_channel.sending_audio", text=text)

        # Send start marker before first chunk
        try:
            await self.send_start_marker(recipient_id)
        except (WebsocketClosed, ServerError):
            call_state.connection_failed = True

        # Is the response interruptible?
        allow_interruptions = kwargs.get("allow_interruptions", True)
        call_state.channel_data["allow_interruptions"] = allow_interruptions

        if cached_audio_bytes:
            audio_stream = self.chunk_audio(cached_audio_bytes)
        else:
            # Todo: make kwargs compatible with engine config
            synth_config = self.tts_engine.config.__class__.from_dict({})
            try:
                audio_stream = self.tts_engine.synthesize(text, synth_config)
            except TTSError:
                # TODO: add message that works without tts, e.g. loading from disc
                audio_stream = self.chunk_audio(generate_silence())

        async for audio_bytes in audio_stream:
            collected_audio_bytes = RasaAudioBytes(collected_audio_bytes + audio_bytes)

            # Check if we have enough new bytes to send
            current_buffer_size = len(collected_audio_bytes) - last_sent_offset
            should_send = current_buffer_size >= self.min_buffer_size

            if should_send:
                try:
                    # Track TTS first byte time
                    if not first_audio_sent:
                        self._track_tts_first_byte_latency()
                        first_audio_sent = True

                    # Send only the new bytes since last send
                    new_bytes = RasaAudioBytes(collected_audio_bytes[last_sent_offset:])
                    await self.send_audio_bytes(recipient_id, new_bytes)
                    last_sent_offset = len(collected_audio_bytes)

                    full_seconds_of_audio = len(collected_audio_bytes) // HERTZ
                    if full_seconds_of_audio > seconds_marker:
                        await self.send_intermediate_marker(recipient_id)
                        seconds_marker = full_seconds_of_audio

                except (WebsocketClosed, ServerError):
                    # ignore sending error, and keep collecting and caching audio bytes
                    call_state.connection_failed = True

        # Send any remaining audio not yet sent
        remaining_bytes = len(collected_audio_bytes) - last_sent_offset
        if remaining_bytes > 0:
            try:
                # Track TTS first byte time if not already tracked
                if not first_audio_sent:
                    self._track_tts_first_byte_latency()

                new_bytes = RasaAudioBytes(collected_audio_bytes[last_sent_offset:])
                await self.send_audio_bytes(recipient_id, new_bytes)
            except (WebsocketClosed, ServerError):
                # ignore sending error
                call_state.connection_failed = True

        # Track TTS completion time
        self._track_tts_complete_latency()

        try:
            await self.send_end_marker(recipient_id)
        except (WebsocketClosed, ServerError):
            # ignore sending error
            pass
        call_state.latest_bot_audio_id = self.latest_message_id

        if not cached_audio_bytes:
            self.tts_cache.put(text, collected_audio_bytes)

    async def send_audio_bytes(
        self, recipient_id: str, audio_bytes: RasaAudioBytes
    ) -> None:
        channel_bytes = self.rasa_audio_bytes_to_channel_bytes(audio_bytes)
        message = self.channel_bytes_to_message(recipient_id, channel_bytes)
        await self.voice_websocket.send(message)

    async def chunk_audio(
        self, audio_bytes: RasaAudioBytes, chunk_size: int = 2048
    ) -> AsyncIterator[RasaAudioBytes]:
        """Generate chunks from cached audio bytes."""
        offset = 0
        while offset < len(audio_bytes):
            chunk = audio_bytes[offset : offset + chunk_size]
            if len(chunk):
                yield RasaAudioBytes(chunk)
            offset += chunk_size
        return

    async def hangup(self, recipient_id: str, **kwargs: Any) -> None:
        call_state.should_hangup = True


class VoiceInputChannel(InputChannel):
    # All children of this class require a voice license to be used.
    requires_voice_license = True

    def __init__(
        self,
        server_url: str,
        asr_config: Dict,
        tts_config: Dict,
        interruptions: Optional[Dict[str, Any]] = None,
    ):
        if self.requires_voice_license:
            validate_voice_license_scope()

        self.server_url = server_url
        self.asr_config = asr_config
        self.tts_config = tts_config
        self.tts_cache = TTSCache(tts_config.get("cache_size", 1000))
        if interruptions:
            self.interruption_config = InterruptionConfig(**interruptions)
        else:
            self.interruption_config = InterruptionConfig()

        if self.interruption_config.enabled:
            mark_as_beta_feature(f"Interruption Handling in {self.name()}")

        logger.info(
            "voice_channel.initialized",
            name=self.name(),
            server_url=self.server_url,
            asr_config=self.asr_config,
            tts_config=self.tts_config,
            interruption_config=self.interruption_config,
        )

    def get_sender_id(self, call_parameters: CallParameters) -> str:
        """Get the sender ID for the channel."""
        return call_parameters.call_id

    async def monitor_silence_timeout(self, asr_event_queue: asyncio.Queue) -> None:
        timeout = call_state.silence_timeout
        if not timeout:
            return
        logger.debug("voice_channel.silence_timeout_watch_started", timeout=timeout)
        await asyncio.sleep(timeout)
        await asr_event_queue.put(UserSilence())
        logger.debug("voice_channel.silence_timeout_tripped")

    @staticmethod
    def _cancel_silence_timeout_watcher() -> None:
        """Cancels the silent timeout task if it exists."""
        if call_state.silence_timeout_watcher:
            logger.debug("voice_channel.cancelling_current_timeout_watcher_task")
            call_state.silence_timeout_watcher.cancel()
            call_state.silence_timeout_watcher = None

    @classmethod
    def validate_basic_credentials(cls, credentials: Optional[Dict[str, Any]]) -> None:
        """Validate the basic credentials for the voice channel."""
        if not credentials:
            cls.raise_missing_credentials_exception()
        if not isinstance(credentials, dict):
            raise InvalidConfigException(
                "Credentials must be a dictionary for voice channel."
            )

        required_keys = {"server_url", "asr", "tts"}
        credentials_keys = set(credentials.keys())
        if not required_keys.issubset(credentials_keys):
            missing_fields = required_keys - credentials_keys
            raise InvalidConfigException(
                f"Missing required fields in credentials: {', '.join(missing_fields)} "
                f"for channel {cls.name()}"
            )

    @classmethod
    def from_credentials(
        cls, credentials: Optional[Dict[str, Any]]
    ) -> VoiceInputChannel:
        raise NotImplementedError

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
            text=USER_CONVERSATION_SESSION_START,
            output_channel=output_channel,
            sender_id=self.get_sender_id(call_parameters),
            input_channel=self.name(),
            metadata=asdict(call_parameters),
        )
        await on_new_message(message)

    def map_input_message(
        self,
        message: Any,
        ws: Websocket,
    ) -> VoiceChannelAction:
        """Map a channel input message to a voice channel action."""
        raise NotImplementedError

    def should_interrupt(self, e: ASREvent) -> bool:
        """Determine if the current ASR event should interrupt playback.
        Returns True if the bot response is interruptible
        And if the user spoke more than 3 words.

        Arguments:
            e: The ASR event to evaluate.

        Returns:
            True if the event should interrupt playback, False otherwise.
        """
        # Are interruptions are enabled for the channel?
        if not self.interruption_config.enabled:
            return False

        # Is the bot response interruptible?
        if not call_state.channel_data.get("allow_interruptions", True):
            return False

        # Did the user speak more than 3 words?
        min_words = self.interruption_config.min_words
        if isinstance(e, UserIsSpeaking):
            translator = str.maketrans("", "", string.punctuation)
            words = e.text.translate(translator).split()
            return len(words) >= min_words
        return False

    async def interrupt_playback(
        self, ws: Websocket, call_parameters: CallParameters
    ) -> None:
        """Interrupt the current playback of audio.

        This function is used for interruption handling.
        As not all channels support flushing bot audio buffer,
        if a channel does not implement it. It has no effect."""
        pass

    async def run_audio_streaming(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        channel_websocket: Websocket,
    ) -> None:
        """Pipe input audio to ASR and consume ASR events simultaneously."""
        _call_state.set(CallState())
        asr_engine = asr_engine_from_config(self.asr_config)
        tts_engine = tts_engine_from_config(self.tts_config)
        asr_event_queue: asyncio.Queue = asyncio.Queue()
        await asr_engine.connect()

        call_parameters = await self.collect_call_parameters(channel_websocket)
        if call_parameters is None:
            raise ValueError("Failed to extract call parameters for call.")
        await self.start_session(
            channel_websocket, on_new_message, tts_engine, call_parameters
        )

        async def consume_audio_bytes() -> None:
            async for message in channel_websocket:
                was_bot_speaking_before = call_state.is_bot_speaking
                channel_action = self.map_input_message(message, channel_websocket)
                is_bot_speaking_after = call_state.is_bot_speaking

                if not was_bot_speaking_before and is_bot_speaking_after:
                    logger.debug("voice_channel.bot_started_speaking")
                    # relevant when the bot speaks multiple messages in one turn
                    self._cancel_silence_timeout_watcher()

                # we just stopped speaking, starting a watcher for silence timeout
                if was_bot_speaking_before and not is_bot_speaking_after:
                    logger.debug("voice_channel.bot_stopped_speaking")
                    self._cancel_silence_timeout_watcher()
                    call_state.silence_timeout_watcher = asyncio.create_task(
                        self.monitor_silence_timeout(asr_event_queue)
                    )
                if isinstance(channel_action, NewAudioAction):
                    await asr_engine.send_audio_chunks(channel_action.audio_bytes)
                elif isinstance(channel_action, EndConversationAction):
                    # end stream event came from the other side
                    await self.handle_disconnect(
                        channel_websocket, on_new_message, tts_engine, call_parameters
                    )
                    break

        async def receive_asr_events() -> None:
            async for event in asr_engine.stream_asr_events():
                await asr_event_queue.put(event)

        async def handle_asr_events() -> None:
            while True:
                event = await asr_event_queue.get()
                await self.handle_asr_event(
                    event,
                    channel_websocket,
                    on_new_message,
                    tts_engine,
                    call_parameters,
                )

        async def asr_keep_alive_task() -> None:
            interval = getattr(asr_engine.config, "keep_alive_interval", 5)
            while True:
                await asyncio.sleep(interval)
                await asr_engine.send_keep_alive()

        tasks = [
            asyncio.create_task(consume_audio_bytes()),
            asyncio.create_task(receive_asr_events()),
            asyncio.create_task(handle_asr_events()),
            asyncio.create_task(asr_keep_alive_task()),
        ]
        await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        await tts_engine.close_connection()
        await asr_engine.close_connection()
        await channel_websocket.close()
        self._cancel_silence_timeout_watcher()

    def create_output_channel(
        self, voice_websocket: Websocket, tts_engine: TTSEngine
    ) -> VoiceOutputChannel:
        """Create a matching voice output channel for this voice input channel."""
        raise NotImplementedError

    def _track_asr_latency(self) -> None:
        """Track and log ASR processing latency."""
        if call_state.user_speech_start_time:
            call_state.asr_latency_ms = (
                time.time() - call_state.user_speech_start_time
            ) * 1000
            logger.debug(
                "voice_channel.asr_latency", latency_ms=call_state.asr_latency_ms
            )

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
            logger.debug(
                "VoiceInputChannel.handle_asr_event.new_transcript", transcript=e.text
            )
            call_state.is_user_speaking = False

            # Track ASR and Rasa latencies
            self._track_asr_latency()
            call_state.rasa_processing_start_time = time.time()

            output_channel = self.create_output_channel(voice_websocket, tts_engine)
            message = UserMessage(
                text=e.text,
                output_channel=output_channel,
                sender_id=self.get_sender_id(call_parameters),
                input_channel=self.name(),
                metadata=asdict(call_parameters),
            )
            await on_new_message(message)
        elif isinstance(e, UserIsSpeaking):
            # Track when user starts speaking for ASR latency calculation
            if not call_state.is_user_speaking:
                call_state.user_speech_start_time = time.time()
            self._cancel_silence_timeout_watcher()
            call_state.is_user_speaking = True
            if self.should_interrupt(e):
                await self.interrupt_playback(voice_websocket, call_parameters)
        elif isinstance(e, UserSilence):
            output_channel = self.create_output_channel(voice_websocket, tts_engine)
            message = UserMessage(
                text=USER_CONVERSATION_SILENCE_TIMEOUT,
                output_channel=output_channel,
                sender_id=self.get_sender_id(call_parameters),
                input_channel=self.name(),
                metadata=asdict(call_parameters),
            )
            await on_new_message(message)

    async def handle_disconnect(
        self,
        channel_websocket: Websocket,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        tts_engine: TTSEngine,
        call_parameters: CallParameters,
    ) -> None:
        """Handle disconnection from the channel."""
        output_channel = self.create_output_channel(channel_websocket, tts_engine)
        message = UserMessage(
            text=USER_CONVERSATION_SESSION_END,
            output_channel=output_channel,
            sender_id=self.get_sender_id(call_parameters),
            input_channel=self.name(),
        )
        await on_new_message(message)
