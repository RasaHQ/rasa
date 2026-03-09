from __future__ import annotations

import asyncio
import copy
import string
import time
from dataclasses import asdict, dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Text,
    Tuple,
    cast,
)

import structlog
from sanic import Blueprint, Websocket  # type: ignore
from sanic.exceptions import WebsocketClosed

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
from rasa.core.channels.voice_stream.asr import BUILT_IN_ASR_ENGINES
from rasa.core.channels.voice_stream.asr.asr_engine import ASREngine
from rasa.core.channels.voice_stream.asr.asr_event import (
    ASREvent,
    NewTranscript,
    UserIsSpeaking,
    UserSilence,
)
from rasa.core.channels.voice_stream.audio_bytes import (
    MULAW_8KHZ,
    AudioFormat,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.call_state import (
    CallState,
    RasaIsListening,
    RasaIsProcessing,
    UserStoppedSpeaking,
    _call_state,
    call_state,
)
from rasa.core.channels.voice_stream.call_state import (
    UserIsSpeaking as UserIsSpeakingCallStateMessage,
)
from rasa.core.channels.voice_stream.tts import BUILT_IN_TTS_ENGINES
from rasa.core.channels.voice_stream.tts.tts_cache import TTSCache
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine, TTSError
from rasa.core.channels.voice_stream.util import (
    generate_silence,
)
from rasa.hooks import hookimpl
from rasa.plugin import plugin_manager
from rasa.shared.core.constants import LANGUAGE_SLOT, SILENCE_TIMEOUT_SLOT
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.utils.common import (
    class_from_module_path,
    mark_as_beta_feature,
)
from rasa.utils.io import remove_emojis

if TYPE_CHECKING:
    from rasa.core.agent import Agent
    from rasa.shared.core.trackers import DialogueStateTracker

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


@dataclass
class DTMFInputAction(VoiceChannelAction):
    digit: str


def asr_engine_from_config(
    asr_config: Dict,
    format: AudioFormat,
    language: str,
    additional_languages: Optional[List[str]] = None,
) -> ASREngine:
    if not asr_config:
        raise ValueError("ASR configuration dictionary cannot be empty")

    if "name" not in asr_config:
        raise ValueError(
            "ASR configuration must contain 'name' key specifying the engine type"
        )

    name = str(asr_config["name"])
    asr_config = copy.copy(asr_config)
    asr_config.pop("name")

    built_in_asr_engine_loader = BUILT_IN_ASR_ENGINES.get(name.lower())

    if built_in_asr_engine_loader:
        clazz = built_in_asr_engine_loader()
        return clazz.from_config_dict(
            asr_config, format, language, additional_languages
        )
    else:
        mark_as_beta_feature("Custom ASR Engine")
        try:
            asr_engine_class = class_from_module_path(name)
            return asr_engine_class.from_config_dict(
                asr_config, format, language, additional_languages
            )
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


def tts_engine_from_config(
    tts_config: Dict,
    format: AudioFormat,
    language: str,
    additional_languages: Optional[List[str]] = None,
) -> TTSEngine:
    if not tts_config:
        raise ValueError("TTS configuration dictionary cannot be empty")

    if "name" not in tts_config:
        raise ValueError(
            "TTS configuration must contain 'name' key specifying the engine type"
        )

    name = str(tts_config["name"])
    tts_config = copy.copy(tts_config)
    tts_config.pop("name")

    tts_engine_loader = BUILT_IN_TTS_ENGINES.get(name.lower())
    if tts_engine_loader:
        clazz = tts_engine_loader()
        return clazz.from_config_dict(
            tts_config, format, language, additional_languages
        )
    else:
        mark_as_beta_feature("Custom TTS Engine")
        try:
            tts_engine_class = class_from_module_path(name)
            return tts_engine_class.from_config_dict(
                tts_config, format, language, additional_languages
            )
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
        audio_format: AudioFormat,
        min_buffer_size: int = 0,
    ):
        super().__init__()
        self.voice_websocket = voice_websocket
        self.tts_engine = tts_engine
        self.tts_cache = tts_cache
        self.min_buffer_size = min_buffer_size
        self.audio_format = audio_format
        self.latest_message_id: Optional[str] = None

        # For streaming responses - background task that sends TTS audio
        self.audio_sender_task: Optional[asyncio.Task] = None

    @property
    def supports_streaming(self) -> bool:
        """Whether this channel supports streaming responses."""
        return self.tts_engine.streaming_input

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
        try:
            await self.voice_websocket.send(marker_message)
        except WebsocketClosed:
            call_state.connection_failed = True
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

    def get_current_language(self) -> Optional[str]:
        """Get the current language from the tracker state."""
        if self.tracker_state:
            return self.tracker_state["slots"].get(LANGUAGE_SLOT)
        return None

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

    async def _stream_tts(self, recipient_id: str) -> None:
        """Background task: listens to TTS audio stream and sends to channel.

        This pulls audio from TTS and directly sends it to the channel.
        """
        await self._stream_audio_to_channel(
            recipient_id, self.tts_engine.stream_audio()
        )

    async def _synthesize_and_stream_tts(
        self, recipient_id: str, text: str
    ) -> Optional[RasaAudioBytes]:
        """Use producer/consumer pattern to send TTS audio and collect for caching.

        Returns the collected audio bytes for caching.
        """
        try:
            audio_stream = self.tts_engine.synthesize(text)
        except TTSError as e:
            logger.error("voice_channel.tts_synthesis_error", error=str(e))
            # TODO: add message that works without tts, e.g. loading from disc
            audio_stream = self.chunk_audio(generate_silence(self.audio_format))

        collected_audio = await self._stream_audio_to_channel(
            recipient_id, audio_stream
        )
        return collected_audio

    async def _send_cached_audio(
        self, recipient_id: str, cached_audio: RasaAudioBytes
    ) -> None:
        """Send cached audio directly to websocket."""
        await self._stream_audio_to_channel(
            recipient_id, self.chunk_audio(cached_audio)
        )

    async def _stream_audio_to_channel(
        self, recipient_id: str, audio_stream: AsyncIterator[RasaAudioBytes]
    ) -> Optional[RasaAudioBytes]:
        """Send audio from an async iterator to the channel.

        This function does a lot of things,
        - Tracks TTS first byte latency
        - Sends intermediate marker messages every second of audio
        - Collects audio bytes to return for caching
        - Buffer audio to send in chunks of min_buffer_size for efficiency
        - Track leftover byte for L16 format to ensure we always send
          complete samples (2 bytes per sample)
        """
        collected_audio = RasaAudioBytes(b"", format=self.audio_format)
        last_sent_offset = 0
        first_byte_received = False
        seconds_marker = -1
        leftover_byte: bytes = b""

        # Set stop_streaming_output_audio_chunks to match if user is speaking.
        # We are fixing this for A1, when Azure TTS is using HTTP mode
        # to synthesize audio.
        from rasa.core.channels.voice_stream.tts.azure import AzureTTS

        if cast(AzureTTS, self.tts_engine):
            call_state.stop_streaming_output_audio_chunks = call_state.is_user_speaking

        async for audio_chunk in audio_stream:
            collected_audio = collected_audio + audio_chunk

            if call_state.stop_streaming_output_audio_chunks:
                return None

            # Track TTS first byte time
            if not first_byte_received:
                self._track_tts_first_byte_latency()
                first_byte_received = True

            # Check if we have enough new bytes to send
            current_buffer_size = len(collected_audio) - last_sent_offset
            if current_buffer_size < self.min_buffer_size:
                # Not enough new audio to send yet, continue accumulating
                continue

            try:
                # send only the new bytes since last sent offset
                new_bytes = leftover_byte + collected_audio.data[last_sent_offset:]
                leftover_byte = b""

                # Ensure even byte length for L16 audio (2 bytes per sample)
                if len(new_bytes) % 2 != 0:
                    leftover_byte = new_bytes[-1:]
                    new_bytes = new_bytes[:-1]

                # Send if there are new bytes
                if len(new_bytes) > 0:
                    await self.send_audio_bytes(
                        recipient_id,
                        RasaAudioBytes(new_bytes, format=self.audio_format),
                    )
                last_sent_offset = len(collected_audio)

                # send intermediate marker every second of audio
                full_seconds_of_audio = int(collected_audio.full_seconds())
                if full_seconds_of_audio > seconds_marker:
                    await self.send_intermediate_marker(recipient_id)
                    seconds_marker = full_seconds_of_audio
            except WebsocketClosed:
                call_state.connection_failed = True
                # Continue collecting for cache even if send fails

        # send any remaining bytes (including leftover)
        await self._send_remaining_bytes(
            recipient_id, collected_audio, last_sent_offset, leftover_byte
        )

        return collected_audio

    async def _send_remaining_bytes(
        self,
        recipient_id: str,
        collected_audio: RasaAudioBytes,
        last_sent_offset: int,
        leftover_byte: bytes,
    ) -> None:
        """Send any remaining bytes after audio stream is complete."""
        remaining_bytes = len(collected_audio) - last_sent_offset
        if remaining_bytes == 0 and len(leftover_byte) == 0:
            return

        try:
            new_bytes = leftover_byte + collected_audio.data[last_sent_offset:]
            # Pad with zero byte if odd length (final chunk)
            if len(new_bytes) % 2 != 0:
                new_bytes = new_bytes + b"\x00"
            if len(new_bytes) > 0:
                await self.send_audio_bytes(
                    recipient_id,
                    RasaAudioBytes(new_bytes, format=self.audio_format),
                )
        except WebsocketClosed:
            # ignore sending error
            call_state.connection_failed = True

    async def send_response_chunk_start(
        self, recipient_id: Text, **kwargs: Any
    ) -> None:
        """Start streaming response session.

        Starts background task (listens to TTS audio, sends to websocket).
        """
        await super().send_response_chunk_start(recipient_id, **kwargs)

        # Let TTS engine prepare for this response (e.g., mode selection)
        await self.tts_engine.prepare_response(
            streaming_config=kwargs.get("streaming_config")
        )

        if not self.tts_engine.streaming_input:
            # Engine does not support streaming input
            # fallback to non-streaming synthesis
            return

        self.audio_sender_task = asyncio.create_task(self._stream_tts(recipient_id))
        await self.send_start_marker(recipient_id)
        logger.debug("voice_channel.start_streaming_response")

    async def send_response_chunk(
        self, recipient_id: str, chunk: str, **kwargs: Any
    ) -> None:
        """Send text chunk to TTS.

        The TTS engine will process this and the background consumer task
        will receive the audio and send it to the websocket.
        """
        await super().send_response_chunk(recipient_id, chunk, **kwargs)

        if not self.tts_engine.streaming_input:
            # Engine does not support streaming input
            # fallback to non-streaming synthesis
            return

        chunk = remove_emojis(chunk)
        await self.tts_engine.send_text_chunk(chunk)

    async def send_response_chunk_end(self, recipient_id: Text, **kwargs: Any) -> None:
        """End streaming response session.

        1. Flush TTS engine (process any remaining text)
        2. Wait for background task to finish sending all audio

        Args:
            recipient_id: The recipient ID.
            **kwargs: Additional arguments.
        """
        await super().send_response_chunk_end(recipient_id, **kwargs)

        if not self.tts_engine.streaming_input:
            # fallback to non-streaming synthesis
            return

        await self.tts_engine.signal_text_done()
        if self.audio_sender_task:
            await self.audio_sender_task
        await self.send_end_marker(recipient_id)
        call_state.latest_bot_audio_id = self.latest_message_id
        logger.debug("voice_channel.end_streaming_response")

    async def send_text_message(
        self, recipient_id: str, text: str, **kwargs: Any
    ) -> None:
        if self._is_duplicate_of_last_streamed_response(text):
            logger.debug("voice_channel.skip_non_streaming_response")
            return

        self._track_rasa_processing_latency()
        call_state.tts_start_time = time.time()

        text = remove_emojis(text)
        self.update_silence_timeout()

        # Check cache first
        cached_audio_bytes = self.tts_cache.get(text, self.audio_format)
        logger.debug(
            "voice_channel.sending_audio", text=text, cached=bool(cached_audio_bytes)
        )

        # Send start marker
        await self.send_start_marker(recipient_id)

        # Is the response interruptible?
        allow_interruptions = kwargs.get("allow_interruptions", True)
        call_state.channel_data["allow_interruptions"] = allow_interruptions

        if cached_audio_bytes:
            await self._send_cached_audio(recipient_id, cached_audio_bytes)
        else:
            collected_audio = await self._synthesize_and_stream_tts(recipient_id, text)
            if collected_audio:
                self.tts_cache.put(text, collected_audio)

        # Track TTS completion time
        self._track_tts_complete_latency()
        await self.send_end_marker(recipient_id)

        call_state.latest_bot_audio_id = self.latest_message_id

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
                yield chunk
            offset += chunk_size
        return

    async def hangup(self, recipient_id: str, **kwargs: Any) -> None:
        call_state.should_hangup = True

    async def send_turn_end_marker(self, recipient_id: str) -> None:
        """Send a marker to indicate the bot has finished its turn.

        Used internally by Rasa during conversation simulations.
        This is called after all bot messages in a turn have been sent.
        """
        pass

    async def notify_message_processing_started(self) -> None:
        """Notify that message is being processed."""
        await call_state.enqueue_event(RasaIsProcessing())

    async def notify_message_processing_completed(self) -> None:
        """Notify that message processing is completed."""
        await call_state.enqueue_event(RasaIsListening())


class VoiceLanguageChangePlugin:
    """Plugin that notifies ASR and TTS engines when the language slot changes.

    Registered per call in `run_audio_streaming` so each instance is tied to
    a specific sender and its own ASR/TTS engine pair.
    """

    def __init__(
        self, sender_id: str, asr_engine: ASREngine, tts_engine: TTSEngine
    ) -> None:
        self.sender_id = sender_id
        self.asr_engine = asr_engine
        self.tts_engine = tts_engine

    # Lifecycle methods
    def register_hook(self) -> None:
        """Register tasks for language change handling."""
        pm = plugin_manager()
        pm.register(self)

    async def unregister_hook(self) -> None:
        """Unregister tasks for language change handling."""
        pm = plugin_manager()
        pm.unregister(self)

    # hook implementation
    @hookimpl
    def after_action_executed(
        self, tracker: "DialogueStateTracker"
    ) -> Optional[Awaitable[None]]:
        """Check for language slot changes and notify ASR and TTS engines."""
        if tracker.sender_id != self.sender_id:
            return None

        language_slot = tracker.slots.get(LANGUAGE_SLOT)
        if not language_slot or not language_slot.value:
            # language slot is not set - this shouldn't be the case but
            # we don't want to update ASR / TTS engines in this case
            return None

        # This hook is sync, so return an awaitable and let the processor
        # await it before running side effects (e.g. TTS synthesis).
        return self.update_language(language_slot.value)

    async def update_language(self, new_language: str) -> None:
        """Update the language slot and notify ASR and TTS engines."""
        # Hook callbacks can run from non-voice channel contexts where `_call_state`
        # is unbound. Accessing `call_state` via LocalProxy in that case raises.
        current_call_state = _call_state.get(None)
        old_language = (
            current_call_state.current_language if current_call_state else None
        )

        if new_language == old_language:
            # language slot has not changed - no need to update ASR / TTS engines
            return

        logger.debug(
            "voice_channel.language_changed_after_action",
            old_language=old_language,
            new_language=new_language,
        )

        if current_call_state is not None:
            current_call_state.current_language = new_language

        await self.asr_engine.set_language(new_language)
        await self.tts_engine.set_language(new_language)


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

        self.agent: Optional[Agent] = None
        self.audio_format = MULAW_8KHZ
        self.server_url = server_url
        self.asr_config = asr_config
        self.tts_config = tts_config
        self.tts_cache = TTSCache(tts_config.get("cache_size", 1000))
        self.interruption_config = (
            InterruptionConfig(**interruptions)
            if interruptions
            else InterruptionConfig()
        )

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

    def _register_listeners(self, bp: Blueprint) -> None:
        """Attach shared listeners to a blueprint."""

        @bp.listener("after_server_start")  # type: ignore[misc]
        async def after_server_start(app: Any, loop: Any) -> None:
            if hasattr(app.ctx, "agent"):
                self.agent = app.ctx.agent

    def get_sender_id(self, call_parameters: CallParameters) -> str:
        """Get the sender ID for the channel."""
        return call_parameters.call_id

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
        self,
        channel_websocket: Websocket,
        request: Optional[Any] = None,
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
        sender_id = self.get_sender_id(call_parameters)
        message = UserMessage(
            text=USER_CONVERSATION_SESSION_START,
            output_channel=output_channel,
            sender_id=sender_id,
            input_channel=self.name(),
            metadata=asdict(call_parameters),
        )
        await on_new_message(message)
        await output_channel.send_turn_end_marker(sender_id)

    async def map_input_message(
        self,
        message: Any,
        ws: Websocket,
    ) -> VoiceChannelAction:
        """Map a channel input message to a voice channel action."""
        raise NotImplementedError

    def should_interrupt(self, e: ASREvent) -> bool:
        """Determine if the current ASR event should interrupt playback.

        Returns True if the bot response is interruptible
        and if the user spoke more than 3 words.

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
        if isinstance(e, (NewTranscript, UserIsSpeaking)):
            translator = str.maketrans("", "", string.punctuation)
            words = e.text.translate(translator).split()
            can_interrupt = len(words) >= min_words and call_state.is_bot_speaking
            return can_interrupt
        return False

    async def interrupt_playback(
        self, ws: Websocket, call_parameters: CallParameters
    ) -> None:
        """Interrupt the current playback of audio.

        This function is used for interruption handling.
        As not all channels support flushing bot audio buffer,
        if a channel does not implement it. It has no effect.
        """
        pass

    async def receive_asr_events(
        self,
        asr_engine: ASREngine,
        tts_engine: TTSEngine,
        asr_event_queue: asyncio.Queue,
        ws: Websocket,
        call_parameters: CallParameters,
    ) -> None:
        async for event in asr_engine.stream_asr_events():
            await asr_event_queue.put(event)
            if self.should_interrupt(event):
                logger.debug("voice_channel.asr_event_should_interrupt", ev=event)
                await tts_engine.stop_streaming()
                # We only stop sending audio bytes which came from Azure TTS in order
                # not to break Deepgram, Cartesia and Rime
                from rasa.core.channels.voice_stream.tts.azure import AzureTTS

                if cast(AzureTTS, tts_engine):
                    call_state.stop_streaming_output_audio_chunks = True
                await self.interrupt_playback(ws, call_parameters)

    async def handle_asr_events(
        self,
        asr_event_queue: asyncio.Queue,
        ws: Websocket,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        tts_engine: TTSEngine,
        call_parameters: CallParameters,
        asr_engine: ASREngine,
    ) -> None:
        while True:
            event = await asr_event_queue.get()
            await self.handle_asr_event(
                event,
                ws,
                on_new_message,
                tts_engine,
                call_parameters,
                asr_engine,
            )

    async def asr_keep_alive_task(self, asr_engine: ASREngine) -> None:
        interval = getattr(asr_engine.config, "keep_alive_interval", 5)
        while True:
            await asyncio.sleep(interval)
            await asr_engine.send_keep_alive()

    @property
    def additional_languages(self) -> List[str]:
        if (
            self.agent
            and self.agent.processor
            and self.agent.processor.model_metadata
            and self.agent.processor.model_metadata.additional_languages
        ):
            return self.agent.processor.model_metadata.additional_languages or []
        return []

    @property
    def language(self) -> str:
        if (
            self.agent
            and self.agent.processor
            and self.agent.processor.model_metadata
            and self.agent.processor.model_metadata.language
        ):
            return self.agent.processor.model_metadata.language
        return "en"

    def _initialize_call_state(self) -> None:
        call_state_ = CallState(
            internal_queue=asyncio.Queue(),
            asr_event_queue=asyncio.Queue(),
        )
        call_state_.start_state_monitoring()
        call_state_.current_language = self.language
        _call_state.set(call_state_)

    def _get_asr_and_tts_engines(self) -> Tuple[ASREngine, TTSEngine]:
        asr_engine = asr_engine_from_config(
            asr_config=self.asr_config,
            format=self.audio_format,
            language=self.language,
            additional_languages=self.additional_languages,
        )
        tts_engine = tts_engine_from_config(
            tts_config=self.tts_config,
            format=self.audio_format,
            language=self.language,
            additional_languages=self.additional_languages,
        )
        return asr_engine, tts_engine

    async def run_audio_streaming(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        channel_websocket: Websocket,
        request: Optional[Any] = None,
    ) -> None:
        """Pipe input audio to ASR and consume ASR events simultaneously."""
        self._initialize_call_state()

        call_parameters = await self.collect_call_parameters(channel_websocket, request)
        if call_parameters is None:
            raise ValueError("Failed to extract call parameters for call.")

        # Initialize ASR and TTS based on config
        asr_engine, tts_engine = self._get_asr_and_tts_engines()

        # Connect both ASR and TTS at the beginning
        await asr_engine.connect()
        await tts_engine.connect()

        sender_id = self.get_sender_id(call_parameters)
        language_plugin = VoiceLanguageChangePlugin(sender_id, asr_engine, tts_engine)
        tasks: List[asyncio.Task[Any]] = []

        try:
            language_plugin.register_hook()

            await self.start_session(
                channel_websocket, on_new_message, tts_engine, call_parameters
            )

            async def consume_audio_bytes() -> None:
                is_disconnected = False
                try:
                    async for message in channel_websocket:
                        channel_action = await self.map_input_message(
                            message, channel_websocket
                        )

                        if isinstance(channel_action, NewAudioAction):
                            await asr_engine.send_audio_chunks(
                                channel_action.audio_bytes
                            )
                        if isinstance(channel_action, DTMFInputAction):
                            await self.gather_dtmf_input(
                                channel_websocket,
                                tts_engine,
                                on_new_message,
                                call_parameters,
                                channel_action,
                            )
                        elif isinstance(channel_action, EndConversationAction):
                            # end stream event came from the other side
                            is_disconnected = True
                            await self.handle_disconnect(
                                channel_websocket,
                                on_new_message,
                                tts_engine,
                                call_parameters,
                            )
                            break
                except Exception as e:
                    logger.error("voice_channel.audio_streaming_error", error=str(e))
                    raise e
                finally:
                    # The websocket was closed cleanly by the remote end without sending
                    # an application-level disconnect message (e.g. Jambonz closes the
                    # websocket when the user hangs up without sending
                    # a "stop"-like event).
                    logger.info(
                        "voice_channel.websocket_closed_by_remote",
                        call_id=call_parameters.call_id,
                    )
                    if not is_disconnected:
                        # Avoid double disconnect handling
                        await self.handle_disconnect(
                            channel_websocket,
                            on_new_message,
                            tts_engine,
                            call_parameters,
                        )

            tasks = [
                asyncio.create_task(consume_audio_bytes()),
                asyncio.create_task(
                    self.receive_asr_events(
                        asr_engine,
                        tts_engine,
                        call_state.asr_event_queue,
                        channel_websocket,
                        call_parameters,
                    )
                ),
                asyncio.create_task(
                    self.handle_asr_events(
                        call_state.asr_event_queue,
                        channel_websocket,
                        on_new_message,
                        tts_engine,
                        call_parameters,
                        asr_engine,
                    )
                ),
                asyncio.create_task(self.asr_keep_alive_task(asr_engine)),
            ]
            await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            # Cancel all remaining tasks
            for task in tasks:
                task.cancel()

            # Wait for cancellations to complete, suppressing CancelledError
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Cleanup connections
            await language_plugin.unregister_hook()
            await asr_engine.close_connection()
            await tts_engine.close_connection()
            await channel_websocket.close()
            call_state.stop_all()

    def create_output_channel(
        self,
        voice_websocket: Websocket,
        tts_engine: TTSEngine,
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
        asr_event: ASREvent,
        voice_websocket: Websocket,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        tts_engine: TTSEngine,
        call_parameters: CallParameters,
        asr_engine: ASREngine,
    ) -> None:
        """Handle a new event from the ASR system."""
        if isinstance(asr_event, NewTranscript) and asr_event.text:
            logger.debug(
                "VoiceInputChannel.handle_asr_event.new_transcript",
                transcript=asr_event.text,
            )
            await call_state.enqueue_event(UserStoppedSpeaking())

            # Track ASR and Rasa latencies
            self._track_asr_latency()
            call_state.rasa_processing_start_time = time.time()

            if (
                call_state.is_collecting_dtmf
                and call_state.dtmf_config
                and not call_state.dtmf_config.allow_audio_input
            ):
                # currently collecting DTMF input, ignore audio input
                logger.info(
                    "VoiceInputChannel.handle_asr_event.ignoring_audio_during_dtmf_collection"
                )
                return

            output_channel = self.create_output_channel(voice_websocket, tts_engine)
            sender_id = self.get_sender_id(call_parameters)
            message = UserMessage(
                text=asr_event.text,
                output_channel=output_channel,
                sender_id=sender_id,
                input_channel=self.name(),
                metadata=asdict(call_parameters),
            )
            await on_new_message(message)
            await output_channel.send_turn_end_marker(sender_id)
        elif isinstance(asr_event, UserIsSpeaking):
            if not call_state.is_user_speaking:
                call_state.user_speech_start_time = time.time()
            await call_state.enqueue_event(UserIsSpeakingCallStateMessage())
        elif isinstance(asr_event, UserSilence):
            call_state.dtmf_buffer = ""
            output_channel = self.create_output_channel(voice_websocket, tts_engine)
            message = UserMessage(
                text=USER_CONVERSATION_SILENCE_TIMEOUT,
                output_channel=output_channel,
                sender_id=self.get_sender_id(call_parameters),
                input_channel=self.name(),
                metadata=asdict(call_parameters),
            )
            await on_new_message(message)

    async def gather_dtmf_input(
        self,
        channel_websocket: Websocket,
        tts_engine: TTSEngine,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        call_parameters: CallParameters,
        dtmf_action: DTMFInputAction,
    ) -> None:
        """Handle DTMF input gathering."""
        if not call_state.is_collecting_dtmf or not call_state.dtmf_config:
            return
        logger.debug(
            "voice_channel.dtmf_input_received",
            digit=dtmf_action.digit,
            buffer=call_state.dtmf_buffer,
        )
        call_state.dtmf_buffer += dtmf_action.digit

        # check completion criteria
        config = call_state.dtmf_config
        if config.length and len(call_state.dtmf_buffer) >= config.length:
            await self.submit_dtmf_input(
                channel_websocket,
                tts_engine,
                on_new_message,
                call_parameters,
                call_state.dtmf_buffer,
            )
        elif config.finish_on_key and dtmf_action.digit == config.finish_on_key:
            # remove the finish key from the buffer
            dtmf_input = call_state.dtmf_buffer[:-1]
            await self.submit_dtmf_input(
                channel_websocket,
                tts_engine,
                on_new_message,
                call_parameters,
                dtmf_input,
            )

    async def submit_dtmf_input(
        self,
        channel_websocket: Websocket,
        tts_engine: TTSEngine,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        call_parameters: CallParameters,
        dtmf_input: str,
    ) -> None:
        call_state.is_collecting_dtmf = False
        call_state.dtmf_buffer = ""
        output_channel = self.create_output_channel(channel_websocket, tts_engine)
        message = UserMessage(
            text=dtmf_input,
            output_channel=output_channel,
            sender_id=self.get_sender_id(call_parameters),
            input_channel=self.name(),
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
