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
    Dict,
    List,
    Optional,
    Text,
    Tuple,
)

import structlog
from pydantic import BaseModel
from sanic import Websocket  # type: ignore
from sanic.exceptions import WebsocketClosed

from rasa.core.channels.channel import (
    InputChannel,
    OutputChannel,
    RuntimeAgent,
)
from rasa.core.channels.conversation_queue.events import (
    BargeInInputEvent,
    DTMFInputEvent,
    FinalTranscriptInputEvent,
    SessionEndedInputEvent,
    SessionStartedInputEvent,
    SilenceDetectedInputEvent,
    VoiceInputEvent,
)
from rasa.core.channels.conversation_queue.queue import (
    ConversationQueue,
    InMemoryConversationQueue,
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
    InterruptionConfig,
    MarkerType,
    RasaIsListening,
    RasaIsProcessing,
    StepType,
    UserStoppedSpeaking,
    _call_state,
    call_state,
)
from rasa.core.channels.voice_stream.call_state import (
    UserIsSpeaking as UserIsSpeakingCallStateMessage,
)
from rasa.core.channels.voice_stream.tts import BUILT_IN_TTS_ENGINES
from rasa.core.channels.voice_stream.tts.tts_cache import TTSCache
from rasa.core.channels.voice_stream.tts.tts_engine import (
    StreamState,
    TTSEngine,
    TTSError,
)
from rasa.core.channels.voice_stream.util import generate_silence
from rasa.core.policies.flows.constants import COLLECT_STEP_TYPE, STEP_TYPE_METADATA_KEY
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
    from rasa.engine.storage.storage import ModelMetadata
    from rasa.shared.core.trackers import DialogueStateTracker

logger = structlog.get_logger(__name__)

# define constants for the voice channel
DEFAULT_MIN_DELAY_BETWEEN_BOT_MESSAGES_SECONDS = 1
DEFAULT_MIN_DELAY_AFTER_FILLER_BOT_MESSAGES_SECONDS = 2


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


class MarkerInput(BaseModel):
    recipient_id: str
    marker_type: MarkerType
    step_type: Optional[StepType] = None


@dataclass
class MarkerMessageOutput:
    message_id: str
    message: str


class VoiceOutputChannel(OutputChannel):
    def __init__(
        self,
        voice_websocket: Websocket,
        tts_engine: TTSEngine,
        tts_cache: TTSCache,
        audio_format: AudioFormat,
        min_buffer_size: int = 0,
        min_delay_between_bot_messages_seconds: float = (
            DEFAULT_MIN_DELAY_BETWEEN_BOT_MESSAGES_SECONDS
        ),
        min_delay_after_filler_seconds: float = (
            DEFAULT_MIN_DELAY_AFTER_FILLER_BOT_MESSAGES_SECONDS
        ),
    ):
        super().__init__()
        self.voice_websocket = voice_websocket
        self.tts_engine = tts_engine
        self.tts_cache = tts_cache
        self.min_buffer_size = min_buffer_size
        self.audio_format = audio_format
        self.min_delay_between_bot_messages_seconds = (
            min_delay_between_bot_messages_seconds
        )
        self.min_delay_after_filler_seconds = min_delay_after_filler_seconds

        self.latest_message_id: Optional[str] = None

        # For streaming responses - background task that sends TTS audio
        self.audio_sender_task: Optional[asyncio.Task] = None

        # When the last bot message ended (streaming or non-streaming). Used to
        # enforce a minimum pacing gap before the next bot message.
        self._last_bot_message_end_time: Optional[float] = None

        # Set by ReAct agents after each streamed LLM reply (see OutputChannel hook).
        self._last_completed_bot_message_was_filler: bool = False

        self.stream_interrupted = False

    def note_last_streamed_bot_message_was_filler(self, was_filler: bool) -> None:
        """Store filler classification for the next inter-message pacing decision."""
        self._last_completed_bot_message_was_filler = was_filler

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

    def create_marker_message(self, marker_input: MarkerInput) -> MarkerMessageOutput:
        """Create a marker message for a specific channel."""
        raise NotImplementedError

    async def send_marker_message(self, marker_input: MarkerInput) -> None:
        """Send a message that marks positions in the audio stream."""
        marker_message = self.create_marker_message(marker_input)
        await self._send_marker_message_via_websocket(
            marker_message.message_id, marker_message.message
        )

    async def _send_marker_message_via_websocket(
        self, mark_id: str, marker_message: str
    ) -> None:
        try:
            await self.voice_websocket.send(marker_message)
        except WebsocketClosed:
            call_state.connection_failed = True
        self.latest_message_id = mark_id

    async def send_start_marker(self, marker_input: MarkerInput) -> None:
        """Send a marker message before the first audio chunk."""
        # Default implementation uses the generic marker message
        await self.send_marker_message(marker_input)

    async def send_intermediate_marker(self, marker_input: MarkerInput) -> None:
        """Send a marker message during audio streaming."""
        await self.send_marker_message(marker_input)

    async def send_end_marker(self, marker_input: MarkerInput) -> None:
        """Send a marker message after the last audio chunk."""
        await self.send_marker_message(marker_input)

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

        self.tts_engine.stop_streaming_output_audio_chunks = call_state.is_user_speaking

        async for audio_chunk in audio_stream:
            collected_audio = collected_audio + audio_chunk

            if self.tts_engine.stop_streaming_output_audio_chunks:
                continue

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
                    await self.send_intermediate_marker(
                        MarkerInput(
                            recipient_id=recipient_id,
                            marker_type=MarkerType.INTERMEDIATE,
                        )
                    )
                    seconds_marker = full_seconds_of_audio
            except WebsocketClosed:
                call_state.connection_failed = True
                # Continue collecting for cache even if send fails

        # send any remaining bytes (including leftover)
        await self._send_remaining_bytes(
            recipient_id, collected_audio, last_sent_offset, leftover_byte
        )

        return collected_audio

    async def apply_pacing_delay(self, recipient_id: str, seconds: float) -> None:
        """Apply pacing delay by sending silence as audio (generic, all channels).

        Sends N seconds of silence on the same audio path as TTS to enforce
        a natural gap between consecutive bot utterances. No server sleep.
        """
        if seconds <= 0:
            return
        silence = generate_silence(self.audio_format, length_in_seconds=seconds)
        await self._stream_audio_to_channel(recipient_id, self.chunk_audio(silence))

    async def _apply_min_delay_between_messages(self, recipient_id: str) -> None:
        """Enforce minimum delay between consecutive bot messages.

        Uses a longer gap after a filler message (see
        :meth:`note_last_streamed_bot_message_was_filler`) and a shorter gap
        between other consecutive utterances. Resets ``_last_bot_message_end_time``.
        """
        min_gap_seconds = (
            self.min_delay_after_filler_seconds
            if self._last_completed_bot_message_was_filler
            else self.min_delay_between_bot_messages_seconds
        )
        if min_gap_seconds <= 0 or self._last_bot_message_end_time is None:
            self._last_bot_message_end_time = None
            return

        elapsed = time.monotonic() - self._last_bot_message_end_time
        wait_seconds = min_gap_seconds - elapsed
        if wait_seconds > 0:
            logger.debug(
                "voice_channel.apply_min_delay_between_messages",
                wait_seconds=wait_seconds,
                min_gap_seconds=min_gap_seconds,
                after_filler=self._last_completed_bot_message_was_filler,
            )
            await self.apply_pacing_delay(recipient_id, wait_seconds)
        self._last_bot_message_end_time = None

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
        Enforces min delay since last bot message (pacing silence if needed).
        """
        await self._apply_min_delay_between_messages(recipient_id)

        self.tts_engine.stream_state = StreamState.SENDING_RESPONSE_CHUNKS
        self.stream_interrupted = False

        await super().send_response_chunk_start(recipient_id, **kwargs)

        # Let TTS engine prepare for this response (e.g., mode selection)
        await self.tts_engine.prepare_response(
            streaming_config=kwargs.get("streaming_config")
        )

        if not self.tts_engine.streaming_input:
            # Engine does not support streaming input
            # fallback to non-streaming synthesis
            return

        await self.send_start_marker(
            MarkerInput(
                recipient_id=recipient_id,
                step_type=StepType.COLLECT
                if kwargs.get(STEP_TYPE_METADATA_KEY) == COLLECT_STEP_TYPE
                else StepType.REGULAR_UTTER,
                marker_type=MarkerType.START,
            )
        )

        self.audio_sender_task = asyncio.create_task(self._stream_tts(recipient_id))
        logger.debug("voice_channel.start_streaming_response")

    async def send_response_chunk(
        self, recipient_id: str, chunk: str, **kwargs: Any
    ) -> None:
        """Send text chunk to TTS.

        The TTS engine will process this and the background consumer task
        will receive the audio and send it to the websocket.
        """
        if self.tts_engine.stream_state == StreamState.INTERRUPTED:
            self.stream_interrupted = True
            return

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

        if self.tts_engine.stream_state == StreamState.INTERRUPTED:
            await self.tts_engine.signal_interrupt()
        else:
            await self.tts_engine.signal_text_done()
        self.tts_engine.stream_state = StreamState.RESPONSE_CHUNKS_SENT

        if self.audio_sender_task:
            await self.audio_sender_task

        await self.send_end_marker(
            MarkerInput(
                recipient_id=recipient_id,
                step_type=StepType.COLLECT
                if kwargs.get(STEP_TYPE_METADATA_KEY) == COLLECT_STEP_TYPE
                else StepType.REGULAR_UTTER,
                marker_type=MarkerType.END,
            )
        )

        self.tts_engine.stream_state = StreamState.NO_STREAMING
        self._last_bot_message_end_time = time.monotonic()
        logger.debug("voice_channel.end_streaming_response")

    async def send_text_message(
        self, recipient_id: str, text: str, **kwargs: Any
    ) -> None:
        if (
            self._is_duplicate_of_last_streamed_response(text)
            or self.stream_interrupted
        ):
            logger.debug("voice_channel.skip_non_streaming_response")
            return

        await self._apply_min_delay_between_messages(recipient_id)
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
        await self.send_start_marker(
            MarkerInput(
                recipient_id=recipient_id,
                step_type=StepType.COLLECT
                if kwargs.get(STEP_TYPE_METADATA_KEY) == COLLECT_STEP_TYPE
                else StepType.REGULAR_UTTER,
                marker_type=MarkerType.START,
            )
        )

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

        await self.send_end_marker(
            MarkerInput(
                recipient_id=recipient_id,
                step_type=StepType.COLLECT
                if kwargs.get(STEP_TYPE_METADATA_KEY) == COLLECT_STEP_TYPE
                else StepType.REGULAR_UTTER,
                marker_type=MarkerType.END,
            )
        )

        self._last_bot_message_end_time = time.monotonic()

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
        input_queue: ConversationQueue[VoiceInputEvent],
        call_parameters: CallParameters,
    ) -> None:
        """Start a voice session by enqueuing a SessionStartedInputEvent."""
        event = SessionStartedInputEvent(
            metadata=asdict(call_parameters),
        )
        await input_queue.put(event)

    async def map_input_message(
        self,
        message: Any,
        ws: Websocket,
    ) -> VoiceChannelAction:
        """Map a channel input message to a voice channel action."""
        raise NotImplementedError

    def should_interrupt(self, e: ASREvent) -> bool:
        """Determine if the current ASR event should interrupt bot playback.

        Returns True only when the bot is currently speaking, the bot response
        is interruptible, and the user spoke enough words.

        Arguments:
            e: The ASR event to evaluate.

        Returns:
            True if the event should interrupt playback, False otherwise.
        """
        if not call_state.is_bot_speaking:
            return False

        if not call_state.is_interruptable():
            return False

        min_words = self.interruption_config.min_words
        if isinstance(e, (NewTranscript, UserIsSpeaking)):
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
        if a channel does not implement it. It has no effect.
        """
        pass

    async def receive_asr_events(
        self,
        asr_engine: ASREngine,
        tts_engine: TTSEngine,
        ws: Websocket,
        call_parameters: CallParameters,
        input_queue: ConversationQueue[VoiceInputEvent],
    ) -> None:
        """Route ASR events as user input or barge-ins.

        While the bot is speaking and the current response allows interruption, ASR
        events are treated as possible barge-ins. A barge-in is only accepted when
        `should_interrupt()` sees enough spoken words. Shorter speech is ignored so
        partial words, backchannels, or background audio do not become user turns.

        If interruption handling is not active, ASR events are normal input and are
        passed to `handle_asr_event()`, which applies turn-specific checks such as
        accepting input during collect turns.
        """
        async for event in asr_engine.stream_asr_events():
            should_interrupt = self.should_interrupt(event)
            logger.debug(
                "voice_channel.receive_asr_events",
                ev=event,
                should_interrupt=should_interrupt,
            )

            if (
                call_state.is_interruptable()
                and call_state.is_bot_speaking
                and not should_interrupt
            ):
                continue

            if should_interrupt:
                call_state.stop_silence_monitoring()
                await tts_engine.stop_streaming()
                await self.interrupt_playback(ws, call_parameters)
                await input_queue.put(BargeInInputEvent())

            await self.handle_asr_event(
                event,
                input_queue,
                call_parameters,
            )

    async def asr_keep_alive_task(self, asr_engine: ASREngine) -> None:
        interval = getattr(asr_engine.config, "keep_alive_interval", 5)
        while True:
            await asyncio.sleep(interval)
            await asr_engine.send_keep_alive()

    def _initialize_call_state(self, model_metadata: Optional["ModelMetadata"]) -> None:
        call_state_ = CallState(
            internal_queue=asyncio.Queue(),
            interruption_config=self.interruption_config,
        )
        call_state_.start_state_monitoring()
        call_state_.current_language = (
            model_metadata.language
            if model_metadata and model_metadata.language
            else "en"
        )
        _call_state.set(call_state_)

    def _get_asr_and_tts_engines(
        self, model_metadata: Optional["ModelMetadata"]
    ) -> Tuple[ASREngine, TTSEngine]:
        language = (
            model_metadata.language
            if model_metadata and model_metadata.language
            else "en"
        )
        additional_languages = (
            model_metadata.additional_languages if model_metadata else None
        )
        asr_engine = asr_engine_from_config(
            asr_config=self.asr_config,
            format=self.audio_format,
            language=language,
            additional_languages=additional_languages,
        )
        tts_engine = tts_engine_from_config(
            tts_config=self.tts_config,
            format=self.audio_format,
            language=language,
            additional_languages=additional_languages,
        )
        return asr_engine, tts_engine

    async def run_audio_streaming(
        self,
        agent: RuntimeAgent,
        channel_websocket: Websocket,
        request: Optional[Any] = None,
    ) -> None:
        """Pipe input audio to ASR and consume ASR events simultaneously."""
        model_metadata = agent.model_metadata
        self._initialize_call_state(model_metadata)

        call_parameters = await self.collect_call_parameters(channel_websocket, request)
        if call_parameters is None:
            raise ValueError("Failed to extract call parameters for call.")

        # Initialize ASR and TTS based on config
        asr_engine, tts_engine = self._get_asr_and_tts_engines(model_metadata)

        # Connect both ASR and TTS at the beginning
        await asr_engine.connect()
        await tts_engine.connect()

        sender_id = self.get_sender_id(call_parameters)

        # Create input queue for this conversation
        input_queue: ConversationQueue[VoiceInputEvent] = InMemoryConversationQueue[
            VoiceInputEvent
        ](conversation_id=sender_id, input_channel=self.name(), maxsize=50)
        call_state.input_queue = input_queue
        logger.info(
            "voice_channel.input_queue_created",
            conversation_id=sender_id,
            call_id=call_parameters.call_id,
        )

        output_channel = self.create_output_channel(channel_websocket, tts_engine)
        language_plugin = VoiceLanguageChangePlugin(sender_id, asr_engine, tts_engine)
        tasks: List[asyncio.Task[Any]] = []

        try:
            language_plugin.register_hook()

            # Start the agent conversation handler
            agent_task = asyncio.create_task(
                agent.handle_conversation(input_queue, output_channel)
            )
            tasks.append(agent_task)

            # Send session start event
            await self.start_session(input_queue, call_parameters)

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
                                input_queue,
                                call_parameters,
                                channel_action,
                            )
                        elif isinstance(channel_action, EndConversationAction):
                            # end stream event came from the other side
                            is_disconnected = True
                            await self.handle_disconnect(
                                input_queue,
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
                            input_queue,
                            call_parameters,
                        )

            # Extend (not reassign) so agent_task stays in `tasks` for
            # asyncio.wait and finally-block cancellation.
            tasks.extend(
                [
                    asyncio.create_task(consume_audio_bytes()),
                    asyncio.create_task(
                        self.receive_asr_events(
                            asr_engine,
                            tts_engine,
                            channel_websocket,
                            call_parameters,
                            input_queue,
                        )
                    ),
                    asyncio.create_task(self.asr_keep_alive_task(asr_engine)),
                ]
            )
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
        input_queue: ConversationQueue[VoiceInputEvent],
        call_parameters: CallParameters,
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

            if call_state.can_collect_audio_during_dtmf():
                # currently collecting DTMF input, ignore audio input
                logger.info(
                    "VoiceInputChannel.handle_asr_event.ignoring_audio_during_dtmf_collection"
                )
                return

            if not call_state.can_queue_user_message():
                logger.info(
                    "VoiceInputChannel.handle_asr_event.ignoring_audio_during_regular_utterance",
                    conversation_id=input_queue.conversation_id,
                    transcript=asr_event.text,
                )
                call_state.current_bot_utterance_type = None
                return

            # Enqueue FinalTranscriptInputEvent
            event: VoiceInputEvent = FinalTranscriptInputEvent(
                text=asr_event.text,
                metadata=asdict(call_parameters),
            )
            await input_queue.put(event)
            call_state.current_bot_utterance_type = None
        elif isinstance(asr_event, UserIsSpeaking):
            if not call_state.is_user_speaking:
                call_state.user_speech_start_time = time.time()
            await call_state.enqueue_event(UserIsSpeakingCallStateMessage())
        elif isinstance(asr_event, UserSilence):
            call_state.dtmf_buffer = ""
            event = SilenceDetectedInputEvent(metadata=asdict(call_parameters))
            await input_queue.put(event)

    async def gather_dtmf_input(
        self,
        input_queue: ConversationQueue[VoiceInputEvent],
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
                input_queue,
                call_parameters,
                call_state.dtmf_buffer,
            )
        elif config.finish_on_key and dtmf_action.digit == config.finish_on_key:
            # remove the finish key from the buffer
            dtmf_input = call_state.dtmf_buffer[:-1]
            await self.submit_dtmf_input(
                input_queue,
                call_parameters,
                dtmf_input,
            )

    async def submit_dtmf_input(
        self,
        input_queue: ConversationQueue[VoiceInputEvent],
        call_parameters: CallParameters,
        dtmf_input: str,
    ) -> None:
        """Submit collected DTMF input to the queue."""
        call_state.is_collecting_dtmf = False
        call_state.dtmf_buffer = ""

        # Enqueue DTMFInputEvent
        event = DTMFInputEvent(text=dtmf_input)
        await input_queue.put(event)

    async def handle_disconnect(
        self,
        input_queue: ConversationQueue[VoiceInputEvent],
        call_parameters: CallParameters,
    ) -> None:
        """Handle disconnection from the channel."""
        event = SessionEndedInputEvent()
        await input_queue.put(event)
