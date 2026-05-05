from __future__ import annotations

import abc
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, cast

import structlog
from werkzeug.local import LocalProxy

from rasa.core.channels.voice_stream.asr.asr_event import UserSilence
from rasa.shared.core.flows.steps.collect import DTMFConfig

logger = structlog.get_logger(__name__)


@dataclass
class VoiceInputChannelMessage(abc.ABC):
    @abc.abstractmethod
    def process(self, call_state_input: CallState) -> None:
        pass


@dataclass
class RasaIsListening(VoiceInputChannelMessage):
    def process(self, call_state_input: CallState) -> None:
        call_state_input.is_rasa_listening = True


@dataclass
class RasaIsProcessing(VoiceInputChannelMessage):
    def process(self, call_state_input: CallState) -> None:
        call_state_input.is_rasa_listening = False


@dataclass
class BotIsSpeaking(VoiceInputChannelMessage):
    def process(self, call_state_input: CallState) -> None:
        call_state_input.is_bot_speaking = True


@dataclass
class BotStoppedSpeaking(VoiceInputChannelMessage):
    def process(self, call_state_input: CallState) -> None:
        call_state_input.is_bot_speaking = False


@dataclass
class UserIsSpeaking(VoiceInputChannelMessage):
    def process(self, call_state_input: CallState) -> None:
        call_state_input.is_user_speaking = True


@dataclass
class UserStoppedSpeaking(VoiceInputChannelMessage):
    def process(self, call_state_input: CallState) -> None:
        call_state_input.is_user_speaking = False


DEFAULT_INTERRUPTION_MIN_WORDS = 3


@dataclass
class InterruptionConfig:
    enabled: bool = False
    min_words: int = DEFAULT_INTERRUPTION_MIN_WORDS


# Per voice session data
# This is similar to how flask makes the "request" object available as a global variable
# It's a "global" variable that is local to an async task (i.e. websocket session)
@dataclass
class CallState:
    internal_queue: asyncio.Queue
    asr_event_queue: asyncio.Queue
    is_user_speaking: bool = False
    is_bot_speaking: bool = False
    is_rasa_listening: bool = False
    silence_timeout_watcher: Optional[asyncio.Task] = None
    silence_timeout: Optional[float] = None
    latest_bot_audio_id: Optional[str] = None
    should_hangup: bool = False
    connection_failed: bool = False

    interruption_config: InterruptionConfig = field(default_factory=InterruptionConfig)

    # Latency tracking - start times only
    user_speech_start_time: Optional[float] = None
    rasa_processing_start_time: Optional[float] = None
    tts_start_time: Optional[float] = None

    # Calculated latencies (used by channels like browser_audio)
    asr_latency_ms: Optional[float] = None
    rasa_processing_latency_ms: Optional[float] = None
    tts_first_byte_latency_ms: Optional[float] = None
    tts_complete_latency_ms: Optional[float] = None

    # DTMF State
    is_collecting_dtmf: bool = False
    dtmf_config: Optional[DTMFConfig] = None
    dtmf_buffer: str = ""

    # Generic field for channel-specific state data
    channel_data: Dict[str, Any] = field(default_factory=dict)
    monitor_task: Optional[asyncio.Task] = None

    def is_interruptable(self) -> bool:
        if not self.interruption_config.enabled:
            return False

        # Is the bot response interruptible?
        if not self.channel_data.get("allow_interruptions", True):
            return False

        return True

    def can_collect_audio_during_dtmf(self) -> bool:
        return (
            self.is_collecting_dtmf
            and self.dtmf_config is not None
            and not self.dtmf_config.allow_audio_input
        )

    def can_receive_user_input(self) -> bool:
        return self.is_rasa_listening and not self.is_bot_speaking

    async def monitor_silence_timeout(self) -> None:
        timeout = self.silence_timeout
        if not timeout:
            logger.debug(
                "voice_channel.silence_timeout_watcher.timeout_none",
                message="Silence timeout is not set. Skipping silence monitoring.",
            )
            return
        logger.debug("voice_channel.silence_timeout_watch_started", timeout=timeout)
        await asyncio.sleep(timeout)
        await self.asr_event_queue.put(UserSilence())
        logger.debug("voice_channel.silence_timeout_triggered", timeout=timeout)

    def start_silence_monitoring(self) -> None:
        logger.debug(
            "voice_channel.start_silence_monitoring",
            silence_timeout=self.silence_timeout,
        )
        self.stop_silence_monitoring()
        logger.debug("voice_channel.start_silence_monitoring.create_task")
        self.silence_timeout_watcher = asyncio.create_task(
            self.monitor_silence_timeout()
        )

    def stop_silence_monitoring(self) -> None:
        if self.silence_timeout_watcher:
            logger.debug("voice_channel.stop_silence_monitoring")
            self.silence_timeout_watcher.cancel()
            self.silence_timeout_watcher = None

    def start_state_monitoring(self) -> None:
        if self.monitor_task is not None:
            logger.debug(
                "voice_channel.call_state.start_state_monitoring.already_started"
            )
            return

        async def monitor_internal_queue() -> None:
            logger.debug("voice_channel.call_state.start_state_monitoring.starting")
            message = None

            while True:
                try:
                    message = await self.internal_queue.get()
                    if isinstance(message, VoiceInputChannelMessage):
                        message.process(self)

                    if (
                        self.is_rasa_listening
                        and not self.is_bot_speaking
                        and not self.is_user_speaking
                    ):
                        self.start_silence_monitoring()

                    if (
                        self.is_bot_speaking
                        or self.is_user_speaking
                        or not self.is_rasa_listening
                    ):
                        self.stop_silence_monitoring()
                except Exception as e:
                    logger.error(
                        "voice_channel.call_state.start_state_monitoring.exception",
                        exception=e,
                        message=message,
                    )
                finally:
                    if message:
                        self.internal_queue.task_done()
                        message = None

        self.monitor_task = asyncio.create_task(monitor_internal_queue())

    async def enqueue_event(self, message: VoiceInputChannelMessage) -> None:
        await self.internal_queue.put(message)

    def stop_signal_processing(self) -> None:
        if self.monitor_task is not None:
            logger.debug("voice_channel.call_state.stop_signal_processing")
            self.monitor_task.cancel()
            self.monitor_task = None

    def stop_all(self) -> None:
        self.stop_signal_processing()
        self.stop_silence_monitoring()

    # Language state - tracks the current language slot value
    current_language: Optional[str] = None


_call_state: ContextVar[CallState] = ContextVar("call_state")
call_state: CallState = cast(CallState, LocalProxy(_call_state))
