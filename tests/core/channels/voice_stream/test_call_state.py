import asyncio
from typing import Any, Dict, Optional

import pytest

from rasa.core.channels.conversation_queue.events import (
    SilenceDetectedInputEvent,
    VoiceInputEvent,
)
from rasa.core.channels.conversation_queue.queue import InMemoryConversationQueue
from rasa.core.channels.voice_stream.call_state import (
    BotIsSpeaking,
    BotStoppedSpeaking,
    CallState,
    InterruptionConfig,
    RasaIsListening,
    RasaIsProcessing,
    UserIsSpeaking,
    UserStoppedSpeaking,
)
from rasa.shared.core.flows.steps.collect import DTMFConfig
from tests.core.channels.voice_stream.conftest import wait_for_task_to_become_cancelled


@pytest.fixture
def call_state_instance() -> CallState:
    """Return a CallState with fresh queues."""
    return CallState(
        internal_queue=asyncio.Queue(),
    )


def test_call_state_default_values(call_state_instance: CallState) -> None:
    """CallState initialises with sensible defaults."""

    assert call_state_instance.is_user_speaking is False
    assert call_state_instance.is_bot_speaking is False
    assert call_state_instance.is_rasa_listening is False
    assert call_state_instance.silence_timeout_watcher is None
    assert call_state_instance.silence_timeout is None
    assert call_state_instance.should_hangup is False
    assert call_state_instance.connection_failed is False
    assert call_state_instance.is_collecting_dtmf is False
    assert call_state_instance.dtmf_config is None
    assert call_state_instance.dtmf_buffer == ""
    assert call_state_instance.channel_data == {}
    assert call_state_instance.monitor_task is None
    assert call_state_instance.current_language is None
    assert call_state_instance.user_speech_start_time is None
    assert call_state_instance.rasa_processing_start_time is None
    assert call_state_instance.tts_start_time is None
    assert call_state_instance.asr_latency_ms is None
    assert call_state_instance.rasa_processing_latency_ms is None
    assert call_state_instance.tts_first_byte_latency_ms is None
    assert call_state_instance.tts_complete_latency_ms is None


async def test_monitor_silence_timeout_puts_silence_event_on_input_queue(
    call_state_instance: CallState,
) -> None:
    """monitor_silence_timeout puts a silence event on the input queue after
    the timeout elapses."""
    call_state_instance.silence_timeout = 0.01
    input_queue = InMemoryConversationQueue[VoiceInputEvent](
        conversation_id="test",
        input_channel="test_channel",
    )
    call_state_instance.input_queue = input_queue

    await call_state_instance.monitor_silence_timeout()

    events = await input_queue.drain()
    assert events == [SilenceDetectedInputEvent()]


async def test_monitor_silence_timeout_does_nothing_when_timeout_is_none(
    call_state_instance: CallState,
) -> None:
    """monitor_silence_timeout returns immediately without queuing anything when
    silence_timeout is None."""
    call_state_instance.silence_timeout = None

    await call_state_instance.monitor_silence_timeout()

    assert call_state_instance.input_queue is None


async def test_monitor_silence_timeout_does_nothing_when_timeout_is_zero(
    call_state_instance: CallState,
) -> None:
    """monitor_silence_timeout treats 0 as falsy and returns without queuing."""
    call_state_instance.silence_timeout = 0

    await call_state_instance.monitor_silence_timeout()

    assert call_state_instance.input_queue is None


async def test_start_silence_monitoring_creates_watcher_task(
    call_state_instance: CallState,
) -> None:
    """start_silence_monitoring creates a silence_timeout_watcher task."""
    call_state_instance.silence_timeout = 10

    call_state_instance.start_silence_monitoring()

    assert call_state_instance.silence_timeout_watcher is not None
    assert not call_state_instance.silence_timeout_watcher.done()

    # Cleanup
    call_state_instance.stop_silence_monitoring()


async def test_stop_silence_monitoring_cancels_watcher_and_sets_none(
    call_state_instance: CallState,
) -> None:
    """stop_silence_monitoring cancels the watcher task and sets it to None."""
    call_state_instance.silence_timeout = 10
    call_state_instance.start_silence_monitoring()
    assert call_state_instance.silence_timeout_watcher is not None

    call_state_instance.stop_silence_monitoring()

    assert call_state_instance.silence_timeout_watcher is None


async def test_stop_silence_monitoring_is_safe_when_no_watcher(
    call_state_instance: CallState,
) -> None:
    """stop_silence_monitoring is a no-op when there is no active watcher."""
    # Should not raise
    call_state_instance.stop_silence_monitoring()
    assert call_state_instance.silence_timeout_watcher is None


async def test_start_silence_monitoring_replaces_existing_watcher(
    call_state_instance: CallState,
) -> None:
    """Calling start_silence_monitoring a second time cancels the previous watcher
    and creates a fresh one."""
    call_state_instance.start_silence_monitoring()
    first_watcher = call_state_instance.silence_timeout_watcher

    call_state_instance.start_silence_monitoring()
    second_watcher = call_state_instance.silence_timeout_watcher

    assert first_watcher is not second_watcher
    # Task will either be cancelling or cancelled.
    # Because this is time-dependent we need to check if any of the statuses are set.

    await wait_for_task_to_become_cancelled(first_watcher)

    # Cleanup
    call_state_instance.stop_silence_monitoring()


async def test_put_message_on_input_channel_queue_adds_to_internal_queue(
    call_state_instance: CallState,
) -> None:
    """put_message_on_input_channel_queue enqueues the message on internal_queue."""
    message = RasaIsListening()

    await call_state_instance.enqueue_event(message)

    assert call_state_instance.internal_queue.qsize() == 1
    assert await call_state_instance.internal_queue.get() is message


async def test_monitor_sets_is_rasa_listening_on_rasa_is_listening(
    call_state_instance: CallState,
) -> None:
    """RasaIsListening sets is_rasa_listening to True."""
    call_state_instance.start_state_monitoring()

    call_state_instance.is_rasa_listening = False

    await call_state_instance.enqueue_event(RasaIsListening())
    await call_state_instance.internal_queue.join()

    assert call_state_instance.is_rasa_listening is True
    call_state_instance.stop_all()


async def test_monitor_clears_is_rasa_listening_on_rasa_is_processing(
    call_state_instance: CallState,
) -> None:
    """RasaIsProcessing sets is_rasa_listening to False."""
    call_state_instance.is_rasa_listening = True
    call_state_instance.start_state_monitoring()

    await call_state_instance.enqueue_event(RasaIsProcessing())
    await call_state_instance.internal_queue.join()

    assert call_state_instance.is_rasa_listening is False
    call_state_instance.stop_all()


async def test_monitor_sets_is_bot_speaking_on_bot_is_speaking(
    call_state_instance: CallState,
) -> None:
    """BotIsSpeaking sets is_bot_speaking to True."""
    call_state_instance.start_state_monitoring()
    call_state_instance.is_bot_speaking = False

    await call_state_instance.enqueue_event(BotIsSpeaking())
    await call_state_instance.internal_queue.join()

    assert call_state_instance.is_bot_speaking is True
    call_state_instance.stop_all()


async def test_monitor_clears_is_bot_speaking_on_bot_stopped_speaking(
    call_state_instance: CallState,
) -> None:
    """BotStoppedSpeaking sets is_bot_speaking to False."""
    call_state_instance.is_bot_speaking = True
    call_state_instance.start_state_monitoring()

    await call_state_instance.enqueue_event(BotStoppedSpeaking())
    await call_state_instance.internal_queue.join()

    assert call_state_instance.is_bot_speaking is False
    call_state_instance.stop_all()


async def test_monitor_sets_is_user_speaking_on_user_is_speaking(
    call_state_instance: CallState,
) -> None:
    """UserIsSpeaking sets is_user_speaking to True."""
    call_state_instance.start_state_monitoring()

    await call_state_instance.enqueue_event(UserIsSpeaking())
    await call_state_instance.internal_queue.join()

    assert call_state_instance.is_user_speaking is True
    call_state_instance.stop_all()


async def test_monitor_clears_is_user_speaking_on_user_stopped_speaking(
    call_state_instance: CallState,
) -> None:
    """UserStoppedSpeaking sets is_user_speaking to False and records
    user_speech_start_time."""
    call_state_instance.is_user_speaking = True
    call_state_instance.start_state_monitoring()

    await call_state_instance.enqueue_event(UserStoppedSpeaking())
    await call_state_instance.internal_queue.join()

    assert call_state_instance.is_user_speaking is False
    call_state_instance.stop_all()


async def test_monitor_starts_silence_monitoring_when_rasa_listening_and_idle(
    call_state_instance: CallState,
) -> None:
    """The monitor loop starts silence monitoring when rasa is listening and
    neither the bot nor the user is speaking."""
    call_state_instance.is_bot_speaking = False
    call_state_instance.is_user_speaking = False
    call_state_instance.start_state_monitoring()

    # RasaIsListening → all three conditions for start_silence_monitoring are met
    await call_state_instance.enqueue_event(RasaIsListening())
    await call_state_instance.internal_queue.join()

    assert call_state_instance.silence_timeout_watcher is not None
    call_state_instance.stop_all()


async def test_monitor_stops_silence_monitoring_when_bot_starts_speaking(
    call_state_instance: CallState,
) -> None:
    """The monitor loop stops silence monitoring when the bot starts speaking."""
    call_state_instance.is_rasa_listening = True
    call_state_instance.is_user_speaking = False
    call_state_instance.start_state_monitoring()
    # Prime the watcher
    call_state_instance.start_silence_monitoring()
    assert call_state_instance.silence_timeout_watcher is not None
    silence_watcher_task = call_state_instance.silence_timeout_watcher

    await call_state_instance.enqueue_event(BotIsSpeaking())
    await call_state_instance.internal_queue.join()

    assert call_state_instance.silence_timeout_watcher is None
    await wait_for_task_to_become_cancelled(silence_watcher_task)
    call_state_instance.stop_all()


async def test_monitor_stops_silence_monitoring_when_user_starts_speaking(
    call_state_instance: CallState,
) -> None:
    """The monitor loop stops silence monitoring when the user starts speaking."""
    call_state_instance.is_rasa_listening = True
    call_state_instance.is_bot_speaking = False
    call_state_instance.start_state_monitoring()
    call_state_instance.start_silence_monitoring()
    assert call_state_instance.silence_timeout_watcher is not None
    silence_watcher_task = call_state_instance.silence_timeout_watcher

    await call_state_instance.enqueue_event(UserIsSpeaking())
    await call_state_instance.internal_queue.join()

    assert call_state_instance.silence_timeout_watcher is None
    await wait_for_task_to_become_cancelled(silence_watcher_task)
    call_state_instance.stop_all()


async def test_start_monitoring_is_idempotent(
    call_state_instance: CallState,
) -> None:
    """Calling start_monitoring_to_input_channel_queue twice keeps the original task."""
    call_state_instance.start_state_monitoring()
    first_task = call_state_instance.monitor_task

    call_state_instance.start_state_monitoring()

    assert call_state_instance.monitor_task is first_task
    call_state_instance.stop_all()


async def test_stop_monitoring_cancels_monitor_task(
    call_state_instance: CallState,
) -> None:
    """stop_monitoring_to_input_channel_queue cancels the monitor task."""
    call_state_instance.start_state_monitoring()
    assert call_state_instance.monitor_task is not None

    monitor_task = call_state_instance.monitor_task

    call_state_instance.stop_signal_processing()

    # Because of the async nature of stopping the processing task we need to check
    # if the task is either in cancelled or cancelling call_state_instance.
    await wait_for_task_to_become_cancelled(monitor_task)


async def test_stop_monitoring_is_safe_when_no_task(
    call_state_instance: CallState,
) -> None:
    """stop_monitoring_to_input_channel_queue is a no-op when monitor_task is None."""
    # Should not raise
    call_state_instance.stop_signal_processing()


async def test_stop_all_cancels_monitor_task_and_silence_watcher(
    call_state_instance: CallState,
) -> None:
    """stop_all cancels both the monitor task and the silence watcher."""
    call_state_instance.start_state_monitoring()
    call_state_instance.start_silence_monitoring()

    assert call_state_instance.monitor_task is not None
    monitor_task = call_state_instance.monitor_task
    assert call_state_instance.silence_timeout_watcher is not None
    silence_watcher_task = call_state_instance.silence_timeout_watcher

    call_state_instance.stop_all()

    assert call_state_instance.monitor_task is None
    await wait_for_task_to_become_cancelled(monitor_task)
    assert call_state_instance.silence_timeout_watcher is None
    await wait_for_task_to_become_cancelled(silence_watcher_task)


async def test_stop_all_is_safe_when_nothing_running(
    call_state_instance: CallState,
) -> None:
    """stop_all is a no-op when neither monitor_task nor silence_timeout_watcher
    are set."""
    # Should not raise
    call_state_instance.stop_all()
    assert call_state_instance.monitor_task is None
    assert call_state_instance.silence_timeout_watcher is None


@pytest.mark.parametrize(
    "interruption_enabled, channel_data, expected",
    [
        # interruption_disabled
        (False, {}, False),
        # enabled_allow_interruptions_not_set
        (True, {}, True),
        # enabled_allow_interruptions_true
        (True, {"allow_interruptions": True}, True),
        # enabled_allow_interruptions_false
        (True, {"allow_interruptions": False}, False),
    ],
)
def test_is_interruptable(
    call_state_instance: CallState,
    interruption_enabled: bool,
    channel_data: Dict[str, Any],
    expected: bool,
) -> None:
    """is_interruptable returns True only when interruption is enabled and
    allow_interruptions is not explicitly False in channel_data."""
    call_state_instance.interruption_config = InterruptionConfig(
        enabled=interruption_enabled
    )
    call_state_instance.channel_data = channel_data

    assert call_state_instance.is_interruptable() is expected


@pytest.mark.parametrize(
    "is_collecting_dtmf, dtmf_config, expected",
    [
        # not_collecting_dtmf
        (False, DTMFConfig(allow_audio_input=False), False),
        # dtmf_config_is_none
        (True, None, False),
        # audio_input_allowed
        (True, DTMFConfig(allow_audio_input=True), False),
        # collecting_and_audio_disabled
        (True, DTMFConfig(allow_audio_input=False), True),
    ],
)
def test_can_collect_audio_during_dtmf(
    call_state_instance: CallState,
    is_collecting_dtmf: bool,
    dtmf_config: Optional[DTMFConfig],
    expected: bool,
) -> None:
    """can_collect_audio_during_dtmf returns True only when collecting DTMF and
    allow_audio_input is False."""
    call_state_instance.is_collecting_dtmf = is_collecting_dtmf
    call_state_instance.dtmf_config = dtmf_config

    assert call_state_instance.can_collect_audio_during_dtmf() is expected


@pytest.mark.parametrize(
    "is_rasa_listening, is_bot_speaking, expected",
    [
        # listening_and_bot_silent
        (True, False, True),
        # not_listening_and_bot_silent
        (False, False, False),
        # listening_but_bot_speaking
        (True, True, False),
        # not_listening_and_bot_speaking
        (False, True, False),
    ],
)
def test_can_receive_user_input(
    call_state_instance: CallState,
    is_rasa_listening: bool,
    is_bot_speaking: bool,
    expected: bool,
) -> None:
    """can_receive_user_input returns True only when rasa is listening and the
    bot is not speaking."""
    call_state_instance.is_rasa_listening = is_rasa_listening
    call_state_instance.is_bot_speaking = is_bot_speaking

    assert call_state_instance.can_receive_user_input() is expected
