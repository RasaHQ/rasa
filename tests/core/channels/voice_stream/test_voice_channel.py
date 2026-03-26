import asyncio
import time
from dataclasses import asdict
from typing import Any, AsyncIterator, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch

from data.test_voice_channel.custom_asr_engine import CustomASREngine
from data.test_voice_channel.custom_tts_engine import CustomTTSEngine
from rasa.core.channels.constants import USER_CONVERSATION_SILENCE_TIMEOUT
from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.asr.asr_event import (
    ASREvent,
    NewTranscript,
    UserIsSpeaking,
    UserSilence,
)
from rasa.core.channels.voice_stream.audio_bytes import AudioFormat
from rasa.core.channels.voice_stream.call_state import (
    RasaIsListening,
    RasaIsProcessing,
    UserStoppedSpeaking,
    _call_state,
    call_state,
)
from rasa.core.channels.voice_stream.call_state import (
    UserIsSpeaking as UserIsSpeakingCallStateMessage,
)
from rasa.core.channels.voice_stream.tts.azure import AzureTTS
from rasa.core.channels.voice_stream.voice_channel import (
    DTMFInputAction,
    VoiceInputChannel,
    VoiceLanguageChangePlugin,
    VoiceOutputChannel,
    asr_engine_from_config,
    tts_engine_from_config,
)
from rasa.shared.constants import AZURE_SPEECH_API_KEY_ENV_VAR
from rasa.shared.core.flows.steps.collect import DTMFConfig


class StubVoiceInputChannel(VoiceInputChannel):
    """Minimal VoiceInputChannel stub for testing _on_asr_event_received.

    The test only calls _on_asr_event_received, should_interrupt, and
    interrupt_playback; inherited name() and other methods are unused.
    """

    pass


class StubVoiceOutputChannel(VoiceOutputChannel):
    """Minimal VoiceOutputChannel stub for unit tests.

    Implements the three abstract methods with no-op / trivial bodies so that
    the concrete channel machinery is not needed.
    """

    def rasa_audio_bytes_to_channel_bytes(self, rasa_audio_bytes: Any) -> bytes:
        return rasa_audio_bytes.data

    def channel_bytes_to_message(self, recipient_id: str, channel_bytes: bytes) -> str:
        return channel_bytes.hex()

    def create_marker_message(self, recipient_id: str):
        return "{}", "marker-id"


async def test_azure_tts_engine_from_config(
    mulaw_format: AudioFormat, monkeypatch: MonkeyPatch
):
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "some key")

    config = {"name": "azure"}
    tts_engine = tts_engine_from_config(config, language="en", format=mulaw_format)
    assert isinstance(tts_engine, AzureTTS)
    default_config = AzureTTS.get_default_config("en")
    assert tts_engine.config.speech_region == default_config.speech_region
    if tts_engine.session:
        await tts_engine.session.close()


def test_tts_engine_from_config_fails_for_not_implemented_engine(
    mulaw_format: AudioFormat,
):
    config = {"name": "XY_non_existent"}
    with pytest.raises(ImportError):
        tts_engine_from_config(config, language="en", format=mulaw_format)


def test_custom_asr_service(mulaw_format: AudioFormat) -> None:
    # Given a custom ASR engine
    config = {
        "name": "data.test_voice_channel.custom_asr_engine.CustomASREngine",
        "endpoint": "http://localhost:8000",
    }

    # When the ASR engine is created from the config
    asr_engine = asr_engine_from_config(config, language="en", format=mulaw_format)

    # Then the ASR engine should be an instance of the custom ASR engine
    assert isinstance(asr_engine, CustomASREngine)


def test_custom_tts_service(mulaw_format: AudioFormat) -> None:
    # Given a custom TTS engine
    config = {
        "name": "data.test_voice_channel.custom_tts_engine.CustomTTSEngine",
        "server_url": "http://localhost:8000",
    }

    # When the TTS engine is created from the config
    tts_engine = tts_engine_from_config(config, language="en", format=mulaw_format)

    # Then the TTS engine should be an instance of the custom TTS engine
    assert isinstance(tts_engine, CustomTTSEngine)


@pytest.mark.parametrize(
    "config,expected_error",
    [
        ({}, "ASR configuration dictionary cannot be empty"),
        (
            {"key": "value"},
            "ASR configuration must contain 'name' key specifying the engine type",
        ),
        (None, "ASR configuration dictionary cannot be empty"),
    ],
)
def test_asr_engine_config_validation(
    config: Optional[Dict], expected_error: str, mulaw_format: AudioFormat
):
    """Test validation of ASR engine configuration."""
    with pytest.raises(ValueError, match=expected_error):
        asr_engine_from_config(config, language="en", format=mulaw_format)


@pytest.mark.parametrize(
    "config, expected_error",
    [
        ({}, "TTS configuration dictionary cannot be empty"),
        (
            {"key": "value"},
            "TTS configuration must contain 'name' key specifying the engine type",
        ),
        (None, "TTS configuration dictionary cannot be empty"),
    ],
)
def test_tts_engine_config_validation(
    config: Optional[Dict], expected_error: str, mulaw_format: AudioFormat
):
    """Test validation of TTS engine configuration."""
    with pytest.raises(ValueError, match=expected_error):
        tts_engine_from_config(config, language="en", format=mulaw_format)


@pytest.mark.parametrize(
    "digit",
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "#", "*"],
)
def test_dtmf_input_action_all_digits(digit):
    """Test DTMFInputAction with all valid DTMF digits."""
    action = DTMFInputAction(digit=digit)
    assert action.digit == digit


@pytest.mark.usefixtures("setup_call_state")
def test_dtmf_config_in_call_state():
    """Test storing DTMF config in call state."""
    call_state = _call_state.get()

    config = DTMFConfig(length=6, allow_audio_input=False)
    call_state.dtmf_config = config
    call_state.is_collecting_dtmf = True

    assert call_state.dtmf_config is not None
    assert call_state.dtmf_config.length == 6
    assert call_state.dtmf_config.allow_audio_input is False
    assert call_state.is_collecting_dtmf is True


@pytest.mark.usefixtures("setup_call_state")
def test_dtmf_config_none_when_not_collecting():
    """Test that DTMF config can be None when not collecting."""
    call_state = _call_state.get()
    call_state.is_collecting_dtmf = False
    call_state.dtmf_config = None

    assert call_state.dtmf_config is None
    assert call_state.is_collecting_dtmf is False


@pytest.fixture
def call_parameters() -> CallParameters:
    return CallParameters(
        call_id="call_123",
        user_phone="+123",
        bot_phone="+456",
        stream_id="stream_456",
        direction="inbound",
    )


def create_stub_voice_input_channel(
    interruption_config: Dict[str, Any],
) -> StubVoiceInputChannel:
    return StubVoiceInputChannel(
        server_url="https://example.com",
        asr_config={"name": "azure"},
        tts_config={"name": "azure"},
        interruptions=interruption_config,
    )


def _make_mock_asr_engine(events: list) -> MagicMock:
    """Create a mock ASR engine that yields the given events from stream_asr_events."""

    async def _stream() -> AsyncIterator[ASREvent]:
        for event in events:
            yield event

    mock_asr_engine = MagicMock()
    mock_asr_engine.stream_asr_events = _stream
    return mock_asr_engine


def _make_channel_and_mocks(
    mock_validate_voice_license_scope: Any,
    interruption_config: Dict[str, Any],
) -> tuple:
    """Create channel test doubles.

    Returns a stub channel with `interrupt_playback` mocked and a mocked TTS engine.
    """
    channel = create_stub_voice_input_channel(interruption_config)

    channel.interrupt_playback = AsyncMock()

    mock_tts_engine = MagicMock()
    mock_tts_engine.stop_streaming = AsyncMock()

    return (
        channel,
        mock_tts_engine,
    )


@pytest.mark.parametrize(
    "asr_event",
    [
        NewTranscript(text="one two three"),
        UserIsSpeaking(text="one two three"),
    ],
    ids=["NewTranscript", "UserIsSpeaking"],
)
@pytest.mark.parametrize(
    "allow_interruptions_dict",
    [
        {},
        {"allow_interruptions": True},
    ],
)
@pytest.mark.usefixtures("setup_call_state")
async def test_stop_streaming_and_interrupt_playback_on_interruption(
    allow_interruptions_dict: Dict[str, bool],
    mock_validate_voice_license_scope,
    asr_event: ASREvent,
    call_parameters: CallParameters,
) -> None:
    """Test that ASR event triggers interruption when its text is above threshold."""
    # Given a channel with interruptions enabled and the bot currently speaking

    mock_web_socket = MagicMock()

    channel, mock_tts_engine = _make_channel_and_mocks(
        mock_validate_voice_license_scope,
        interruption_config={"enabled": True, "min_words": 3},
    )
    call_state.channel_data.update(allow_interruptions_dict)
    call_state.is_bot_speaking = True

    asr_event_queue: asyncio.Queue = asyncio.Queue()
    mock_asr_engine = _make_mock_asr_engine([asr_event])

    # set silence timeout to a big value to ensure that its async.Task will
    # be running through duration of the test
    call_state.silence_timeout = 120
    call_state.start_silence_monitoring()

    silence_timeout_watcher = call_state.silence_timeout_watcher

    # make sure that silence timeout watcher is running
    assert silence_timeout_watcher is not None

    # When receive_asr_events processes the event
    await channel.receive_asr_events(
        asr_engine=mock_asr_engine,
        tts_engine=mock_tts_engine,
        asr_event_queue=asr_event_queue,
        ws=mock_web_socket,
        call_parameters=call_parameters,
    )

    # Then the event is queued
    assert asr_event_queue.qsize() == 1
    assert await asr_event_queue.get() is asr_event

    # And stop_streaming and interrupt_playback are called once
    assert call_state.silence_timeout_watcher is None
    mock_tts_engine.stop_streaming.assert_awaited_once()
    channel.interrupt_playback.assert_awaited_once_with(
        mock_web_socket, call_parameters
    )


@pytest.mark.parametrize(
    "allow_interruptions_dict",
    [
        {},
        {"allow_interruptions": True},
    ],
)
@pytest.mark.usefixtures("setup_call_state")
async def test_receive_asr_events_does_not_interrupt_when_words_below_threshold(
    allow_interruptions_dict: Dict[str, bool],
    mock_validate_voice_license_scope,
    call_parameters: CallParameters,
) -> None:
    """Test that interruptions do not trigger when word count is below threshold."""
    # Given events with fewer words than the min_words threshold (3)

    mock_web_socket = MagicMock()
    channel, mock_tts_engine = _make_channel_and_mocks(
        mock_validate_voice_license_scope,
        interruption_config={"enabled": True, "min_words": 3},
    )
    call_state.channel_data = allow_interruptions_dict
    call_state.is_bot_speaking = True

    events = [
        NewTranscript(text="one two"),
        UserIsSpeaking(text="hi"),
    ]
    asr_event_queue: asyncio.Queue = asyncio.Queue()
    mock_asr_engine = _make_mock_asr_engine(events)

    # When receive_asr_events processes the events
    await channel.receive_asr_events(
        asr_engine=mock_asr_engine,
        tts_engine=mock_tts_engine,
        asr_event_queue=asr_event_queue,
        ws=mock_web_socket,
        call_parameters=call_parameters,
    )

    # Then neither stop_streaming nor interrupt_playback are called
    mock_tts_engine.stop_streaming.assert_not_awaited()
    channel.interrupt_playback.assert_not_awaited()

    # But the events are still queued
    assert asr_event_queue.qsize() == 2


@pytest.mark.parametrize(
    "allow_interruptions_dict",
    [
        {},
        {"allow_interruptions": True},
    ],
)
@pytest.mark.usefixtures("setup_call_state")
async def test_interruptions_for_multiple_asr_events_in_sequence(
    allow_interruptions_dict: Dict[str, bool],
    mock_validate_voice_license_scope,
    call_parameters: CallParameters,
) -> None:
    """Tests that multiple consecutive ASR events triggers interruptions."""

    mock_websocket = MagicMock()
    # Given events with fewer words than the min_words threshold (3)
    channel, mock_tts_engine = _make_channel_and_mocks(
        mock_validate_voice_license_scope,
        interruption_config={"enabled": True, "min_words": 3},
    )
    call_state.channel_data = allow_interruptions_dict
    call_state.is_bot_speaking = True

    interruptible_events = [
        NewTranscript(text="one two three"),
        UserIsSpeaking(text="one two three four"),
    ]
    asr_event_queue: asyncio.Queue = asyncio.Queue()
    mock_asr_engine = _make_mock_asr_engine(interruptible_events)

    # When receive_asr_events processes both events
    await channel.receive_asr_events(
        asr_engine=mock_asr_engine,
        tts_engine=mock_tts_engine,
        asr_event_queue=asr_event_queue,
        ws=mock_websocket,
        call_parameters=call_parameters,
    )

    # Then stop_streaming and interrupt_playback are called once per interruptible event
    assert mock_tts_engine.stop_streaming.await_count == 2
    assert channel.interrupt_playback.await_count == 2
    channel.interrupt_playback.assert_has_calls(
        [call(mock_websocket, call_parameters)] * 2
    )


@pytest.mark.parametrize(
    "allow_interruptions_dict",
    [
        {},
        {"allow_interruptions": True},
    ],
)
@pytest.mark.usefixtures("setup_call_state")
async def test_interruptions_not_firing_when_disabled(
    allow_interruptions_dict: Dict[str, bool],
    mock_validate_voice_license_scope,
    call_parameters: CallParameters,
) -> None:
    """Test that interruptions are not firing if they are disabled."""
    # Given events with fewer words than the min_words threshold (3)

    mock_web_socket = MagicMock()
    channel, mock_tts_engine = _make_channel_and_mocks(
        mock_validate_voice_license_scope,
        interruption_config={"enabled": False, "min_words": 3},
    )
    call_state.channel_data = allow_interruptions_dict
    call_state.is_bot_speaking = True

    events = [
        NewTranscript(text="one two"),
        UserIsSpeaking(text="hi"),
    ]
    asr_event_queue: asyncio.Queue = asyncio.Queue()
    mock_asr_engine = _make_mock_asr_engine(events)

    # When receive_asr_events processes the events
    await channel.receive_asr_events(
        asr_engine=mock_asr_engine,
        tts_engine=mock_tts_engine,
        asr_event_queue=asr_event_queue,
        ws=mock_web_socket,
        call_parameters=call_parameters,
    )

    # Then neither stop_streaming nor interrupt_playback are called
    mock_tts_engine.stop_streaming.assert_not_awaited()
    channel.interrupt_playback.assert_not_awaited()

    # But the events are still queued
    assert asr_event_queue.qsize() == 2


@pytest.mark.usefixtures("setup_call_state")
async def test_language_change_hook_schedules_update() -> None:
    call_state.current_language = "en-US"
    asr_engine = MagicMock()
    tts_engine = MagicMock()
    asr_engine.set_language = AsyncMock()
    tts_engine.set_language = AsyncMock()

    plugin = VoiceLanguageChangePlugin(
        sender_id="sender-1", asr_engine=asr_engine, tts_engine=tts_engine
    )
    language_slot = MagicMock()
    language_slot.value = "de-DE"
    tracker = MagicMock(sender_id="sender-1", slots={"language": language_slot})

    awaitable = plugin.after_action_executed(tracker)
    assert awaitable is not None
    await awaitable

    asr_engine.set_language.assert_awaited_once_with("de-DE")
    tts_engine.set_language.assert_awaited_once_with("de-DE")
    assert call_state.current_language == "de-DE"


@pytest.mark.usefixtures("setup_call_state")
async def test_language_change_hook_ignores_other_senders() -> None:
    call_state.current_language = "en-US"
    asr_engine = MagicMock()
    tts_engine = MagicMock()
    asr_engine.set_language = AsyncMock()
    tts_engine.set_language = AsyncMock()

    plugin = VoiceLanguageChangePlugin(
        sender_id="sender-1", asr_engine=asr_engine, tts_engine=tts_engine
    )
    language_slot = MagicMock()
    language_slot.value = "de-DE"
    tracker = MagicMock(sender_id="sender-2", slots={"language": language_slot})

    awaitable = plugin.after_action_executed(tracker)
    assert awaitable is None

    asr_engine.set_language.assert_not_awaited()
    tts_engine.set_language.assert_not_awaited()
    assert call_state.current_language == "en-US"


async def test_language_change_hook_without_call_state_context() -> None:
    asr_engine = MagicMock()
    tts_engine = MagicMock()
    asr_engine.set_language = AsyncMock()
    tts_engine.set_language = AsyncMock()

    plugin = VoiceLanguageChangePlugin(
        sender_id="sender-1", asr_engine=asr_engine, tts_engine=tts_engine
    )
    language_slot = MagicMock()
    language_slot.value = "de-DE"
    tracker = MagicMock(sender_id="sender-1", slots={"language": language_slot})

    # Simulate non-voice channel execution where `_call_state` is unbound.
    token = _call_state.set(None)
    try:
        awaitable = plugin.after_action_executed(tracker)
        assert awaitable is not None
        await awaitable
    finally:
        _call_state.reset(token)

    asr_engine.set_language.assert_awaited_once_with("de-DE")
    tts_engine.set_language.assert_awaited_once_with("de-DE")


@pytest.mark.usefixtures("mock_validate_voice_license_scope")
async def test_run_audio_streaming_unregisters_language_plugin_on_session_error() -> (
    None
):
    channel = create_stub_voice_input_channel(interruption_config={"enabled": True})
    mock_websocket = MagicMock()
    mock_websocket.close = AsyncMock()

    call_params = CallParameters(
        call_id="call_123",
        user_phone="+123",
        bot_phone="+456",
        stream_id="stream_456",
        direction="inbound",
    )

    asr_engine = MagicMock()
    asr_engine.connect = AsyncMock()
    asr_engine.close_connection = AsyncMock()
    tts_engine = MagicMock()
    tts_engine.connect = AsyncMock()
    tts_engine.close_connection = AsyncMock()

    channel.collect_call_parameters = AsyncMock(return_value=call_params)
    channel._get_asr_and_tts_engines = MagicMock(return_value=(asr_engine, tts_engine))
    channel.start_session = AsyncMock(side_effect=RuntimeError("start session failed"))

    with patch(
        "rasa.core.channels.voice_stream.voice_channel.VoiceLanguageChangePlugin"
    ) as plugin_cls:
        mock_plugin = MagicMock()
        mock_plugin.register_hook = MagicMock()
        mock_plugin.unregister_hook = AsyncMock()
        plugin_cls.return_value = mock_plugin

        with pytest.raises(RuntimeError, match="start session failed"):
            await channel.run_audio_streaming(AsyncMock(), mock_websocket)

    mock_plugin.register_hook.assert_called_once()
    mock_plugin.unregister_hook.assert_awaited_once()
    asr_engine.close_connection.assert_awaited_once()
    tts_engine.close_connection.assert_awaited_once()
    mock_websocket.close.assert_awaited_once()


# --- apply_pacing_delay and _apply_min_delay_between_messages ---


@pytest.fixture
def stub_output_channel(mulaw_format) -> StubVoiceOutputChannel:
    """Create a stub VoiceOutputChannel for testing pacing delay methods."""
    mock_ws = MagicMock()
    mock_tts = MagicMock()
    mock_tts.streaming_input = True
    return StubVoiceOutputChannel(
        voice_websocket=mock_ws,
        tts_engine=mock_tts,
        tts_cache={},
        audio_format=mulaw_format,
        min_delay_between_bot_messages_seconds=1.5,
        min_delay_after_filler_seconds=1.5,
    )


@pytest.mark.parametrize("seconds", [0, -0.1])
async def test_apply_pacing_delay_does_nothing_when_seconds_non_positive(
    stub_output_channel: StubVoiceOutputChannel,
    seconds: float,
) -> None:
    """When seconds <= 0, apply_pacing_delay does not stream any audio."""
    with patch.object(
        stub_output_channel,
        "_stream_audio_to_channel",
        new_callable=AsyncMock,
    ) as mock_stream:
        await stub_output_channel.apply_pacing_delay("recipient_1", seconds)
        mock_stream.assert_not_awaited()


async def test_apply_pacing_delay_streams_silence_when_seconds_positive(
    stub_output_channel: StubVoiceOutputChannel,
    mulaw_format,
) -> None:
    """When seconds > 0, apply_pacing_delay streams silence to the channel."""
    with patch.object(
        stub_output_channel,
        "_stream_audio_to_channel",
        new_callable=AsyncMock,
    ) as mock_stream:
        await stub_output_channel.apply_pacing_delay("recipient_1", 2.0)
        mock_stream.assert_awaited_once()
        call_args = mock_stream.await_args
        assert call_args[0][0] == "recipient_1"
        # Second arg is async generator from chunk_audio(silence); consume it
        stream = call_args[0][1]
        chunks = []
        async for ch in stream:
            chunks.append(ch)
        assert len(chunks) >= 1
        total_bytes = sum(len(c.data) for c in chunks)
        expected_bytes = int(2.0 * mulaw_format.bytes_per_second)
        assert total_bytes == expected_bytes


async def test_apply_min_delay_between_messages_does_nothing_when_min_delay_zero(
    stub_output_channel: StubVoiceOutputChannel,
) -> None:
    """When both pacing delays <= 0, no pacing delay is applied."""
    stub_output_channel.min_delay_after_filler_seconds = 0
    stub_output_channel.min_delay_between_bot_messages_seconds = 0
    stub_output_channel._last_bot_message_end_time = time.monotonic()
    with patch.object(
        stub_output_channel,
        "apply_pacing_delay",
        new_callable=AsyncMock,
    ) as mock_apply:
        await stub_output_channel._apply_min_delay_between_messages("recipient_1")
        mock_apply.assert_not_awaited()
    assert stub_output_channel._last_bot_message_end_time is None


async def test_apply_min_delay_between_messages_does_nothing_when_last_end_time_none(
    stub_output_channel: StubVoiceOutputChannel,
) -> None:
    """When _last_bot_message_end_time is None, no pacing delay is applied."""
    stub_output_channel._last_bot_message_end_time = None
    with patch.object(
        stub_output_channel,
        "apply_pacing_delay",
        new_callable=AsyncMock,
    ) as mock_apply:
        await stub_output_channel._apply_min_delay_between_messages("recipient_1")
        mock_apply.assert_not_awaited()
    assert stub_output_channel._last_bot_message_end_time is None


async def test_apply_min_delay_between_messages_no_apply_when_elapsed_exceeds_min(
    stub_output_channel: StubVoiceOutputChannel,
) -> None:
    """When enough time passed since last message, no extra silence is sent."""
    stub_output_channel._last_bot_message_end_time = time.monotonic() - 5.0
    with patch.object(
        stub_output_channel,
        "apply_pacing_delay",
        new_callable=AsyncMock,
    ) as mock_apply:
        await stub_output_channel._apply_min_delay_between_messages("recipient_1")
        mock_apply.assert_not_awaited()
    assert stub_output_channel._last_bot_message_end_time is None


async def test_apply_min_delay_between_messages_calls_apply_pacing_delay_and_resets(
    stub_output_channel: StubVoiceOutputChannel,
) -> None:
    """apply_pacing_delay is called and _last_bot_message_end_time reset."""
    stub_output_channel.note_last_streamed_bot_message_was_filler(True)
    stub_output_channel._last_bot_message_end_time = time.monotonic() - 0.5
    with patch.object(
        stub_output_channel,
        "apply_pacing_delay",
        new_callable=AsyncMock,
    ) as mock_apply:
        await stub_output_channel._apply_min_delay_between_messages("recipient_1")
        mock_apply.assert_awaited_once()
        awaited_seconds = mock_apply.await_args[0][1]
        assert 0.9 <= awaited_seconds <= 1.1
    assert stub_output_channel._last_bot_message_end_time is None


@pytest.mark.parametrize(
    (
        "was_filler",
        "min_between",
        "min_after_filler",
        "elapsed",
        "expected_low",
        "expected_high",
    ),
    [
        (False, 0.5, 2.0, 0.1, 0.35, 0.45),
        (True, 0.4, 2.5, 0.1, 2.35, 2.45),
    ],
    ids=[
        "after_non_filler_uses_min_between",
        "after_filler_uses_min_after_filler",
    ],
)
async def test_apply_min_delay_between_messages_depends_on_filler_flag(
    stub_output_channel: StubVoiceOutputChannel,
    was_filler: bool,
    min_between: float,
    min_after_filler: float,
    elapsed: float,
    expected_low: float,
    expected_high: float,
) -> None:
    stub_output_channel.min_delay_between_bot_messages_seconds = min_between
    stub_output_channel.min_delay_after_filler_seconds = min_after_filler
    stub_output_channel.note_last_streamed_bot_message_was_filler(was_filler)
    stub_output_channel._last_bot_message_end_time = time.monotonic() - elapsed
    with patch.object(
        stub_output_channel,
        "apply_pacing_delay",
        new_callable=AsyncMock,
    ) as mock_apply:
        await stub_output_channel._apply_min_delay_between_messages("recipient_1")
        mock_apply.assert_awaited_once()
        awaited_seconds = mock_apply.await_args[0][1]
    assert expected_low <= awaited_seconds <= expected_high


@pytest.fixture
def stub_voice_input_channel(
    mock_validate_voice_license_scope,
) -> StubVoiceInputChannel:
    """Return a StubVoiceInputChannel with a mocked output channel."""
    channel = StubVoiceInputChannel(
        server_url="https://example.com",
        asr_config={"name": "deepgram"},
        tts_config={"name": "azure"},
    )
    return channel


def _attach_mock_output_channel(channel: StubVoiceInputChannel) -> MagicMock:
    """Replace create_output_channel with a mock that returns a mock output channel."""
    mock_output_channel = MagicMock()
    mock_output_channel.send_turn_end_marker = AsyncMock()
    mock_output_channel.check_language_change = MagicMock(return_value=None)
    channel.create_output_channel = MagicMock(return_value=mock_output_channel)
    return mock_output_channel


@pytest.mark.usefixtures("mock_validate_voice_license_scope", "setup_call_state")
async def test_handle_asr_event_new_transcript_calls_on_new_message(
    call_parameters: CallParameters,
    stub_voice_input_channel: StubVoiceInputChannel,
):
    """NewTranscript with non-empty text triggers on_new_message and
    send_turn_end_marker.
    """

    mock_output_channel = _attach_mock_output_channel(stub_voice_input_channel)

    voice_websocket = MagicMock()
    on_new_message = AsyncMock()
    tts_engine = MagicMock()
    asr_engine = MagicMock()
    asr_engine.set_language = AsyncMock()

    await stub_voice_input_channel.handle_asr_event(
        asr_event=NewTranscript(text="hello world"),
        voice_websocket=voice_websocket,
        on_new_message=on_new_message,
        tts_engine=tts_engine,
        call_parameters=call_parameters,
        asr_engine=asr_engine,
    )

    on_new_message.assert_awaited_once()
    sent_message = on_new_message.call_args[0][0]
    assert sent_message.text == "hello world"
    assert sent_message.sender_id == call_parameters.call_id
    mock_output_channel.send_turn_end_marker.assert_awaited_once_with(
        call_parameters.call_id
    )
    # No language change → set_language should NOT be called
    asr_engine.set_language.assert_not_awaited()
    message = call_state.internal_queue.get_nowait()
    assert isinstance(message, UserStoppedSpeaking)


@pytest.mark.usefixtures("mock_validate_voice_license_scope", "setup_call_state")
async def test_handle_asr_event_new_transcript_empty_text_ignored(
    call_parameters: CallParameters,
    stub_voice_input_channel: StubVoiceInputChannel,
):
    """NewTranscript with empty text should not call on_new_message."""

    _attach_mock_output_channel(stub_voice_input_channel)

    on_new_message = AsyncMock()
    asr_engine = MagicMock()
    asr_engine.set_language = AsyncMock()

    await stub_voice_input_channel.handle_asr_event(
        asr_event=NewTranscript(text=""),
        voice_websocket=MagicMock(),
        on_new_message=on_new_message,
        tts_engine=MagicMock(),
        call_parameters=call_parameters,
        asr_engine=asr_engine,
    )

    on_new_message.assert_not_awaited()
    asr_engine.set_language.assert_not_awaited()
    assert call_state.internal_queue.qsize() == 0


@pytest.mark.usefixtures("mock_validate_voice_license_scope", "setup_call_state")
async def test_handle_asr_event_new_transcript_ignored_during_dtmf_collection(
    call_parameters: CallParameters,
    stub_voice_input_channel: StubVoiceInputChannel,
):
    """NewTranscript is silently ignored while collecting DTMF and
    allow_audio_input is False.
    """

    _attach_mock_output_channel(stub_voice_input_channel)

    # Simulate active DTMF collection that blocks audio input
    call_state.is_collecting_dtmf = True
    call_state.dtmf_config = DTMFConfig(length=4, allow_audio_input=False)

    on_new_message = AsyncMock()
    asr_engine = MagicMock()
    asr_engine.set_language = AsyncMock()

    await stub_voice_input_channel.handle_asr_event(
        asr_event=NewTranscript(text="some words"),
        voice_websocket=MagicMock(),
        on_new_message=on_new_message,
        tts_engine=MagicMock(),
        call_parameters=call_parameters,
        asr_engine=asr_engine,
    )

    on_new_message.assert_not_awaited()
    asr_engine.set_language.assert_not_awaited()
    message = call_state.internal_queue.get_nowait()
    assert isinstance(message, UserStoppedSpeaking)


@pytest.mark.usefixtures("mock_validate_voice_license_scope", "setup_call_state")
async def test_handle_asr_event_new_transcript_allowed_when_dtmf_allows_audio(
    call_parameters: CallParameters,
    stub_voice_input_channel: StubVoiceInputChannel,
):
    """NewTranscript is processed normally when allow_audio_input is True,
    even while collecting DTMF.
    """

    _attach_mock_output_channel(stub_voice_input_channel)

    call_state.is_collecting_dtmf = True
    call_state.dtmf_config = DTMFConfig(length=4, allow_audio_input=True)

    on_new_message = AsyncMock()
    asr_engine = MagicMock()
    asr_engine.set_language = AsyncMock()

    await stub_voice_input_channel.handle_asr_event(
        asr_event=NewTranscript(text="hello"),
        voice_websocket=MagicMock(),
        on_new_message=on_new_message,
        tts_engine=MagicMock(),
        call_parameters=call_parameters,
        asr_engine=asr_engine,
    )

    on_new_message.assert_awaited_once()
    message = call_state.internal_queue.get_nowait()
    assert isinstance(message, UserStoppedSpeaking)


@pytest.mark.usefixtures("mock_validate_voice_license_scope", "setup_call_state")
async def test_handle_asr_event_user_silence_sends_silence_timeout_message(
    call_parameters: CallParameters,
    stub_voice_input_channel: StubVoiceInputChannel,
):
    """UserSilence event calls on_new_message with USER_CONVERSATION_SILENCE_TIMEOUT
    and clears the DTMF buffer.
    """

    mock_output_channel = _attach_mock_output_channel(stub_voice_input_channel)

    call_state.dtmf_buffer = "123"
    on_new_message = AsyncMock()

    await stub_voice_input_channel.handle_asr_event(
        asr_event=UserSilence(),
        voice_websocket=MagicMock(),
        on_new_message=on_new_message,
        tts_engine=MagicMock(),
        call_parameters=call_parameters,
        asr_engine=MagicMock(),
    )

    on_new_message.assert_awaited_once()
    sent_message = on_new_message.call_args[0][0]
    assert sent_message.text == USER_CONVERSATION_SILENCE_TIMEOUT
    assert sent_message.sender_id == call_parameters.call_id
    assert sent_message.output_channel == mock_output_channel
    assert sent_message.input_channel == stub_voice_input_channel.name()
    assert sent_message.metadata == asdict(call_parameters)
    assert call_state.dtmf_buffer == ""


@pytest.mark.usefixtures("mock_validate_voice_license_scope", "setup_call_state")
async def test_handle_asr_event_new_transcript_puts_user_stopped_speaking_in_queue(
    call_parameters: CallParameters,
    stub_voice_input_channel: StubVoiceInputChannel,
):
    """UserIsSpeaking puts UserIsSpeakingCallStateMessage into the
    internal queue.
    """

    on_new_message = AsyncMock()

    await stub_voice_input_channel.handle_asr_event(
        asr_event=UserIsSpeaking(text="hello"),
        voice_websocket=MagicMock(),
        on_new_message=on_new_message,
        tts_engine=MagicMock(),
        call_parameters=call_parameters,
        asr_engine=MagicMock(),
    )

    message = call_state.internal_queue.get_nowait()
    assert isinstance(message, UserIsSpeakingCallStateMessage)


# ---------------------------------------------------------------------------
# Fixtures and tests for VoiceOutputChannel.notify_message_processing_*
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_voice_output_channel(mulaw_format: AudioFormat) -> StubVoiceOutputChannel:
    """Return a StubVoiceOutputChannel with mocked websocket and TTS engine."""
    return StubVoiceOutputChannel(
        voice_websocket=MagicMock(),
        tts_engine=MagicMock(),
        tts_cache=MagicMock(),
        audio_format=mulaw_format,
    )


@pytest.mark.usefixtures("setup_call_state")
async def test_notify_message_processing_started_enqueues_rasa_is_processing(
    stub_voice_output_channel: StubVoiceOutputChannel,
):
    """notify_message_processing_started puts RasaIsProcessing on the internal queue."""

    await stub_voice_output_channel.notify_message_processing_started()

    assert call_state.internal_queue.qsize() == 1
    message = call_state.internal_queue.get_nowait()
    assert isinstance(message, RasaIsProcessing)


@pytest.mark.usefixtures("setup_call_state")
async def test_notify_message_processing_completed_enqueues_rasa_is_listening(
    stub_voice_output_channel: StubVoiceOutputChannel,
):
    """notify_message_processing_completed puts RasaIsListening
    on the internal queue.
    """

    await stub_voice_output_channel.notify_message_processing_completed()

    assert call_state.internal_queue.qsize() == 1
    message = call_state.internal_queue.get_nowait()
    assert isinstance(message, RasaIsListening)


@pytest.mark.usefixtures("setup_call_state")
async def test_notify_message_processing_started_then_completed_order(
    stub_voice_output_channel: StubVoiceOutputChannel,
):
    """Calling started then completed enqueues messages in FIFO order."""

    await stub_voice_output_channel.notify_message_processing_started()
    await stub_voice_output_channel.notify_message_processing_completed()

    assert call_state.internal_queue.qsize() == 2
    assert isinstance(call_state.internal_queue.get_nowait(), RasaIsProcessing)
    assert isinstance(call_state.internal_queue.get_nowait(), RasaIsListening)
