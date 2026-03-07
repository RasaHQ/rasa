import asyncio
from typing import Any, AsyncIterator, Dict
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from data.test_voice_channel.custom_asr_engine import CustomASREngine
from data.test_voice_channel.custom_tts_engine import CustomTTSEngine
from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.asr.asr_event import (
    ASREvent,
    NewTranscript,
    UserIsSpeaking,
)
from rasa.core.channels.voice_stream.call_state import _call_state, call_state
from rasa.core.channels.voice_stream.tts.azure import AzureTTS
from rasa.core.channels.voice_stream.voice_channel import (
    DTMFInputAction,
    VoiceInputChannel,
    VoiceLanguageChangePlugin,
    asr_engine_from_config,
    tts_engine_from_config,
)
from rasa.shared.core.flows.steps.collect import DTMFConfig


class StubVoiceInputChannel(VoiceInputChannel):
    """Minimal VoiceInputChannel stub for testing _on_asr_event_received.

    The test only calls _on_asr_event_received, should_interrupt, and
    interrupt_playback; inherited name() and other methods are unused.
    """

    pass


async def test_azure_tts_engine_from_config(mulaw_format):
    config = {"name": "azure"}
    tts_engine = tts_engine_from_config(config, language="en", format=mulaw_format)
    assert isinstance(tts_engine, AzureTTS)
    default_config = AzureTTS.get_default_config()
    assert tts_engine.config.speech_region == default_config.speech_region
    if tts_engine.session:
        await tts_engine.session.close()


def test_tts_engine_from_config_fails_for_not_implemented_engine(mulaw_format):
    config = {"name": "XY_non_existent"}
    with pytest.raises(ImportError):
        tts_engine_from_config(config, language="en", format=mulaw_format)


def test_custom_asr_service(mulaw_format) -> None:
    # Given a custom ASR engine
    config = {
        "name": "data.test_voice_channel.custom_asr_engine.CustomASREngine",
        "endpoint": "http://localhost:8000",
    }

    # When the ASR engine is created from the config
    asr_engine = asr_engine_from_config(config, language="en", format=mulaw_format)

    # Then the ASR engine should be an instance of the custom ASR engine
    assert isinstance(asr_engine, CustomASREngine)


def test_custom_tts_service(mulaw_format) -> None:
    # Given a custom TTS engine
    config = {
        "name": "data.test_voice_channel.custom_tts_engine.CustomTTSEngine",
        "server_url": "http://localhost:8000",
    }

    # When the TTS engine is created from the config
    tts_engine = tts_engine_from_config(config, language="en", format=mulaw_format)

    # Then the ASR engine should be an instance of the custom ASR engine
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
def test_asr_engine_config_validation(config, expected_error, mulaw_format):
    """Test validation of ASR engine configuration."""
    with pytest.raises(ValueError, match=expected_error):
        asr_engine_from_config(config, language="en", format=mulaw_format)


@pytest.mark.parametrize(
    "config,expected_error",
    [
        ({}, "TTS configuration dictionary cannot be empty"),
        (
            {"key": "value"},
            "TTS configuration must contain 'name' key specifying the engine type",
        ),
        (None, "TTS configuration dictionary cannot be empty"),
    ],
)
def test_tts_engine_config_validation(config, expected_error, mulaw_format):
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


def test_dtmf_config_in_call_state(setup_call_state):
    """Test storing DTMF config in call state."""
    call_state = _call_state.get()

    config = DTMFConfig(length=6, allow_audio_input=False)
    call_state.dtmf_config = config
    call_state.is_collecting_dtmf = True

    assert call_state.dtmf_config is not None
    assert call_state.dtmf_config.length == 6
    assert call_state.dtmf_config.allow_audio_input is False
    assert call_state.is_collecting_dtmf is True


def test_dtmf_config_none_when_not_collecting(setup_call_state):
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
async def test_stop_streaming_and_interrupt_playback_on_interruption(
    allow_interruptions_dict: Dict[str, bool],
    mock_validate_voice_license_scope,
    setup_call_state,
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
async def test_receive_asr_events_does_not_interrupt_when_words_below_threshold(
    allow_interruptions_dict: Dict[str, bool],
    mock_validate_voice_license_scope,
    setup_call_state,
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
async def test_interruptions_for_multiple_asr_events_in_sequence(
    allow_interruptions_dict: Dict[str, bool],
    mock_validate_voice_license_scope,
    setup_call_state,
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
async def test_interruptions_not_firing_when_disabled(
    allow_interruptions_dict: Dict[str, bool],
    mock_validate_voice_license_scope,
    setup_call_state,
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


async def test_language_change_hook_schedules_update(
    setup_call_state,
) -> None:
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


async def test_language_change_hook_ignores_other_senders(
    setup_call_state,
) -> None:
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


async def test_run_audio_streaming_unregisters_language_plugin_on_session_error(
    mock_validate_voice_license_scope: Any,
) -> None:
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
