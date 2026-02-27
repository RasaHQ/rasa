import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from data.test_voice_channel.custom_asr_engine import CustomASREngine
from data.test_voice_channel.custom_tts_engine import CustomTTSEngine
from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.asr.asr_event import (
    NewTranscript,
    UserIsSpeaking,
)
from rasa.core.channels.voice_stream.call_state import _call_state, call_state
from rasa.core.channels.voice_stream.tts.azure import AzureTTS
from rasa.core.channels.voice_stream.voice_channel import (
    DTMFInputAction,
    VoiceInputChannel,
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


async def test_on_asr_event_received_calls_should_interrupt_and_interrupt_playback(
    mock_validate_voice_license_scope,
    setup_call_state,
):
    """_on_asr_event_received calls should_interrupt for every event
    and interrupt_playback only when should_interrupt returns True.
    """
    channel = StubVoiceInputChannel(
        server_url="https://example.com",
        asr_config={"name": "deepgram"},
        tts_config={"name": "azure"},
        interruptions={"enabled": True, "min_words": 3},
    )
    call_state.channel_data["allow_interruptions"] = True

    asr_event_queue: asyncio.Queue = asyncio.Queue()
    channel_websocket = MagicMock()
    call_parameters = CallParameters(
        call_id="call_123",
        user_phone="+123",
        bot_phone="+456",
        stream_id="stream_456",
        direction="inbound",
    )

    events = [
        UserIsSpeaking(text="one"),
        NewTranscript(text="one two three"),
        UserIsSpeaking(text="one two three four"),
    ]
    should_interrupt_calls = []
    base_should_interrupt = VoiceInputChannel.should_interrupt

    def record_and_should_interrupt(e: Any) -> bool:
        should_interrupt_calls.append(e)
        return base_should_interrupt(channel, e)

    channel.should_interrupt = record_and_should_interrupt
    channel.interrupt_playback = AsyncMock()

    for event in events:
        await asr_event_queue.put(event)
        if channel.should_interrupt(event):
            await channel.interrupt_playback(channel_websocket, call_parameters)

    assert should_interrupt_calls == events
    assert channel.interrupt_playback.await_count == 2
    channel.interrupt_playback.assert_has_calls(
        [call(channel_websocket, call_parameters)] * 2
    )
    assert asr_event_queue.qsize() == 3
    for expected in events:
        assert await asr_event_queue.get() is expected
