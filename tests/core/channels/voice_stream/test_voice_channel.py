import pytest

from data.test_voice_channel.custom_asr_engine import CustomASREngine
from data.test_voice_channel.custom_tts_engine import CustomTTSEngine
from rasa.core.channels.voice_stream.call_state import _call_state
from rasa.core.channels.voice_stream.tts.azure import AzureTTS
from rasa.core.channels.voice_stream.voice_channel import (
    DTMFInputAction,
    asr_engine_from_config,
    tts_engine_from_config,
)
from rasa.shared.core.flows.steps.collect import DTMFConfig


async def test_azure_tts_engine_from_config():
    config = {"name": "azure"}
    tts_engine = tts_engine_from_config(config)
    assert isinstance(tts_engine, AzureTTS)
    default_config = AzureTTS.get_default_config()
    assert tts_engine.config.language == default_config.language
    assert tts_engine.config.speech_region == default_config.speech_region
    if tts_engine.session:
        await tts_engine.session.close()


def test_tts_engine_from_config_fails_for_not_implemented_engine():
    config = {"name": "XY_non_existent"}
    with pytest.raises(ImportError):
        tts_engine_from_config(config)


def test_custom_asr_service() -> None:
    # Given a custom ASR engine
    config = {
        "name": "data.test_voice_channel.custom_asr_engine.CustomASREngine",
        "endpoint": "http://localhost:8000",
        "language": "en",
    }

    # When the ASR engine is created from the config
    asr_engine = asr_engine_from_config(config)

    # Then the ASR engine should be an instance of the custom ASR engine
    assert isinstance(asr_engine, CustomASREngine)


def test_custom_tts_service() -> None:
    # Given a custom TTS engine
    config = {
        "name": "data.test_voice_channel.custom_tts_engine.CustomTTSEngine",
        "server_url": "http://localhost:8000",
        "language": "hi",
    }

    # When the TTS engine is created from the config
    tts_engine = tts_engine_from_config(config)

    # Then the ASR engine should be an instance of the custom ASR engine
    assert isinstance(tts_engine, CustomTTSEngine)
    assert tts_engine.config.language == "hi"


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
def test_asr_engine_config_validation(config, expected_error):
    """Test validation of ASR engine configuration."""
    with pytest.raises(ValueError, match=expected_error):
        asr_engine_from_config(config)


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
def test_tts_engine_config_validation(config, expected_error):
    """Test validation of TTS engine configuration."""
    with pytest.raises(ValueError, match=expected_error):
        tts_engine_from_config(config)


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
