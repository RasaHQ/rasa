import pytest
from rasa.core.channels.voice_stream.tts.azure import AzureTTS
from rasa.core.channels.voice_stream.voice_channel import (
    tts_engine_from_config,
    asr_engine_from_config,
)
from data.test_voice_channel.custom_asr_engine import CustomASREngine
from data.test_voice_channel.custom_tts_engine import CustomTTSEngine


def test_azure_tts_engine_from_config():
    config = {"name": "azure"}
    tts_engine = tts_engine_from_config(config)
    assert isinstance(tts_engine, AzureTTS)
    default_config = AzureTTS.get_default_config()
    assert tts_engine.config.language == default_config.language
    assert tts_engine.config.speech_region == default_config.speech_region


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
