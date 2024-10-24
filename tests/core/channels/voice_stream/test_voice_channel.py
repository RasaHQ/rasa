import pytest

from rasa.core.channels.voice_stream.tts.azure import AzureTTS
from rasa.core.channels.voice_stream.voice_channel import tts_engine_from_config


def test_azure_tts_engine_from_config():
    config = {"name": "azure", "voice": "james"}
    tts_engine = tts_engine_from_config(config)
    assert isinstance(tts_engine, AzureTTS)
    assert tts_engine.config.voice == "james"
    default_config = AzureTTS.get_default_config()
    assert tts_engine.config.language == default_config.language
    assert tts_engine.config.speech_region == default_config.speech_region


def test_tts_engine_from_config_fails_for_not_implemented_engine():
    config = {"name": "XY_non_existent"}
    with pytest.raises(NotImplementedError):
        tts_engine_from_config(config)
