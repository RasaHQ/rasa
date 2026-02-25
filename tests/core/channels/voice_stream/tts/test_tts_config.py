import pytest

from rasa.core.channels.voice_stream.tts.azure import AzureTTSConfig
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngineConfig


def test_tts_config_merge():
    default_tts_config = TTSEngineConfig("en", "ultra-test")
    user_tts_config = TTSEngineConfig(None, "simple-test")

    merged_tts_config = default_tts_config.merge(user_tts_config)

    assert merged_tts_config.language == "en"
    assert merged_tts_config.voice == "simple-test"


def test_tts_config_from_dict():
    config = TTSEngineConfig.from_dict({"language": "en"})
    assert config.language == "en"
    assert config.voice is None


def test_tts_config_from_dict_extra_params():
    with pytest.raises(Exception):  # Pydantic raises ValidationError for extra fields
        TTSEngineConfig.from_dict({"voice": "test-voice", "accent": "british"})


def test_tts_config_from_dict_type():
    config = AzureTTSConfig.from_dict({"voice": "Jenny"})
    assert isinstance(config, AzureTTSConfig)
    assert config.voice == "Jenny"
