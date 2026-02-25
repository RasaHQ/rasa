from unittest import mock

import pytest
from pytest import MonkeyPatch

from rasa.core.channels.voice_stream.tts.cartesia import CartesiaTTS
from rasa.core.channels.voice_stream.tts.tts_engine import TTSError
from rasa.shared.exceptions import ProviderClientValidationError


async def test_environment_validation():
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            CartesiaTTS(rasa_language="en")
        assert e.match(CartesiaTTS.required_env_vars[0])
        assert e.match("TTS Engine CartesiaTTS")


# TODO: Cartesia has stopped sending Status 400 for invalid requests
# @pytest.mark.parametrize(
#     "bad_config",
#     [
#         CartesiaTTSConfig.from_dict({"model_id": "nonexistent"}),
#         CartesiaTTSConfig.from_dict({"voice": "non_existent_voice"}),
#     ],
# )
# async def test_synthesis_error(bad_config):
#     tts_engine = CartesiaTTS()
#     text = "Hello there!"
#     with pytest.raises(TTSError):
#         async for chunk in tts_engine.synthesize(text, bad_config):
#             pass


async def test_synthesis_bad_api_key(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "bad_key")
    tts_engine = CartesiaTTS(rasa_language="en")
    text = "Hello there!"
    with pytest.raises(TTSError):
        async for chunk in tts_engine.synthesize(text):
            pass


def test_default_config():
    config = CartesiaTTS.get_default_config()
    assert "en" in config.language_map
    assert config.language_map["en"].language == "en"
    assert config.language_map["en"].voice == "f786b574-daa5-4673-aa0c-cbe3e8534c02"
    assert config.model_id == "sonic-3"
    assert config.version == "2025-04-16"


async def test_tts_session_sharing():
    tts_engine = CartesiaTTS(rasa_language="en")
    tts_engine_2 = CartesiaTTS(rasa_language="en")
    assert tts_engine_2.session is tts_engine.session
