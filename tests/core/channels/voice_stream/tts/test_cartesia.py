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
            CartesiaTTS()
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
    tts_engine = CartesiaTTS()
    text = "Hello there!"
    with pytest.raises(TTSError):
        async for chunk in tts_engine.synthesize(text):
            pass


def test_default_config():
    config = CartesiaTTS.get_default_config()
    assert config.language == "en"
    assert config.voice == "248be419-c632-4f23-adf1-5324ed7dbf1d"
    assert config.model_id == "sonic-english"
    assert config.version == "2024-06-10"


async def test_tts_session_sharing():
    tts_engine = CartesiaTTS()
    tts_engine_2 = CartesiaTTS()
    assert tts_engine_2.session is tts_engine.session
