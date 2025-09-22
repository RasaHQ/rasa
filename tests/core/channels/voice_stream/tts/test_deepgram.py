from unittest import mock

import pytest
from pytest import MonkeyPatch

from rasa.core.channels.voice_stream.tts.deepgram import DeepgramTTS
from rasa.core.channels.voice_stream.tts.tts_engine import TTSError
from rasa.shared.exceptions import ProviderClientValidationError


async def test_environment_validation():
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            DeepgramTTS()
        assert e.match(DeepgramTTS.required_env_vars[0])
        assert e.match("TTS Engine DeepgramTTS")


async def test_synthesis_bad_api_key(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "bad_key")
    tts_engine = DeepgramTTS()
    text = "Hello there!"
    with pytest.raises(TTSError):
        async for chunk in tts_engine.synthesize(text):
            pass


def test_default_config():
    config = DeepgramTTS.get_default_config()
    assert config.endpoint == "wss://api.deepgram.com/v1/speak"
    assert config.model_id == "aura-2-andromeda-en"


async def test_tts_session_sharing():
    tts_engine = DeepgramTTS()
    tts_engine_2 = DeepgramTTS()
    assert tts_engine_2.session is tts_engine.session
