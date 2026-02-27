from unittest import mock

import pytest
from pytest import MonkeyPatch

from rasa.core.channels.voice_stream.audio_bytes import L16_24KHZ, L16_48KHZ, MULAW_8KHZ
from rasa.core.channels.voice_stream.tts.deepgram import DeepgramTTS
from rasa.core.channels.voice_stream.tts.tts_engine import TTSError
from rasa.shared.exceptions import ProviderClientValidationError


async def test_environment_validation(mulaw_format):
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            DeepgramTTS(rasa_language="en", format=mulaw_format)
        assert e.match(DeepgramTTS.required_env_vars[0])
        assert e.match("TTS Engine DeepgramTTS")


async def test_synthesis_bad_api_key(monkeypatch: MonkeyPatch, mulaw_format):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "bad_key")
    tts_engine = DeepgramTTS(rasa_language="en", format=mulaw_format)
    text = "Hello there!"
    with pytest.raises(TTSError):
        async for chunk in tts_engine.synthesize(text):
            pass


def test_default_config():
    config = DeepgramTTS.get_default_config()
    assert config.endpoint == "wss://api.deepgram.com/v1/speak"
    assert "en" in config.language_map
    assert config.language_map["en"].model == "aura-2-andromeda-en"


async def test_tts_session_sharing(mulaw_format):
    tts_engine = DeepgramTTS(rasa_language="en", format=mulaw_format)
    tts_engine_2 = DeepgramTTS(rasa_language="en", format=mulaw_format)
    assert tts_engine_2.session is tts_engine.session


@pytest.mark.parametrize(
    "format",
    [
        MULAW_8KHZ,
        L16_24KHZ,
        L16_48KHZ,
    ],
)
async def test_configuration_format(format):
    tts_engine = DeepgramTTS.from_config_dict(
        config={}, rasa_language="en", format=format
    )
    assert tts_engine.audio_format == format
