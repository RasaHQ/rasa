import json
from unittest import mock
from unittest.mock import AsyncMock, patch

import pytest
from pytest import MonkeyPatch

from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR
from rasa.core.channels.voice_stream.tts.cartesia import CartesiaTTS
from rasa.core.channels.voice_stream.tts.tts_engine import TTSError
from rasa.shared.exceptions import ProviderClientValidationError
from tests.core.channels.voice_stream.tts.test_tts import (
    run_single_utterance_through_tts_and_asr,
)


async def test_environment_validation():
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            CartesiaTTS()
        assert e.match(CartesiaTTS.required_env_vars[0])
        assert e.match("TTS Engine CartesiaTTS")


async def test_synthesis_with_asr():
    tts_engine = CartesiaTTS()
    text = "hello my name is Edgar"
    asr_engine = DeepgramASR()

    await run_single_utterance_through_tts_and_asr(text, asr_engine, tts_engine)


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


def test_tts_request_body():
    config = CartesiaTTS.get_default_config()
    text = "Hi there, how can I help you today?"
    request_body = json.dumps(CartesiaTTS.get_request_body(text, config))
    assert text in request_body
    assert config.voice in request_body
    assert config.language in request_body
    assert config.model_id in request_body


def test_tts_headers():
    config = CartesiaTTS.get_default_config()
    headers = CartesiaTTS.get_request_headers(config)
    assert "X-API-Key" in headers
    assert headers["Content-Type"] == "application/json"
    assert headers["Cartesia-Version"] == config.version


async def test_tts_session_sharing():
    tts_engine = CartesiaTTS()
    tts_engine_2 = CartesiaTTS()
    assert tts_engine_2.session is tts_engine.session


async def test_synthesize_timeout(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "my key")
    tts_engine = CartesiaTTS()
    text = "Test timeout"
    assert tts_engine.session is not None

    # Mock the response to be an async context manager
    mock_response = AsyncMock()
    # Did this to avoid AttributeError: __aenter__ error
    mock_response.__aenter__.side_effect = TimeoutError("Request timed out")

    # Patch the 'post' method to return the mock response
    with patch.object(tts_engine.session, "post", return_value=mock_response):
        with pytest.raises(TTSError) as exc_info:
            async for chunk in tts_engine.synthesize(text):
                pass

        assert "Request timed out" in str(exc_info.value)
