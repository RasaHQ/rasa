from unittest import mock
from unittest.mock import AsyncMock, patch

import pytest
from pytest import MonkeyPatch

from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR
from rasa.core.channels.voice_stream.tts.azure import AzureTTS, AzureTTSConfig
from rasa.core.channels.voice_stream.tts.tts_engine import TTSError
from rasa.shared.exceptions import ProviderClientValidationError
from tests.core.channels.voice_stream.tts.test_tts import (
    run_single_utterance_through_tts_and_asr,
)


async def test_environment_validation():
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            AzureTTS()
        assert e.match(AzureTTS.required_env_vars[0])
        assert e.match("TTS Engine AzureTTS")


async def test_synthesis_with_asr():
    tts_engine = AzureTTS()
    text = "hello my name is Edgar"
    asr_engine = DeepgramASR()
    await run_single_utterance_through_tts_and_asr(text, asr_engine, tts_engine)


@pytest.mark.parametrize(
    "bad_config",
    [
        AzureTTSConfig.from_dict({"speech_region": "nonexistent"}),
        AzureTTSConfig.from_dict({"voice": "non_existent_voice"}),
    ],
)
async def test_synthesis_error(bad_config):
    tts_engine = AzureTTS()
    text = "Hello there!"
    with pytest.raises(TTSError):
        async for chunk in tts_engine.synthesize(text, bad_config):
            pass


async def test_synthesis_bad_api_key(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("AZURE_SPEECH_API_KEY", "bad key")
    tts_engine = AzureTTS()
    text = "Hello there!"
    with pytest.raises(TTSError):
        async for chunk in tts_engine.synthesize(text):
            pass


def test_azure_default_config():
    config = AzureTTS.get_default_config()
    assert config.language == "en-US"
    assert config.voice == "en-US-JennyNeural"
    assert config.speech_region == "eastus"


def test_tts_url_creation():
    config = AzureTTS.get_default_config()
    azure_tts_endpoint = AzureTTS.get_tts_endpoint(config)
    assert config.speech_region in azure_tts_endpoint
    assert azure_tts_endpoint.startswith("https://")


def test_tts_request_body():
    config = AzureTTS.get_default_config()
    text = "Hi there, how can I help you today?"
    request_body = AzureTTS.create_request_body(text, config)
    assert text in request_body
    assert config.voice in request_body
    assert config.language in request_body


def test_tts_headers():
    headers = AzureTTS.get_request_headers()
    assert "Ocp-Apim-Subscription-Key" in headers
    assert "Content-Type" in headers
    assert "X-Microsoft-OutputFormat" in headers
    assert headers["X-Microsoft-OutputFormat"] == "raw-8khz-8bit-mono-mulaw"


async def test_tts_session_sharing():
    tts_engine = AzureTTS()
    tts_engine_2 = AzureTTS()
    assert tts_engine_2.session is tts_engine.session


async def test_synthesize_timeout(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("AZURE_SPEECH_API_KEY", "my key")
    tts_engine = AzureTTS()
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
