import pytest

from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR
from rasa.core.channels.voice_stream.tts.azure import AzureTTS, AzureTTSConfig
from rasa.core.channels.voice_stream.tts.tts_engine import TTSError
from tests.core.channels.voice_stream.tts.test_tts import (
    run_single_utterance_through_tts_and_asr,
)


async def test_synthesis_with_asr():
    tts_engine = AzureTTS()
    text = "hello there"
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


def test_azure_default_config():
    config = AzureTTS.get_default_config()
    assert config.language == "en-US"
    assert config.voice == "en-US-JennyNeural"
    assert config.speech_region == "germanywestcentral"


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
