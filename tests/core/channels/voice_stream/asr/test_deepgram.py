import json
from unittest import mock

import pytest

from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR
from rasa.shared.exceptions import ProviderClientValidationError


async def test_environment_validation():
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            DeepgramASR(rasa_language="en")
        assert e.match(DeepgramASR.required_env_vars[0])
        assert e.match("ASR Engine DeepgramASR")


def test_configurating_endpoint():
    custom_endpoint = "local_endpoint.myurl.com"
    config = {"endpoint": custom_endpoint}
    asr_engine = DeepgramASR.from_config_dict(config, rasa_language="en")
    assert asr_engine.config.endpoint == custom_endpoint
    assert custom_endpoint in asr_engine._get_api_url()


def test_configurating_endpointing():
    custom_endpointing = 1000
    config = {"endpointing": custom_endpointing}
    asr_engine = DeepgramASR.from_config_dict(config, rasa_language="en")
    assert asr_engine.config.endpointing == custom_endpointing
    assert f"endpointing={custom_endpointing}" in asr_engine._get_query_params()


def test_configurating_language():
    custom_language = "es"
    config = {
        "language_map": {
            "en": {"language": "en", "model": "nova-2-general"},
            "es": {"language": "es", "model": "nova-2-general"},
        }
    }
    asr_engine = DeepgramASR.from_config_dict(
        config, rasa_language="es", additional_languages=["en"]
    )
    assert asr_engine.current_language_config.engine_language_key == custom_language
    assert f"language={custom_language}" in asr_engine._get_query_params()


def test_configurating_utterance_end_detection():
    custom_value = 2000
    config = {"utterance_end_ms": custom_value}
    asr_engine = DeepgramASR.from_config_dict(config, rasa_language="en")
    assert asr_engine.config.utterance_end_ms == custom_value
    assert f"utterance_end_ms={custom_value}" in asr_engine._get_query_params()


def test_turning_off_utterance_end_detection():
    custom_value = -1
    config = {"utterance_end_ms": custom_value}
    asr_engine = DeepgramASR.from_config_dict(config, rasa_language="en")
    assert asr_engine.config.utterance_end_ms == custom_value
    assert "utterance_end_ms" not in asr_engine._get_query_params()


def test_configuration_additional_attributes():
    config = {"testingXYZ@@": "@@"}
    with pytest.raises(
        Exception
    ):  # Pydantic raises ValidationError for missing language_map
        DeepgramASR.from_config_dict(config, rasa_language="en")


@pytest.mark.parametrize(
    "t1,t2,expected_result",
    [
        ("", "", ""),
        ("", "abc", "abc"),
        ("def", "", "def"),
        ("It should not", "boil", "It should not boil"),
        ("It should not ", "boil", "It should not boil"),
        ("It should not", " boil", "It should not boil"),
        ("It should not ", " boil", "It should not boil"),
        ("You may,", "he noted", "You may, he noted"),
        ("You may, ", " he noted", "You may, he noted"),
    ],
)
async def test_transcript_concatenation(t1: str, t2: str, expected_result: str):
    assert expected_result == DeepgramASR.concatenate_transcripts(t1, t2)


@pytest.mark.asyncio
async def test_deepgram_keep_alive_sends_message():
    asr_engine = DeepgramASR(rasa_language="en")
    mock_socket = mock.AsyncMock()
    asr_engine.asr_socket = mock_socket

    await asr_engine.send_keep_alive()

    mock_socket.send.assert_awaited_once_with(json.dumps({"type": "KeepAlive"}))
