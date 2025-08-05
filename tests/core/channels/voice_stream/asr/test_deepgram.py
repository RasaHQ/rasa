import json
from dataclasses import asdict
from unittest import mock

import pytest

from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR
from rasa.shared.exceptions import ProviderClientValidationError


async def test_environment_validation():
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            DeepgramASR()
        assert e.match(DeepgramASR.required_env_vars[0])
        assert e.match("ASR Engine DeepgramASR")


def test_configurating_endpoint():
    custom_endpoint = "local_endpoint.myurl.com"
    config = {"endpoint": custom_endpoint}
    default_config = DeepgramASR.get_default_config()
    asr_engine = DeepgramASR.from_config_dict(config)
    assert asr_engine.config.endpoint == custom_endpoint
    assert asdict(asr_engine.config) == {**asdict(default_config), **config}
    assert custom_endpoint in asr_engine._get_api_url()


def test_configurating_endpointing():
    custom_endpointing = 1000
    config = {"endpointing": custom_endpointing}
    default_config = DeepgramASR.get_default_config()
    asr_engine = DeepgramASR.from_config_dict(config)
    assert asr_engine.config.endpointing == custom_endpointing
    assert asdict(asr_engine.config) == {**asdict(default_config), **config}
    assert f"endpointing={custom_endpointing}" in asr_engine._get_query_params()


def test_configurating_language():
    custom_language = "es"
    config = {"language": custom_language}
    default_config = DeepgramASR.get_default_config()
    asr_engine = DeepgramASR.from_config_dict(config)
    assert asr_engine.config.language == custom_language
    assert asdict(asr_engine.config) == {**asdict(default_config), **config}
    assert f"language={custom_language}" in asr_engine._get_query_params()


def test_configurating_utterance_end_detection():
    custom_value = 2000
    config = {"utterance_end_ms": custom_value}
    default_config = DeepgramASR.get_default_config()
    asr_engine = DeepgramASR.from_config_dict(config)
    assert asr_engine.config.utterance_end_ms == custom_value
    assert asdict(asr_engine.config) == {**asdict(default_config), **config}
    assert f"utterance_end_ms={custom_value}" in asr_engine._get_query_params()


def test_turning_off_utterance_end_detection():
    custom_value = -1
    config = {"utterance_end_ms": custom_value}
    default_config = DeepgramASR.get_default_config()
    asr_engine = DeepgramASR.from_config_dict(config)
    assert asr_engine.config.utterance_end_ms == custom_value
    assert asdict(asr_engine.config) == {**asdict(default_config), **config}
    assert "utterance_end_ms" not in asr_engine._get_query_params()


def test_configuration_additional_attributes():
    config = {"testingXYZ@@": "@@"}
    with pytest.raises(TypeError):
        DeepgramASR.from_config_dict(config)


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
    asr_engine = DeepgramASR()
    mock_socket = mock.AsyncMock()
    asr_engine.asr_socket = mock_socket

    await asr_engine.send_keep_alive()

    mock_socket.send.assert_awaited_once_with(json.dumps({"type": "KeepAlive"}))
