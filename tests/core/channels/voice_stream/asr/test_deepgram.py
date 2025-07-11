import json
from dataclasses import asdict
from unittest import mock

import pytest

from rasa.core.channels.voice_stream.asr.asr_event import UserIsSpeaking
from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR, DeepgramASRConfig
from rasa.shared.exceptions import ProviderClientValidationError
from tests.core.channels.voice_stream.asr import (
    run_single_utterance_transcription,
    run_transcription,
)


async def test_environment_validation():
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            DeepgramASR()
        assert e.match(DeepgramASR.required_env_vars[0])
        assert e.match("ASR Engine DeepgramASR")


async def test_transcription(audio_data_path: str):
    audio_path = audio_data_path + "/01.wav"
    transcript = open(audio_data_path + "/01.txt").read()
    asr_engine = DeepgramASR()

    await run_single_utterance_transcription(audio_path, transcript, asr_engine)


async def test_noisy_transcription(audio_data_path: str):
    audio_path = audio_data_path + "/01_noisy.wav"
    transcript = open(audio_data_path + "/01.txt").read()
    asr_engine = DeepgramASR()

    await run_single_utterance_transcription(audio_path, transcript, asr_engine)


async def test_noisy_transcription_without_utterance_end(audio_data_path: str):
    """Test that the utterance end feature makes a difference."""
    audio_path = audio_data_path + "/02_noisy2.wav"
    transcript = open(audio_data_path + "/02.txt").read()

    # this works fine
    asr_engine = DeepgramASR()
    await run_single_utterance_transcription(audio_path, transcript, asr_engine)

    # now we deactivate utterance_end detection and will not get a finalized transcript
    asr_engine = DeepgramASR(DeepgramASRConfig(utterance_end_ms=0, endpointing=600))

    events = await run_transcription(audio_path, asr_engine)

    assert len(events) > 2
    assert all([isinstance(event, UserIsSpeaking) for event in events])


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
