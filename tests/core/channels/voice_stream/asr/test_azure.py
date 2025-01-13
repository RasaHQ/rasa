from dataclasses import asdict
from unittest import mock

import pytest

from rasa.core.channels.voice_stream.asr.azure import AzureASR
from rasa.exceptions import MissingDependencyException
from rasa.shared.constants import AZURE_SPEECH_API_KEY_ENV_VAR
from rasa.shared.exceptions import ProviderClientValidationError
from tests.core.channels.voice_stream.asr import (
    run_single_utterance_transcription,
)


async def test_environment_validation():
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            AzureASR()
        assert e.match(AZURE_SPEECH_API_KEY_ENV_VAR)
        assert e.match("ASR Engine AzureASR")

    # no package installed
    with mock.patch("importlib.import_module") as mock_call:
        mock_call.side_effect = ImportError
        with pytest.raises(MissingDependencyException) as e:
            AzureASR()
        assert e.match("ASR Engine AzureASR")
        assert e.match(AzureASR.required_packages[0])


async def test_transcription(audio_data_path: str):
    audio_path = audio_data_path + "/01.wav"
    transcript = open(audio_data_path + "/01.txt").read()
    asr_engine = AzureASR()

    await run_single_utterance_transcription(audio_path, transcript, asr_engine)


async def test_noisy_transcription(audio_data_path: str):
    audio_path = audio_data_path + "/01_noisy.wav"
    transcript = open(audio_data_path + "/01.txt").read()
    asr_engine = AzureASR()

    await run_single_utterance_transcription(audio_path, transcript, asr_engine)


async def test_configurating_endpoint():
    custom_region = "local_endpoint.myurl.com"
    config = {"speech_region": custom_region}
    default_config = AzureASR.get_default_config()
    asr_engine = AzureASR.from_config_dict(config)
    assert asr_engine.config.speech_region == custom_region
    assert asdict(asr_engine.config) == {**asdict(default_config), **config}


async def test_configurating_language():
    custom_language = "es"
    config = {"language": custom_language}
    default_config = AzureASR.get_default_config()
    asr_engine = AzureASR.from_config_dict(config)
    assert asr_engine.config.language == custom_language
    assert asdict(asr_engine.config) == {**asdict(default_config), **config}


async def test_configuration_addioinal_attributes():
    config = {"testingXYZ@@": "@@"}
    with pytest.raises(TypeError):
        AzureASR.from_config_dict(config)
