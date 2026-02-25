from unittest import mock

import pytest

from rasa.core.channels.voice_stream.asr.azure import AzureASR, AzureASRConfig
from rasa.exceptions import MissingDependencyException
from rasa.shared.constants import AZURE_SPEECH_API_KEY_ENV_VAR
from rasa.shared.exceptions import ProviderClientValidationError


async def test_environment_validation():
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            AzureASR(rasa_language="en", config=AzureASRConfig(speech_region="eastus"))
        assert e.match(AZURE_SPEECH_API_KEY_ENV_VAR)
        assert e.match("ASR Engine AzureASR")

    # no package installed
    with mock.patch("importlib.import_module") as mock_call:
        mock_call.side_effect = ImportError
        with pytest.raises(MissingDependencyException) as e:
            AzureASR(rasa_language="en", config=AzureASRConfig(speech_region="eastus"))
        assert e.match("ASR Engine AzureASR")
        assert e.match(AzureASR.required_packages[0])


async def test_configurating_endpoint():
    custom_region = "germanywestcentral"
    config = {"speech_region": custom_region}
    asr_engine = AzureASR.from_config_dict(config, rasa_language="en")
    assert asr_engine.config.speech_region == custom_region


async def test_configurating_language():
    config = {
        "speech_host": "custom.host.url",
        "language_map": {
            "en": {"language": "en-US"},
            "es": {"language": "es-ES"},
        },
    }
    asr_engine = AzureASR.from_config_dict(
        config, rasa_language="es", additional_languages=["en"]
    )
    assert asr_engine.current_language_config.engine_language_key == "es-ES"


async def test_configuration_addioinal_attributes():
    config = {"testingXYZ@@": "@@"}
    with pytest.raises(Exception):  # Pydantic raises ValidationError for extra fields
        AzureASR.from_config_dict(config, rasa_language="en")
