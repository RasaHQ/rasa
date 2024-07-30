import pytest
import structlog
from pytest import MonkeyPatch

from rasa.shared.constants import (
    AZURE_API_BASE_ENV_VAR,
    AZURE_API_VERSION_ENV_VAR,
    AZURE_API_KEY_ENV_VAR,
    OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY,
    OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY,
    OPENAI_API_KEY_ENV_VAR,
    AZURE_API_TYPE_ENV_VAR,
    OPENAI_API_BASE_ENV_VAR,
    OPENAI_API_VERSION_ENV_VAR,
    OPENAI_API_TYPE_ENV_VAR,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers.llm.azure_openai_llm_client import (
    AzureOpenAILLMClient,
)
from rasa.shared.providers.llm.llm_client import LLMClient
from tests.utilities import filter_logs


class TestAzureOpenAILLMClient:
    def test_conforms_to_protocol(self, monkeypatch: MonkeyPatch):
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")
        client = AzureOpenAILLMClient(
            deployment="test_deployment",
            api_base="https://my.api.base.com/my_model",
            api_version="2023-01-01",
        )
        assert isinstance(client, LLMClient)

    def test_init_fetches_from_environment_variables(self, monkeypatch: MonkeyPatch):
        # Given

        # Set the environment variables
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")
        monkeypatch.setenv(AZURE_API_BASE_ENV_VAR, "https://my.api.base.com/my_model")
        monkeypatch.setenv(AZURE_API_VERSION_ENV_VAR, "2023-01-01")
        monkeypatch.setenv(AZURE_API_TYPE_ENV_VAR, "test api type")

        # When
        client = AzureOpenAILLMClient(deployment="test_deployment")

        # Then
        assert client.deployment == "test_deployment"
        assert client.model is None
        assert client.api_base == "https://my.api.base.com/my_model"
        assert client.api_version == "2023-01-01"
        assert client._api_key == "my key"
        assert client.api_type == "test api type"

    def test_init_fetches_from_deprecated_environment_variables(
        self, monkeypatch: MonkeyPatch
    ):
        # Given

        # Set the environment variables
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        monkeypatch.setenv(OPENAI_API_BASE_ENV_VAR, "https://my.api.base.com/my_model")
        monkeypatch.setenv(OPENAI_API_VERSION_ENV_VAR, "2023-01-01")
        monkeypatch.setenv(OPENAI_API_TYPE_ENV_VAR, "test api type")

        # When
        client = AzureOpenAILLMClient(deployment="test_deployment")

        # Then
        assert client.deployment == "test_deployment"
        assert client.model is None
        assert client.api_base == "https://my.api.base.com/my_model"
        assert client.api_version == "2023-01-01"
        assert client._api_key == "my key"
        assert client.api_type == "test api type"

    def test_validate_client_setup(self, monkeypatch: MonkeyPatch):
        # Given
        expected_event = "azure_openai_llm_client.not_configured"
        expected_log_level = "error"
        expected_log_message_parts = [
            "Set API Base",
            AZURE_API_BASE_ENV_VAR,
            OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY,
            "Set API Version",
            AZURE_API_VERSION_ENV_VAR,
            OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY,
            "Set API Key",
            AZURE_API_KEY_ENV_VAR,
        ]

        with structlog.testing.capture_logs() as caplog:
            with pytest.raises(ProviderClientValidationError):
                AzureOpenAILLMClient(deployment="test_deployment")

            # Then
            logs = filter_logs(
                caplog, expected_event, expected_log_level, expected_log_message_parts
            )

            assert len(logs) == 1

    @pytest.mark.parametrize(
        "config,"
        "expected_deployment,"
        "expected_api_base,"
        "expected_api_version,"
        "expected_extra_parameters",
        [
            (
                {
                    "deployment": "test_deployment_name",
                    "api_base": "https://my.api.base.com/my_model",
                    "api_version": "2023-01-01",
                    "temperature": 0.2,
                    "max_tokens": 1000,
                },
                "test_deployment_name",
                "https://my.api.base.com/my_model",
                "2023-01-01",
                {"temperature": 0.2, "max_tokens": 1000},
            ),
            # Use deprecated aliases for keys
            (
                {
                    "deployment_name": "test_deployment_name",
                    "openai_api_base": "https://my.api.base.com/my_model",
                    "openai_api_version": "2023-01-01",
                },
                "test_deployment_name",
                "https://my.api.base.com/my_model",
                "2023-01-01",
                {},
            ),
            (
                {
                    "engine": "test_deployment_name",
                    "openai_api_base": "https://my.api.base.com/my_model",
                    "openai_api_version": "2023-01-01",
                },
                "test_deployment_name",
                "https://my.api.base.com/my_model",
                "2023-01-01",
                {},
            ),
        ],
    )
    def test_from_config(
        self,
        config: dict,
        expected_deployment: str,
        expected_api_base: str,
        expected_api_version: str,
        expected_extra_parameters: dict,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")

        # When
        client = AzureOpenAILLMClient.from_config(config)

        # Then
        assert client.deployment == expected_deployment
        assert client.api_base == expected_api_base
        assert client.api_version == expected_api_version
        assert len(client._extra_parameters) == len(expected_extra_parameters)
        for parameter_key, parameter_value in expected_extra_parameters.items():
            assert parameter_key in client._extra_parameters
            assert client._extra_parameters[parameter_key] == parameter_value

    def test_completion(self, monkeypatch: MonkeyPatch):
        # Given
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")
        test_prompt = "Hello, this is a test prompt."
        test_response = "Hello, this is mocked response!"

        client = AzureOpenAILLMClient(
            deployment="test_deployment",
            api_base="https://my.api.base.com/my_model",
            api_version="2023-01-01",
            mock_response=test_response,
        )

        # When
        response = client.completion([test_prompt])

        # Then
        assert response.choices == [test_response]
        assert response.model == client.deployment
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    async def test_acompletion(self, monkeypatch: MonkeyPatch):
        # Given
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")
        test_prompt = "Hello, this is a test prompt."
        test_response = "Hello, this is mocked response!"

        client = AzureOpenAILLMClient(
            deployment="test_deployment",
            api_base="https://my.api.base.com/my_model",
            api_version="2023-01-01",
            mock_response=test_response,
        )

        # When
        response = await client.acompletion([test_prompt])

        # Then
        assert response.choices == [test_response]
        assert response.model == client.deployment
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0
