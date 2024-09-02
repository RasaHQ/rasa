import os
from unittest.mock import patch

import litellm
import pytest
import structlog
from pytest import MonkeyPatch
from rasa.shared.constants import (
    AZURE_API_BASE_ENV_VAR,
    AZURE_API_KEY_ENV_VAR,
    AZURE_API_TYPE_ENV_VAR,
    AZURE_API_VERSION_ENV_VAR,
    OPENAI_API_BASE_ENV_VAR,
    OPENAI_API_KEY_ENV_VAR,
    OPENAI_API_VERSION_ENV_VAR,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers.embedding.azure_openai_embedding_client import (
    AzureOpenAIEmbeddingClient,
)
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from tests.utilities import filter_logs


class TestAzureOpenAIEmbeddingClient:
    @pytest.fixture
    def client(self, monkeypatch: MonkeyPatch) -> AzureOpenAIEmbeddingClient:
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")
        config = {
            "provider": "azure",
            "deployment": "some_azure_deployment",
            "model": "gpt-2024",
            "api_base": "https://test",
            "api_version": "v1",
            "api_type": "azure",
        }
        return AzureOpenAIEmbeddingClient.from_config(config)

    @pytest.fixture
    def embedding_response(self) -> litellm.EmbeddingResponse:
        return litellm.EmbeddingResponse(
            model="gpt-2024",
            data=[
                {"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"},
                {"embedding": [0.4, 0.5, 0.6], "index": 1, "object": "embedding"},
            ],
            usage=litellm.Usage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
        )

    def test_config(self, client: AzureOpenAIEmbeddingClient) -> None:
        assert client.config == {
            "provider": "azure",
            "deployment": "some_azure_deployment",
            "model": "gpt-2024",
            "api_base": "https://test",
            "api_type": "azure",
            "api_version": "v1",
        }

    def test_model(self, client: AzureOpenAIEmbeddingClient) -> None:
        assert client.model == "gpt-2024"

    def test_deployment(self, client: AzureOpenAIEmbeddingClient) -> None:
        assert client.deployment == "some_azure_deployment"

    def test_api_base(self, client: AzureOpenAIEmbeddingClient) -> None:
        assert client.api_base == "https://test"

    def test_api_type(self, client: AzureOpenAIEmbeddingClient) -> None:
        assert client.api_type == "azure"

    def test_api_version(self, client: AzureOpenAIEmbeddingClient) -> None:
        assert client.api_version == "v1"

    def test_model_parameters(self, client: AzureOpenAIEmbeddingClient) -> None:
        assert client._litellm_extra_parameters == {}

    def test_get_litellm_model_name(self, client: AzureOpenAIEmbeddingClient) -> None:
        assert client._litellm_model_name == "azure/some_azure_deployment"

    def test_embedding_fn_args(self, client: AzureOpenAIEmbeddingClient) -> None:
        assert client._embedding_fn_args == {
            "api_base": "https://test",
            "api_type": "azure",
            "api_version": "v1",
            "model": "azure/some_azure_deployment",
            # API key is added through environment variable
            "api_key": "my key",
        }

    def test_validate_documents_pass(self, client: AzureOpenAIEmbeddingClient) -> None:
        documents = ["Hello", "World"]
        assert client.validate_documents(documents) is None

    def test_validate_documents_raises_error_due_to_empty_documents(
        self, client: AzureOpenAIEmbeddingClient
    ) -> None:
        documents = ["   "]
        with pytest.raises(
            ValueError, match="Documents cannot be empty or whitespace."
        ):
            client.validate_documents(documents)

    def test_validate_documents_raises_error_due_to_non_strings(
        self, client: AzureOpenAIEmbeddingClient
    ) -> None:
        documents = ["hello", 1]
        with pytest.raises(ValueError, match="All documents must be strings."):
            client.validate_documents(documents)

    def test_conforms_to_protocol(self, client: AzureOpenAIEmbeddingClient) -> None:
        assert isinstance(client, EmbeddingClient)

    def test_embed(
        self,
        client: AzureOpenAIEmbeddingClient,
        embedding_response: litellm.EmbeddingResponse,
        monkeypatch: MonkeyPatch,
    ) -> None:
        # Given
        test_doc = "this is a test doc."
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")

        # When
        with patch(
            "rasa.shared.providers.embedding._base_litellm_embedding_client.embedding",
            return_value=embedding_response,
        ):
            response = client.embed([test_doc])

        # Then
        assert response.data == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert response.model == "gpt-2024"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30

    async def test_aembed(
        self,
        client: AzureOpenAIEmbeddingClient,
        embedding_response: litellm.EmbeddingResponse,
        monkeypatch: MonkeyPatch,
    ) -> None:
        # Given
        test_doc = "this is a test doc."
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")

        # When
        with patch(
            "rasa.shared.providers.embedding._base_litellm_embedding_client.aembedding",
            return_value=embedding_response,
        ):
            response = await client.aembed([test_doc])

        # Then
        assert response.data == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert response.model == "gpt-2024"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30

    def test_from_config_successful(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        # Given
        monkeypatch.setenv(AZURE_API_BASE_ENV_VAR, "https://test")
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "key")
        monkeypatch.setenv(AZURE_API_VERSION_ENV_VAR, "some_version")
        config = {
            "provider": "azure",
            "deployment": "some_azure_deployment",
            "model": "gpt-2024",
        }

        # When
        with pytest.warns(None) as record:
            client = AzureOpenAIEmbeddingClient.from_config(config)

        # Then
        assert len(record) == 0

        # Then
        assert os.environ.get(AZURE_API_BASE_ENV_VAR) == "https://test"
        assert os.environ.get(AZURE_API_KEY_ENV_VAR) == "key"
        assert os.environ.get(AZURE_API_VERSION_ENV_VAR) == "some_version"
        assert client.api_base == "https://test"
        assert client.api_type == "azure"
        assert client.api_version == "some_version"

    def test_from_config_with_deprecated_env_vars(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        # Given
        monkeypatch.setenv(OPENAI_API_BASE_ENV_VAR, "https://test")
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "key")
        monkeypatch.setenv(OPENAI_API_VERSION_ENV_VAR, "deprecated_v1")
        config = {
            "provider": "azure",
            "deployment": "some_azure_deployment",
            "model": "gpt-2024",
        }

        # When
        with pytest.warns(FutureWarning) as record:
            client = AzureOpenAIEmbeddingClient.from_config(config)

        # Then
        assert len(record) == 3
        assert record[0].message.args[0] == (
            "Usage of OPENAI_API_BASE environment variable for setting the API base of "
            "Azure OpenAI client is deprecated and will be removed in 4.0.0. Please "
            "use AZURE_API_BASE instead."
        )
        assert record[1].message.args[0] == (
            "Usage of OPENAI_API_KEY environment variable for setting the API key of "
            "Azure OpenAI client is deprecated and will be removed in 4.0.0. Please "
            "use AZURE_API_KEY instead."
        )
        assert record[2].message.args[0] == (
            "Usage of OPENAI_API_VERSION environment variable for setting the API "
            "version of Azure OpenAI client is deprecated and will be removed in 4.0.0."
            " Please use AZURE_API_VERSION instead."
        )
        assert client.api_base == "https://test"
        assert client.api_type == "azure"
        assert client.api_version == "deprecated_v1"
        assert os.environ.get(OPENAI_API_KEY_ENV_VAR) == "key"

    def test_from_config_with_combined_deprecated_and_new_env_vars(
        self,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        monkeypatch.setenv(OPENAI_API_BASE_ENV_VAR, "https://test")
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "key")
        monkeypatch.setenv(OPENAI_API_VERSION_ENV_VAR, "deprecated_v1")

        monkeypatch.setenv(AZURE_API_BASE_ENV_VAR, "https://test")
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "key")
        monkeypatch.setenv(AZURE_API_VERSION_ENV_VAR, "deprecated_v1")

        config = {
            "provider": "azure",
            "deployment": "some_azure_deployment",
            "model": "gpt-2024",
        }

        # When
        with pytest.warns(None) as record:
            client = AzureOpenAIEmbeddingClient.from_config(config)

        future_warnings = [
            warning for warning in record if warning.category == FutureWarning
        ]

        # Then
        assert len(future_warnings) == 0
        assert client.api_base == "https://test"
        assert client.api_type == "azure"
        assert client.api_version == "deprecated_v1"
        assert os.environ.get(OPENAI_API_KEY_ENV_VAR) == "key"

    def test_from_config_throws_validation_error(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Given
        # Did not set the required environment variables
        # api_base, api_key, api_version
        config = {
            "provider": "azure",
            "deployment": "some_azure_deployment",
            "model": "gpt-2024",
        }

        # When
        with pytest.raises(
            ProviderClientValidationError
        ) as exc, structlog.testing.capture_logs() as caplog:
            AzureOpenAIEmbeddingClient.from_config(config)

        expected_event = "azure_openai_embedding_client.validate_environment_variables"
        expected_log_level = "error"
        logs = filter_logs(caplog, expected_event, expected_log_level, [])

        # Then
        assert len(logs) == 3
        event_info_string = " ".join([log["event_info"] for log in logs])
        assert "AZURE_API_BASE" in event_info_string
        assert "AZURE_API_KEY" in event_info_string
        assert "AZURE_API_VERSION" in event_info_string
        assert str(exc.value) == (
            "Missing required environment variables/config keys for API calls."
        )

    @pytest.mark.parametrize("deprecated_provider_alias", ["type", "_type"])
    def test_from_config_raises_deprecation_warning_for_deprecated_provider_aliases(
        self, deprecated_provider_alias: str, monkeypatch: MonkeyPatch
    ):
        # Given
        monkeypatch.setenv(
            AZURE_API_KEY_ENV_VAR,
            "test key for "
            "test_azure_openai_embedding_client_from_config_validation_error_2",
        )
        config = {
            deprecated_provider_alias: "azure",
            "deployment": "some_azure_deployment",
            "api_base": "https://test",
            "api_version": "v1",
        }

        # When
        with pytest.warns(None) as record:
            client = AzureOpenAIEmbeddingClient.from_config(config)

        # Then
        future_warnings = [
            warning for warning in record if warning.category == FutureWarning
        ]
        assert len(future_warnings) == 1
        assert f"'{deprecated_provider_alias}' is deprecated" in str(
            future_warnings[0].message
        )
        assert "'provider' instead" in str(future_warnings[0].message)

        assert client.config["provider"] == "azure"
        assert deprecated_provider_alias not in client.config

    def test_init_with_params(self, monkeypatch: MonkeyPatch) -> None:
        # Given
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "some_key")
        client = AzureOpenAIEmbeddingClient(
            deployment="some_azure_deployment",
            model="gpt-2024",
            api_base="https://test",
            api_type="test",
            api_version="v1",
        )

        # Then
        assert client.deployment == "some_azure_deployment"
        assert client.model == "gpt-2024"
        assert client.api_base == "https://test"
        assert client.api_type == "test"
        assert client.api_version == "v1"
        assert client._litellm_extra_parameters == {}

    def test_init_with_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        # Given
        monkeypatch.setenv(AZURE_API_BASE_ENV_VAR, "https://test/env")
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "some_key")
        monkeypatch.setenv(AZURE_API_TYPE_ENV_VAR, "env_test")
        monkeypatch.setenv(AZURE_API_VERSION_ENV_VAR, "env_v1")

        # When
        client = AzureOpenAIEmbeddingClient(deployment="test_dep", model="gpt-2024")

        # Then
        assert client.api_base == "https://test/env"
        assert client.api_type == "env_test"
        assert client.api_version == "env_v1"
        assert client.model == "gpt-2024"
        assert client._litellm_extra_parameters == {}
        assert os.environ.get(AZURE_API_KEY_ENV_VAR) == "some_key"

    @pytest.mark.parametrize(
        "config",
        [
            {
                "provider": "azure",
                "deployment": "some_azure_deployment",
                "api_base": "https://test",
                "api_version": "v1",
                # Stream is forbidden
                "stream": True,
            },
            {
                "provider": "azure",
                "deployment": "some_azure_deployment",
                "api_base": "https://test",
                "api_version": "v1",
                # n is forbidden
                "n": 10,
            },
        ],
    )
    def test_azure_openai_embedding_cannot_be_instantiated_with_forbidden_keys(
        self,
        config: dict,
        monkeypatch: MonkeyPatch,
    ):
        with pytest.raises(ValueError), structlog.testing.capture_logs() as caplog:
            AzureOpenAIEmbeddingClient.from_config(config)

        found_validation_log = False
        for record in caplog:
            if record["event"] == "validate_forbidden_keys":
                found_validation_log = True
                break

        assert found_validation_log

    @pytest.mark.parametrize(
        "config, expected_to_raise_deprecation_warning",
        [
            (
                {
                    "provider": "azure",
                    "deployment": "some_azure_deployment",
                    "model": "test-embedding",
                    "api_base": "https://test",
                    "api_version": "2023-05-15",
                    "timeout": 7,
                },
                False,
            ),
            (
                {
                    "provider": "azure",
                    "deployment": "some_azure_deployment",
                    "model": "test-embedding",
                    "api_base": "https://test",
                    "api_version": "2023-05-15",
                    # Use deprecated key for timeout
                    "request_timeout": 7,
                },
                True,
            ),
        ],
    )
    def test_from_config_correctly_initializes_timeout(
        self,
        config,
        expected_to_raise_deprecation_warning: bool,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "some_key")

        # When
        with pytest.warns(None) as record:
            client = AzureOpenAIEmbeddingClient.from_config(config)

        # Then
        future_warnings = [
            warning for warning in record if warning.category == FutureWarning
        ]
        if expected_to_raise_deprecation_warning:
            assert len(future_warnings) == 1
            assert "timeout" in str(future_warnings[0].message)
            assert "request_timeout" in str(future_warnings[0].message)

        assert "timeout" in client._extra_parameters
        assert client._extra_parameters["timeout"] == 7
