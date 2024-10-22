import os
from unittest.mock import patch

import litellm
import pytest
import structlog
from pytest import MonkeyPatch
from rasa.shared.constants import (
    OPENAI_API_BASE_ENV_VAR,
    OPENAI_API_KEY_ENV_VAR,
    OPENAI_API_TYPE_ENV_VAR,
    OPENAI_API_VERSION_ENV_VAR,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.embedding.openai_embedding_client import (
    OpenAIEmbeddingClient,
)


class TestOpenAIEmbeddingClient:
    @pytest.fixture
    def client(self, monkeypatch: MonkeyPatch) -> OpenAIEmbeddingClient:
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        config = {
            "provider": "openai",
            "api_base": "https://test",
            "api_type": "openai",
            "api_version": "v1",
            "model": "gpt-1000",
        }
        return OpenAIEmbeddingClient.from_config(config)

    @pytest.fixture
    def embedding_response(self) -> litellm.EmbeddingResponse:
        return litellm.EmbeddingResponse(
            model="gpt-1000",
            data=[
                {"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"},
                {"embedding": [0.4, 0.5, 0.6], "index": 1, "object": "embedding"},
            ],
            usage=litellm.Usage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
        )

    def test_config(self, client: OpenAIEmbeddingClient) -> None:
        assert client.config == {
            "provider": "openai",
            "api_base": "https://test",
            "api_version": "v1",
            "api_type": "openai",
            "model": "gpt-1000",
        }

    def test_model(self, client: OpenAIEmbeddingClient) -> None:
        assert client.model == "gpt-1000"

    def test_api_base(self, client: OpenAIEmbeddingClient) -> None:
        assert client.api_base == "https://test"

    def test_api_type(self, client: OpenAIEmbeddingClient) -> None:
        assert client.api_type == "openai"

    def test_api_version(self, client: OpenAIEmbeddingClient) -> None:
        assert client.api_version == "v1"

    def test_model_parameters(self, client: OpenAIEmbeddingClient) -> None:
        assert client._litellm_extra_parameters == {}

    def test_get_litellm_model_name(self, client: OpenAIEmbeddingClient) -> None:
        assert client._litellm_model_name == "openai/gpt-1000"

    def test_embedding_fn_args(self, client: OpenAIEmbeddingClient) -> None:
        assert client._embedding_fn_args == {
            "api_base": "https://test",
            "api_type": "openai",
            "api_version": "v1",
            "model": "openai/gpt-1000",
        }

    def test_validate_documents_pass(self, client: OpenAIEmbeddingClient) -> None:
        documents = ["Hello", "World"]
        assert client.validate_documents(documents) is None

    def test_validate_documents_raises_error_due_to_empty_documents(
        self, client: OpenAIEmbeddingClient
    ) -> None:
        documents = ["   "]
        with pytest.raises(
            ValueError, match="Documents cannot be empty or whitespace."
        ):
            client.validate_documents(documents)

    def test_validate_documents_raises_error_due_to_non_strings(
        self, client: OpenAIEmbeddingClient
    ) -> None:
        documents = ["hello", 1]
        with pytest.raises(ValueError, match="All documents must be strings."):
            client.validate_documents(documents)

    def test_conforms_to_protocol(self, client: OpenAIEmbeddingClient) -> None:
        assert isinstance(client, EmbeddingClient)

    def test_validate_client_setup_success(
        self, client: OpenAIEmbeddingClient, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "test100")
        client.validate_client_setup()
        assert os.environ.get(OPENAI_API_KEY_ENV_VAR) == "test100"

    def test_validate_client_setup_raises_error(
        self, client: OpenAIEmbeddingClient, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.delenv(OPENAI_API_KEY_ENV_VAR)
        with pytest.raises(ProviderClientValidationError):
            client.validate_client_setup()

    def test_embed(
        self,
        client: OpenAIEmbeddingClient,
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
        assert response.model == "gpt-1000"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30

    async def test_aembed(
        self,
        client: OpenAIEmbeddingClient,
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
        assert response.model == "gpt-1000"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30

    def test_init_with_params(self, monkeypatch: MonkeyPatch) -> None:
        # Given
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "some_key")
        client = OpenAIEmbeddingClient(
            model="gpt-1000",
            api_base="https://test",
            api_type="openai",
            api_version="v1",
        )

        # Then
        assert client.model == "gpt-1000"
        assert client.api_base == "https://test"
        assert client.api_type == "openai"
        assert client.api_version == "v1"
        assert client._litellm_extra_parameters == {}

    def test_init_with_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        # Given
        monkeypatch.setenv(OPENAI_API_BASE_ENV_VAR, "https://test/env")
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "some_key")
        monkeypatch.setenv(OPENAI_API_TYPE_ENV_VAR, "openai")
        monkeypatch.setenv(OPENAI_API_VERSION_ENV_VAR, "env_v1")

        # When
        client = OpenAIEmbeddingClient(model="gpt-1000")

        # Then
        assert client.api_base == "https://test/env"
        assert client.api_type == "openai"
        assert client.api_version == "env_v1"
        assert client.model == "gpt-1000"
        assert client._litellm_extra_parameters == {}
        assert os.environ.get(OPENAI_API_KEY_ENV_VAR) == "some_key"

    @pytest.mark.parametrize(
        "config",
        [
            {
                "provider": "openai",
                "model": "test-embedding",
                # Stream is forbidden
                "stream": True,
            },
            {
                "provider": "openai",
                "model": "test-embedding",
                # n is forbidden
                "n": 10,
            },
        ],
    )
    def test_init_with_forbidden_keys(
        self,
        config: dict,
        monkeypatch: MonkeyPatch,
    ):
        with pytest.raises(ValueError), structlog.testing.capture_logs() as caplog:
            OpenAIEmbeddingClient.from_config(config)

        found_validation_log = False
        for record in caplog:
            if record["event"] == "validate_forbidden_keys":
                found_validation_log = True
                break

        assert found_validation_log

    @pytest.mark.parametrize("deprecated_provider_alias", ["type", "_type"])
    def test_from_config_raises_deprecation_warning_for_deprecated_provider_aliases(
        self, deprecated_provider_alias: str, monkeypatch: MonkeyPatch
    ):
        # Given
        monkeypatch.setenv(
            OPENAI_API_KEY_ENV_VAR,
            "test key for "
            "test_from_config_raises_deprecation_warning_for_deprecated_provider_aliases",
        )
        config = {
            deprecated_provider_alias: "openai",
            "model": "gpt-test",
        }

        # When
        with pytest.warns(None) as record:
            client = OpenAIEmbeddingClient.from_config(config)

        # Then
        future_warnings = [
            warning for warning in record if warning.category == FutureWarning
        ]
        assert len(future_warnings) == 1
        assert f"'{deprecated_provider_alias}' is deprecated" in str(
            future_warnings[0].message
        )
        assert "'provider' instead" in str(future_warnings[0].message)

        assert client.config["provider"] == "openai"
        assert deprecated_provider_alias not in client.config

    @pytest.mark.parametrize(
        "config, expected_to_raise_deprecation_warning",
        [
            (
                {
                    "provider": "openai",
                    "model": "test-embedding",
                    "timeout": 7,
                },
                False,
            ),
            (
                {
                    "provider": "openai",
                    "model": "test-embedding",
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
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "some_key")

        # When
        with pytest.warns(None) as record:
            client = OpenAIEmbeddingClient.from_config(config)

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
