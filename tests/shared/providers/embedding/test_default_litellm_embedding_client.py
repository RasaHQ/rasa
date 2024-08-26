from unittest.mock import patch

import litellm
import pytest
import structlog
from pytest import MonkeyPatch
from rasa.shared.providers.embedding.default_litellm_embedding_client import (
    DefaultLiteLLMEmbeddingClient,
)
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient


class TestDefaultLiteLLMEmbeddingClient:
    @pytest.fixture
    def client(self) -> DefaultLiteLLMEmbeddingClient:
        config = {
            "model": "buzz-ai/mock-gpt",
            "extra_param": "abc",
            "extra_param2": "def",
        }
        return DefaultLiteLLMEmbeddingClient.from_config(config)

    @pytest.fixture
    def embedding_response(self) -> litellm.EmbeddingResponse:
        return litellm.EmbeddingResponse(
            model="mock-gpt",
            data=[
                {"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"},
                {"embedding": [0.4, 0.5, 0.6], "index": 1, "object": "embedding"},
            ],
            usage=litellm.Usage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
        )

    def test_config(self, client: DefaultLiteLLMEmbeddingClient) -> None:
        assert client.config == {
            "model": "buzz-ai/mock-gpt",
            "extra_param": "abc",
            "extra_param2": "def",
        }

    def test_model(self, client: DefaultLiteLLMEmbeddingClient) -> None:
        assert client.model == "buzz-ai/mock-gpt"

    def test_litellm_extra_parameters(
        self, client: DefaultLiteLLMEmbeddingClient
    ) -> None:
        assert client._litellm_extra_parameters == {
            "extra_param": "abc",
            "extra_param2": "def",
        }

    def test_litellm_model_name(self, client: DefaultLiteLLMEmbeddingClient) -> None:
        assert client._litellm_model_name == "buzz-ai/mock-gpt"

    def test_embedding_fn_args(self, client: DefaultLiteLLMEmbeddingClient) -> None:
        assert client._embedding_fn_args == {
            "model": "buzz-ai/mock-gpt",
            "extra_param": "abc",
            "extra_param2": "def",
        }

    def test_validate_documents_pass(
        self, client: DefaultLiteLLMEmbeddingClient
    ) -> None:
        documents = ["Hello", "World"]
        assert client.validate_documents(documents) is None

    def test_validate_documents_raises_error_due_to_empty_documents(
        self, client: DefaultLiteLLMEmbeddingClient
    ) -> None:
        documents = ["   "]
        with pytest.raises(
            ValueError, match="Documents cannot be empty or whitespace."
        ):
            client.validate_documents(documents)

    def test_validate_documents_raises_error_due_to_non_strings(
        self, client: DefaultLiteLLMEmbeddingClient
    ) -> None:
        documents = ["hello", 1]
        with pytest.raises(ValueError, match="All documents must be strings."):
            client.validate_documents(documents)

    def test_conforms_to_protocol(self, client: DefaultLiteLLMEmbeddingClient) -> None:
        assert isinstance(client, EmbeddingClient)

    def test_default_litellm_embedding_client_validate_client_setup_success(
        self,
        client: DefaultLiteLLMEmbeddingClient,
    ) -> None:
        client.validate_client_setup()

    def test_default_litellm_embedding_client_embed(
        self,
        client: DefaultLiteLLMEmbeddingClient,
        embedding_response: litellm.EmbeddingResponse,
    ) -> None:
        # Given
        test_doc = "this is a test doc."

        # When
        with patch(
            "rasa.shared.providers.embedding._base_litellm_embedding_client.embedding",
            return_value=embedding_response,
        ):
            response = client.embed([test_doc])

        # Then
        assert response.data == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert response.model == "mock-gpt"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30

    async def test_default_litellm_embedding_client_aembed(
        self,
        client: DefaultLiteLLMEmbeddingClient,
        embedding_response: litellm.EmbeddingResponse,
    ) -> None:
        # Given
        test_doc = "this is a test doc."

        # When
        with patch(
            "rasa.shared.providers.embedding._base_litellm_embedding_client.aembedding",
            return_value=embedding_response,
        ):
            response = await client.aembed([test_doc])

        # Then
        assert response.data == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert response.model == "mock-gpt"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30

    @pytest.mark.parametrize(
        "config",
        [
            {
                "model": "cohere/test-cohere",
                # Stream is forbidden
                "stream": True,
            },
            {
                "model": "cohere/test-cohere",
                # n is forbidden
                "n": 10,
            },
        ],
    )
    def test_default_embedding_cannot_be_instantiated_with_forbidden_keys(
        self,
        config: dict,
        monkeypatch: MonkeyPatch,
    ):
        with pytest.raises(ValueError), structlog.testing.capture_logs() as caplog:
            DefaultLiteLLMEmbeddingClient.from_config(config)

        found_validation_log = False
        for record in caplog:
            if record["event"] == "validate_forbidden_keys":
                found_validation_log = True
                break

        assert found_validation_log
