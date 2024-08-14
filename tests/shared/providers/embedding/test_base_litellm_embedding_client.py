import random
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock

import litellm
import pytest
from pytest import MonkeyPatch

from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.providers.embedding._base_litellm_embedding_client import (
    _BaseLiteLLMEmbeddingClient,
)
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.embedding.embedding_response import EmbeddingResponse
from rasa.shared.providers.llm.llm_client import LLMClient


class TestLiteLLMEmbeddingClient(_BaseLiteLLMEmbeddingClient):
    def __init__(self):
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "_BaseLiteLLMEmbeddingClient":
        return cls()

    @property
    def config(self) -> dict:
        return {}

    @property
    def _litellm_model_name(self) -> str:
        return "openai/test_embedding_model"

    @property
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        return {"test_parameter": "test_value"}

    def validate_client_setup(self) -> None:
        pass


class TestBaseLiteLLMEmbeddingClient:
    @pytest.fixture
    def client(self) -> TestLiteLLMEmbeddingClient:
        return TestLiteLLMEmbeddingClient()

    @pytest.fixture
    def litellm_embedding_response(
        self, client: LLMClient
    ) -> litellm.EmbeddingResponse:
        return litellm.EmbeddingResponse(
            model="test_embedding_model",
            data=[
                {
                    "embedding": [random.uniform(0, 1) for _ in range(100)],
                    "index": 0,
                    "object": "embedding",
                }
            ],
            usage=litellm.Usage(
                prompt_tokens=100, total_tokens=100, completion_tokens=0
            ),
        )

    @pytest.fixture
    def mock_embed(
        self,
        monkeypatch: MonkeyPatch,
        litellm_embedding_response: litellm.EmbeddingResponse,
    ) -> Mock:
        # Create a mock object
        mock = Mock(return_value=litellm_embedding_response)
        # Replace the 'completion' function in its module with the mock
        monkeypatch.setattr(
            "rasa.shared.providers.embedding"
            "._base_litellm_embedding_client.embedding",
            mock,
        )
        return mock

    @pytest.fixture
    def mock_aembed(
        self,
        monkeypatch: MonkeyPatch,
        litellm_embedding_response: litellm.EmbeddingResponse,
    ) -> AsyncMock:
        # Create a mock object
        mock = AsyncMock(return_value=litellm_embedding_response)
        # Replace the 'completion' function in its module with the mock
        monkeypatch.setattr(
            "rasa.shared.providers.embedding"
            "._base_litellm_embedding_client.aembedding",
            mock,
        )
        return mock

    def test_format_response(
        self,
        client: TestLiteLLMEmbeddingClient,
        litellm_embedding_response: litellm.EmbeddingResponse,
    ):
        # Given
        expected_embeddings = [litellm_embedding_response.data[0]["embedding"]]

        # When
        formated_response = client._format_response(litellm_embedding_response)

        # Then
        assert formated_response.model == "test_embedding_model"
        assert formated_response.data == expected_embeddings
        assert formated_response.usage.prompt_tokens == 100
        assert formated_response.usage.completion_tokens == 0
        assert formated_response.usage.total_tokens == 100

    @pytest.mark.parametrize("usage", (None, litellm.Usage()))
    def test_format_response_with_uninitialized_usage(
        self, usage: Optional[litellm.Usage], client: TestLiteLLMEmbeddingClient
    ):
        # Given
        embeddings = [random.uniform(0, 1) for _ in range(100)]
        model_response = litellm.EmbeddingResponse(
            model="test_embedding_model",
            data=[{"embedding": embeddings, "index": 0, "object": "embedding"}],
            usage=usage,
        )

        # When
        formated_response = client._format_response(model_response)

        # Then
        assert formated_response.model == "test_embedding_model"
        assert formated_response.data == [embeddings]
        assert formated_response.usage is not None
        assert formated_response.usage.prompt_tokens == 0
        assert formated_response.usage.completion_tokens == 0
        assert formated_response.usage.total_tokens == 0

    def test_conforms_to_protocol(self, client):
        assert isinstance(client, EmbeddingClient)

    def test_embed(
        self,
        client: TestLiteLLMEmbeddingClient,
        litellm_embedding_response: litellm.EmbeddingResponse,
        mock_embed: Mock,
    ):
        # Given
        documents = ["Test document for embedding"]

        # When
        response = client.embed(documents)

        # Then
        mock_embed.assert_called_once_with(
            input=documents,
            model=client._litellm_model_name,
            test_parameter="test_value",
        )
        assert isinstance(response, EmbeddingResponse)
        assert response.data == [litellm_embedding_response.data[0]["embedding"]]

    def test_embed_encounters_an_error(
        self, client: EmbeddingClient, mock_embed: Mock
    ) -> None:
        mock_embed.side_effect = Exception("API exception raised!")
        with pytest.raises(ProviderClientAPIException):
            client.embed(["test message"])

    async def test_aembed(
        self,
        client: TestLiteLLMEmbeddingClient,
        litellm_embedding_response: litellm.EmbeddingResponse,
        mock_aembed: AsyncMock,
    ):
        # Given
        documents = ["Test document for embedding"]

        # When
        response = await client.aembed(documents)

        # Then
        mock_aembed.assert_called_once_with(
            input=documents,
            model=client._litellm_model_name,
            test_parameter="test_value",
        )
        assert isinstance(response, EmbeddingResponse)
        assert response.data == [litellm_embedding_response.data[0]["embedding"]]

    async def test_aembed_encounters_an_error(
        self, client: EmbeddingClient, mock_aembed: Mock
    ) -> None:
        mock_aembed.side_effect = Exception("API exception raised!")
        with pytest.raises(ProviderClientAPIException):
            await client.aembed(["test message"])
