from typing import Any, Dict
from unittest.mock import patch
import pytest
import structlog

import litellm
from litellm import Router

from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.embedding.litellm_router_embedding_client import (
    LiteLLMRouterEmbeddingClient,
)
from rasa.shared.providers.router.router_client import RouterClient


class TestLiteLLMRouterEmbeddingClient:
    @pytest.fixture
    def client(self) -> LiteLLMRouterEmbeddingClient:
        config = {
            "id": "test-model-group-id",
            "models": [
                {"provider": "cohere", "model": "test-cohere", "api_key": "test"},
                {"provider": "openai", "model": "gpt-4", "api_key": "test"},
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "api_key": "test",
                    "api_base": "test-api-base",
                },
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "model": "test-model",
                    "api_base": "test-api-base",
                    "api_type": "test-api-type",
                    "api_version": "test-api-version",
                    "timeout": 10,
                    "api_key": "test",
                },
                {
                    "provider": "self-hosted",
                    "model": "test-model",
                    "api_base": "test-api-base",
                    "api_key": "test",
                    "api_version": "test-api-version",
                },
            ],
            "router": {"routing_strategy": "test"},
        }
        return LiteLLMRouterEmbeddingClient.from_config(config)

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

    def test_config(self, client: LiteLLMRouterEmbeddingClient) -> None:
        assert client.config == {
            "id": "test-model-group-id",
            "model_list": [
                {
                    "model_name": "test-model-group-id",
                    "litellm_params": {
                        "model": "cohere/test-cohere",
                        "api_key": "test",
                    },
                },
                {
                    "model_name": "test-model-group-id",
                    "litellm_params": {"model": "openai/gpt-4", "api_key": "test"},
                },
                {
                    "model_name": "test-model-group-id",
                    "litellm_params": {
                        "model": "azure/test-deployment",
                        "api_key": "test",
                        "api_base": "test-api-base",
                    },
                },
                {
                    "model_name": "test-model-group-id",
                    "litellm_params": {
                        "model": "azure/test-deployment",
                        "api_base": "test-api-base",
                        "api_version": "test-api-version",
                        "timeout": 10,
                        "api_key": "test",
                    },
                },
                {
                    "model_name": "test-model-group-id",
                    "litellm_params": {
                        "model": "hosted_vllm/test-model",
                        "api_key": "test",
                        "api_base": "test-api-base",
                        "api_version": "test-api-version",
                    },
                },
            ],
            "router": {"routing_strategy": "test"},
            "use_chat_completions_endpoint": True,
        }

    def test_model_group_id(self, client: LiteLLMRouterEmbeddingClient) -> None:
        assert client.model_group_id == "test-model-group-id"

    def test_model_configurations(self, client: LiteLLMRouterEmbeddingClient) -> None:
        assert client.model_configurations == [
            {
                "model_name": "test-model-group-id",
                "litellm_params": {"model": "cohere/test-cohere", "api_key": "test"},
            },
            {
                "model_name": "test-model-group-id",
                "litellm_params": {"model": "openai/gpt-4", "api_key": "test"},
            },
            {
                "model_name": "test-model-group-id",
                "litellm_params": {
                    "model": "azure/test-deployment",
                    "api_key": "test",
                    "api_base": "test-api-base",
                },
            },
            {
                "model_name": "test-model-group-id",
                "litellm_params": {
                    "model": "azure/test-deployment",
                    "api_base": "test-api-base",
                    "api_version": "test-api-version",
                    "timeout": 10,
                    "api_key": "test",
                },
            },
            {
                "model_name": "test-model-group-id",
                "litellm_params": {
                    "model": "hosted_vllm/test-model",
                    "api_key": "test",
                    "api_base": "test-api-base",
                    "api_version": "test-api-version",
                },
            },
        ]

    def test_router_settings(self, client: LiteLLMRouterEmbeddingClient) -> None:
        assert client.router_settings == {"routing_strategy": "test"}

    def test_router_client(self, client: LiteLLMRouterEmbeddingClient) -> None:
        assert isinstance(client.router_client, Router)

    def test_embedding_fn_args(self, client: LiteLLMRouterEmbeddingClient) -> None:
        assert client._embedding_fn_args == {"model": "test-model-group-id"}

    def test_conforms_to_protocol(self, client: LiteLLMRouterEmbeddingClient) -> None:
        assert isinstance(client, EmbeddingClient)
        assert isinstance(client, RouterClient)

    @pytest.mark.parametrize(
        "config",
        [
            {
                "id": "test-model-group-id",
                "models": [
                    {
                        "provider": "cohere",
                        "model": "test-cohere",
                        "api_key": "test",
                        "n": 10,
                    },
                ],
                "router": {},
            },
            {
                "id": "test-model-group-id",
                "models": [
                    {
                        "provider": "cohere",
                        "model": "test-cohere",
                        "api_key": "test",
                        "stream": 10,
                    },
                ],
                "router": {},
            },
        ],
    )
    def test_init_with_forbidden_keys(self, config: dict) -> None:
        with pytest.raises(ValueError), structlog.testing.capture_logs() as caplog:
            LiteLLMRouterEmbeddingClient.from_config(config)

        found_validation_log = False
        for record in caplog:
            if record["event"] == "validate_forbidden_keys":
                found_validation_log = True
                break

        assert found_validation_log

    @patch.object(LiteLLMRouterEmbeddingClient, "embed")
    def test_embed(
        self,
        embed_mock,
        client: LiteLLMRouterEmbeddingClient,
        embedding_response,
    ) -> None:
        # Given
        test_doc = "this is a test doc."
        embed_mock.return_value = embedding_response

        # When
        response = client.embed([test_doc])

        # Then
        assert response.model == "gpt-1000"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30

    @patch.object(LiteLLMRouterEmbeddingClient, "aembed")
    async def test_aembed(
        self,
        embed_mock,
        client: LiteLLMRouterEmbeddingClient,
        embedding_response,
    ) -> None:
        # Given
        test_doc = "this is a test doc."
        embed_mock.return_value = embedding_response

        # When
        response = await client.aembed([test_doc])

        # Then
        assert response.model == "gpt-1000"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30


@pytest.mark.parametrize(
    "config",
    [
        # Missing "api_key"
        {
            "id": "test-model-group-id",
            "models": [
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    # "api_key" missing
                    "api_base": "https://example.azure.com",
                }
            ],
            "router": {"routing_strategy": "test"},
        },
        # Missing "api_base"
        {
            "id": "test-model-group-id",
            "models": [
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "api_key": "test",
                    # "api_base" missing
                }
            ],
            "router": {"routing_strategy": "test"},
        },
    ],
)
def test_missing_keys_in_config_raises_validation_error(config: Dict[str, Any]) -> None:
    """Test that missing required config keys raise a ProviderClientValidationError."""
    with pytest.raises(ProviderClientValidationError):
        LiteLLMRouterEmbeddingClient.from_config(config)


@pytest.mark.parametrize(
    "config",
    [
        # Missing "deployment" for azure model
        {
            "id": "test-model-group-id",
            "models": [
                {
                    "provider": "azure",
                    # "deployment" missing
                    "api_key": "test",
                    "api_base": "https://example.azure.com",
                }
            ],
            "router": {"routing_strategy": "test"},
        },
        # Missing "model" for openai model
        {
            "id": "test-model-group-id",
            "models": [
                {
                    "provider": "openai",
                    # "model" missing
                    "deployment": "test-deployment",
                    "api_key": "test",
                    "api_base": "https://example.azure.com",
                }
            ],
            "router": {"routing_strategy": "test"},
        },
        # Missing "router" entirely
        {
            "id": "test-model-group-id",
            "models": [
                {
                    "provider": "azure",
                    "deployment": "test-deployment",
                    "api_key": "test",
                    "api_base": "https://example.azure.com",
                }
            ],
            # "router" missing
        },
        # Missing "id"
        {
            "models": [
                {
                    "provider": "openai",
                    "model": "test-deployment",
                    "api_key": "test",
                    "api_base": "https://example.azure.com",
                }
            ],
            "router": {},
        },
        # Missing "provider"
        {
            "id": "test-model-group-id",
            "models": [
                {
                    "model": "test-deployment",
                    "api_key": "test",
                    "api_base": "https://example.azure.com",
                }
            ],
            "router": {},
        },
    ],
)
def test_missing_keys_in_config_raises_value_error(config: Dict[str, Any]) -> None:
    """Test that missing required config keys raise a ValueError."""
    with pytest.raises(ValueError):
        LiteLLMRouterEmbeddingClient.from_config(config)


def test_passing_unsupported_config_parameter_does_not_raise_error() -> None:
    """Test that passing unsupported config parameters does not raise an error."""
    config = {
        "id": "test-model-group-id",
        "models": [
            {
                "provider": "cohere",
                "model": "test-cohere",
                "api_key": "test",
            },
        ],
        "router": {"routing_strategy": "test"},
        "unsupported_key": "value",
    }
    LiteLLMRouterEmbeddingClient.from_config(config)


def test_passing_unsupported_model_config_parameter_does_not_raise_error() -> None:
    """Test that passing unsupported model config parameters does not raise an error."""
    config = {
        "id": "test-model-group-id",
        "models": [
            {
                "provider": "cohere",
                "model": "test-cohere",
                "api_key": "test",
                "unsupported_key": "value",
            },
        ],
        "router": {"routing_strategy": "test"},
    }
    LiteLLMRouterEmbeddingClient.from_config(config)
