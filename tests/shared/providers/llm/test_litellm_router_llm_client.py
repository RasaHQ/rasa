import os
from typing import Any, Dict
from unittest.mock import patch

import pytest
import structlog
from litellm import Router
from pytest import MonkeyPatch

from rasa.shared.constants import (
    AZURE_API_BASE_ENV_VAR,
    AZURE_API_KEY_ENV_VAR,
    AZURE_API_VERSION_ENV_VAR,
    OPENAI_API_KEY_ENV_VAR,
    SELF_HOSTED_VLLM_API_KEY_ENV_VAR,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers.llm.litellm_router_llm_client import LiteLLMRouterLLMClient
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMUsage
from rasa.shared.providers.router.router_client import RouterClient


class TestLiteLLMRouterLLMClient:
    @pytest.fixture
    def client(self, monkeypatch: MonkeyPatch) -> LiteLLMRouterLLMClient:
        monkeypatch.setenv("COHERE_API_KEY", "dummy_key_cohere")
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "dummy_key_openai")
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "dummy_key_azure")
        monkeypatch.setenv(AZURE_API_BASE_ENV_VAR, "dummy_base_azure")
        monkeypatch.setenv(AZURE_API_VERSION_ENV_VAR, "dummy_version_azure")

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
        return LiteLLMRouterLLMClient.from_config(config)

    def test_config(self, client: LiteLLMRouterLLMClient) -> None:
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

    def test_model_group_id(self, client: LiteLLMRouterLLMClient) -> None:
        assert client.model_group_id == "test-model-group-id"

    def test_model_configurations(self, client: LiteLLMRouterLLMClient) -> None:
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

    def test_router_settings(self, client: LiteLLMRouterLLMClient) -> None:
        assert client.router_settings == {"routing_strategy": "test"}

    def test_router_client(self, client: LiteLLMRouterLLMClient) -> None:
        assert isinstance(client.router_client, Router)

    def test_completion_fn_args(self, client: LiteLLMRouterLLMClient) -> None:
        assert client._completion_fn_args == {"model": "test-model-group-id"}

    def test_conforms_to_protocol(self, client: LiteLLMRouterLLMClient) -> None:
        assert isinstance(client, LLMClient)
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
            LiteLLMRouterLLMClient.from_config(config)

        found_validation_log = False
        for record in caplog:
            if record["event"] == "validate_forbidden_keys":
                found_validation_log = True
                break

        assert found_validation_log

    @patch.object(LiteLLMRouterLLMClient, "completion")
    def test_llm_router_completion(
        self, mock_completion, client: LiteLLMRouterLLMClient
    ) -> None:
        # Given
        test_prompt = "Hello, this is a test prompt."
        test_response = "Hello, this is a mocked response!"

        # Mocking the response from the `completion` method
        mock_response = LLMResponse(
            id="mock-id",
            created=123456,
            choices=[test_response],
            model="test-model",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=7),
        )
        mock_completion.return_value = mock_response

        # When
        response = client.completion([test_prompt])

        # Then
        assert response.choices == [test_response]
        assert response.usage.prompt_tokens == 5
        assert response.usage.completion_tokens == 7
        assert response.usage.total_tokens == 12

    @patch.object(LiteLLMRouterLLMClient, "acompletion")
    async def test_llm_router_acompletion(
        self, mock_completion, client: LiteLLMRouterLLMClient
    ) -> None:
        # Given
        test_prompt = "Hello, this is a test prompt."
        test_response = "Hello, this is a mocked response!"

        # Mocking the response from the `completion` method
        mock_response = LLMResponse(
            id="mock-id",
            created=123456,
            choices=[test_response],
            model="test-model",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=7),
        )
        mock_completion.return_value = mock_response

        # When
        response = await client.acompletion([test_prompt])

        # Then
        assert response.choices == [test_response]
        assert response.usage.prompt_tokens == 5
        assert response.usage.completion_tokens == 7
        assert response.usage.total_tokens == 12

    @pytest.mark.asyncio
    async def test_acompletion_timeout_enforcement(
        self, client: LiteLLMRouterLLMClient, monkeypatch: MonkeyPatch
    ):
        """Test that timeout error message correctly shows
        'time taken' is equivalent to 'timeout value' defined in 'endpoints.yml'."""
        import asyncio
        from unittest.mock import AsyncMock, PropertyMock

        from rasa.shared.exceptions import ProviderClientAPIException

        # Set small timeout value for timeout
        timeout_value = 0.00001
        monkeypatch.setattr(
            type(client),
            "_litellm_extra_parameters",
            PropertyMock(return_value={"timeout": timeout_value}),
        )

        # Mock router_client.acompletion with a slow operation that will timeout
        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(1)

        monkeypatch.setattr(
            client.router_client,
            "acompletion",
            AsyncMock(side_effect=slow_operation),
        )

        # Verify timeout is enforced and error message is correct
        with pytest.raises(ProviderClientAPIException) as exc_info:
            await client.acompletion("test message")

        # Verify error message shows 'time taken' to be
        # equivalent to 'timeout value'
        error_message = str(exc_info.value.original_exception)
        assert "APITimeoutError" in error_message
        assert f"timeout value={timeout_value:.6f}" in error_message
        assert f"time taken={timeout_value:.6f} seconds" in error_message


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
        LiteLLMRouterLLMClient.from_config(config)


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
        LiteLLMRouterLLMClient.from_config(config)


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
    LiteLLMRouterLLMClient.from_config(config)


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
    LiteLLMRouterLLMClient.from_config(config)


def test_passing_use_chat_completions_endpoint_in_router_config() -> None:
    # Given
    config = {
        "id": "test-model-group-id",
        "models": [
            {
                "provider": "self-hosted",
                "model": "some_model",
                "api_base": "https://example.com",
                "api_key": "test",
            },
        ],
        "router": {
            "routing_strategy": "test",
            "use_chat_completions_endpoint": False,
        },
    }

    # When
    router_client = LiteLLMRouterLLMClient.from_config(config)

    # Then
    assert router_client.use_chat_completions_endpoint is False
    assert router_client.router_settings == {"routing_strategy": "test"}
    assert router_client.model_configurations == [
        {
            "model_name": "test-model-group-id",
            "litellm_params": {
                "model": "hosted_vllm/some_model",
                "api_base": "https://example.com",
                "api_key": "test",
            },
        },
    ]
    assert router_client.config == {
        "id": "test-model-group-id",
        "model_list": [
            {
                "model_name": "test-model-group-id",
                "litellm_params": {
                    "model": "hosted_vllm/some_model",
                    "api_base": "https://example.com",
                    "api_key": "test",
                },
            },
        ],
        "router": {"routing_strategy": "test"},
        "use_chat_completions_endpoint": False,
    }


def test_api_key_automatically_set_in_env_if_missing_for_self_hosted_models(
    monkeypatch: MonkeyPatch,
) -> None:
    # Given
    monkeypatch.delenv(SELF_HOSTED_VLLM_API_KEY_ENV_VAR, raising=False)
    config = {
        "id": "test-model-group-id",
        "models": [
            {
                "provider": "self-hosted",
                "model": "some_model",
                "api_base": "https://example.com",
            }
        ],
        "router": {"routing_strategy": "test"},
    }

    # When
    LiteLLMRouterLLMClient.from_config(config)

    # Then
    assert os.environ.get(SELF_HOSTED_VLLM_API_KEY_ENV_VAR) == "dummy api key"

    # Clean up
    monkeypatch.delenv(SELF_HOSTED_VLLM_API_KEY_ENV_VAR, raising=False)


def test_api_key_not_set_in_env_when_api_key_set_in_config_for_self_hosted(
    monkeypatch: MonkeyPatch,
) -> None:
    # Given
    monkeypatch.delenv(SELF_HOSTED_VLLM_API_KEY_ENV_VAR, raising=False)
    config = {
        "id": "test-model-group-id",
        "models": [
            {
                "provider": "self-hosted",
                "model": "hosted_vllm/some_model",
                "api_base": "https://example2.com",
                "api_key": "test",
            }
        ],
        "router": {"routing_strategy": "test"},
    }

    # When
    LiteLLMRouterLLMClient.from_config(config)

    # Then
    assert os.environ.get(SELF_HOSTED_VLLM_API_KEY_ENV_VAR) is None

    # Clean up
    monkeypatch.delenv(SELF_HOSTED_VLLM_API_KEY_ENV_VAR, raising=False)


def test_api_key_not_set_in_env_when_api_key_set_in_env_for_self_hosted(
    monkeypatch: MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv(SELF_HOSTED_VLLM_API_KEY_ENV_VAR, "test_litellm_router")
    config = {
        "id": "test-model-group-id",
        "models": [
            {
                "provider": "self-hosted",
                "model": "hosted_vllm/some_model",
                "api_base": "https://example2.com",
            }
        ],
        "router": {"routing_strategy": "test"},
    }

    # When
    LiteLLMRouterLLMClient.from_config(config)

    # Then
    assert os.environ.get(SELF_HOSTED_VLLM_API_KEY_ENV_VAR) == "test_litellm_router"

    # Clean up
    monkeypatch.delenv(SELF_HOSTED_VLLM_API_KEY_ENV_VAR, raising=False)
