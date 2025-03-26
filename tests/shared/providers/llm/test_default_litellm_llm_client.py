from typing import List

import pytest
import structlog
from pytest import MonkeyPatch

from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers.llm.default_litellm_llm_client import DefaultLiteLLMClient
from rasa.shared.providers.llm.llm_client import LLMClient


class TestDefaultLiteLLMEmbeddingClient:
    @pytest.fixture
    def client(self) -> DefaultLiteLLMClient:
        config = {
            "model": "mock-gpt",
            "provider": "buzz-ai",
            "extra_param": "abc",
            "extra_param2": "def",
        }
        return DefaultLiteLLMClient.from_config(config)

    def test_config(self, client: DefaultLiteLLMClient) -> None:
        assert client.config == {
            "model": "mock-gpt",
            "provider": "buzz-ai",
            "extra_param": "abc",
            "extra_param2": "def",
        }

    def test_model(self, client: DefaultLiteLLMClient) -> None:
        assert client.model == "mock-gpt"

    def test_litellm_extra_parameters(self, client: DefaultLiteLLMClient) -> None:
        assert client._litellm_extra_parameters == {
            "extra_param": "abc",
            "extra_param2": "def",
        }

    def test_litellm_model_name(self, client: DefaultLiteLLMClient) -> None:
        assert client._litellm_model_name == "buzz-ai/mock-gpt"

    def test_completion_fn_args(self, client: DefaultLiteLLMClient) -> None:
        assert client._completion_fn_args == {
            # this is the litellm_model_name
            "model": "buzz-ai/mock-gpt",
            # extra parameters
            "extra_param": "abc",
            "extra_param2": "def",
            # always present by default
            "drop_params": False,
        }

    def test_conforms_to_protocol(self, client: DefaultLiteLLMClient) -> None:
        assert isinstance(client, LLMClient)

    def test_validate_client_setup_success(
        self,
        client: DefaultLiteLLMClient,
    ) -> None:
        client.validate_client_setup()

    @pytest.mark.parametrize(
        "config",
        [
            {
                "provider": "cohere",
                "model": "test-cohere",
                # Stream is forbidden
                "stream": True,
            },
            {
                "provider": "cohere",
                "model": "test-cohere",
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
            DefaultLiteLLMClient.from_config(config)

        found_validation_log = False
        for record in caplog:
            if record["event"] == "validate_forbidden_keys":
                found_validation_log = True
                break

        assert found_validation_log

    @pytest.mark.parametrize(
        "config, expected_model, expected_litellm_model_name, mock_env_vars",
        [
            (
                {"provider": "cohere", "model": "test-cohere"},
                "test-cohere",
                "cohere/test-cohere",
                ["COHERE_API_KEY"],
            ),
            (
                {"provider": "cohere", "model": "cohere/test-cohere"},
                "cohere/test-cohere",
                "cohere/test-cohere",
                ["COHERE_API_KEY"],
            ),
            (
                {"provider": "sagemaker_chat", "model": "sagemaker_chat/endpoint-xyz"},
                "sagemaker_chat/endpoint-xyz",
                "sagemaker_chat/endpoint-xyz",
                [],
            ),
            (
                {
                    "provider": "huggingface",
                    "model": "rasa/cmd_gen_codellama_13b_calm_demo",
                },
                "rasa/cmd_gen_codellama_13b_calm_demo",
                "huggingface/rasa/cmd_gen_codellama_13b_calm_demo",
                ["HUGGINGFACE_API_KEY"],
            ),
            (
                {
                    "provider": "together_ai",
                    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                },
                "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                ["TOGETHERAI_API_KEY"],
            ),
        ],
    )
    def test_that_litellm_model_name_is_correctly_initialized(
        self,
        config: dict,
        expected_model: str,
        expected_litellm_model_name: str,
        mock_env_vars: List[str],
        monkeypatch: MonkeyPatch,
    ):
        # Given
        for var in mock_env_vars:
            monkeypatch.setenv(
                var,
                "mock key in test_that_litellm_model_name_is_correctly_initialized",
            )
        # When
        client = DefaultLiteLLMClient.from_config(config)
        # Then
        assert client.model == expected_model
        assert client._litellm_model_name == expected_litellm_model_name
        assert client.provider == config["provider"]

    @pytest.mark.parametrize(
        "config, expected_model, api_key_env_var, api_base_env_var, expected_failure",
        [
            (
                {
                    "provider": "ollama",
                    "model": "test_model",
                    "temperature": 0.2,
                    "api_key": "ollama llm embedding validation key",
                    "api_base": "https://ollama.com",
                },
                "test_model",
                None,
                None,
                False,
            ),
            # api key and api_base are set in environment variable
            (
                {
                    "provider": "ollama",
                    "model": "test_model",
                    "temperature": 0.2,
                },
                "test_model",
                "OLLAMA_API_KEY",
                "OLLAMA_API_BASE",
                False,
            ),
            # api base is not set
            (
                {
                    "provider": "ollama",
                    "model": "test_model",
                    "temperature": 0.2,
                    "api_key": "ollama llm embedding validation key",
                },
                "test_model",
                None,
                None,
                True,
            ),
            # api key is not set
            (
                {
                    "provider": "cohere",
                    "model": "test_model",
                    "temperature": 0.2,
                },
                "test_model",
                None,
                None,
                True,
            ),
        ],
    )
    def test_client_validation(
        self,
        config: dict,
        expected_model: str,
        api_key_env_var: str,
        api_base_env_var: str,
        expected_failure: bool,
        monkeypatch: MonkeyPatch,
    ) -> None:
        # Given
        if api_key_env_var:
            monkeypatch.setenv(api_key_env_var, "default llm client validation")
        if api_base_env_var:
            monkeypatch.setenv(api_base_env_var, "https://some.com")

        # When
        if expected_failure:
            with pytest.raises(ProviderClientValidationError):
                DefaultLiteLLMClient.from_config(config)
        else:
            client = DefaultLiteLLMClient.from_config(config)

            # Then
            assert client.model == expected_model

        # Cleanup
        if api_key_env_var:
            monkeypatch.delenv(api_key_env_var, raising=False)
        if api_base_env_var:
            monkeypatch.delenv(api_base_env_var, raising=False)
