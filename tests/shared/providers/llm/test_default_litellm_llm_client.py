import pytest
import structlog
from pytest import MonkeyPatch

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
        "config, expected_model, expected_litellm_model_name",
        [
            (
                {"provider": "cohere", "model": "test-cohere"},
                "test-cohere",
                "cohere/test-cohere",
            ),
            (
                {"provider": "cohere", "model": "cohere/test-cohere"},
                "cohere/test-cohere",
                "cohere/test-cohere",
            ),
        ],
    )
    def test_that_litellm_model_name_is_correctly_initialized(
        self,
        config: dict,
        expected_model: str,
        expected_litellm_model_name: str,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        monkeypatch.setenv(
            "COHERE_API_KEY",
            "mock key in test_that_litellm_model_name_is_correctly_initialized",
        )
        # When
        client = DefaultLiteLLMClient.from_config(config)
        # Then
        assert client.model == expected_model
        assert client._litellm_model_name == expected_litellm_model_name
        assert client.provider == "cohere"
