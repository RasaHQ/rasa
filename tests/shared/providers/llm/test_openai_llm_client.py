import pytest
import structlog
from pytest import MonkeyPatch
from rasa.shared.constants import OPENAI_API_BASE_ENV_VAR, OPENAI_API_KEY_ENV_VAR
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.providers.llm.openai_llm_client import (
    OpenAILLMClient,
)


class TestOpenAILLMClient:
    @pytest.fixture
    def client(self, monkeypatch: MonkeyPatch) -> OpenAILLMClient:
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        return OpenAILLMClient(model="test_model", api_type="openai")

    def test_conforms_to_protocol(self, client: LLMClient, monkeypatch: MonkeyPatch):
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        assert isinstance(client, LLMClient)

    def test_init_fetches_from_environment_variables(self, monkeypatch: MonkeyPatch):
        # Given

        # Set the environment variables
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        monkeypatch.setenv(OPENAI_API_BASE_ENV_VAR, "https://my.api.base.com/my_model")

        # When
        client = OpenAILLMClient(model="test_model", api_base=None)

        # Then
        assert client.api_base == "https://my.api.base.com/my_model"

    @pytest.mark.parametrize(
        "config, expected_model, expected_api_base, expected_extra_parameters",
        [
            (
                {
                    "model": "test_model",
                    "api_type": "openai",
                    "temperature": 0.2,
                    "max_tokens": 1000,
                },
                "test_model",
                None,
                {"temperature": 0.2, "max_tokens": 1000},
            ),
            # Use deprecated alias for model
            (
                {"model_name": "test_model", "api_type": "openai"},
                "test_model",
                None,
                {},
            ),
            # Use api base
            (
                {
                    "model": "test_model",
                    "api_type": "openai",
                    "api_base": "https://my.api.base.com/my_model",
                },
                "test_model",
                "https://my.api.base.com/my_model",
                {},
            ),
            # Deprecated alias to api base
            (
                {
                    "model": "test_model",
                    "api_type": "openai",
                    "openai_api_base": "https://my.api.base.com/my_model",
                },
                "test_model",
                "https://my.api.base.com/my_model",
                {},
            ),
            # Deprecated alias to api type
            (
                {
                    "model": "test_model",
                    "openai_api_type": "openai",
                },
                "test_model",
                None,
                {},
            ),
        ],
    )
    def test_from_config(
        self,
        config: dict,
        expected_model: str,
        expected_api_base: str,
        expected_extra_parameters: dict,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")

        # When
        client = OpenAILLMClient.from_config(config)

        # Then
        assert client.model == expected_model
        assert client.api_base == expected_api_base
        assert len(client._litellm_extra_parameters) == len(expected_extra_parameters)
        for parameter_key, parameter_value in expected_extra_parameters.items():
            assert parameter_key in client._litellm_extra_parameters
            assert client._litellm_extra_parameters[parameter_key] == parameter_value

    @pytest.mark.parametrize(
        "invalid_config",
        [
            {
                # Missing `api_type`
                "model": "test-gpt",
            },
            {
                # Bypassing with LiteLLM only approach
                "model": "openai/test-gpt",
            },
            {
                # Invalid value for `api_type`
                "model": "test-gpt",
                "api_type": "invalid_value",
            },
        ],
    )
    def test_from_config_fails_if_required_keys_are_not_present(
        self,
        invalid_config: dict,
    ):
        with pytest.raises(ValueError):
            OpenAILLMClient.from_config(invalid_config)

    def test_completion(
        self,
        client: OpenAILLMClient,
        monkeypatch: MonkeyPatch,
    ) -> None:
        # Given
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        test_prompt = "Hello, this is a test prompt."
        test_response = "Hello, this is mocked response!"
        # LiteLLM supports mocking response for testing purposes
        client._extra_parameters = {"mock_response": test_response}

        # When
        response = client.completion([test_prompt])

        # Then
        assert response.choices == [test_response]
        assert response.model == client.model
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    async def test_acompletion(
        self, client: OpenAILLMClient, monkeypatch: MonkeyPatch
    ) -> None:
        # Given
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")

        test_prompt = "Hello, this is a test prompt."
        test_response = "Hello, this is mocked response!"
        # LiteLLM supports mocking response for testing purposes
        client._extra_parameters = {"mock_response": test_response}

        # When
        response = await client.acompletion([test_prompt])

        # Then
        assert response.choices == [test_response]
        assert response.model == client.model
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    @pytest.mark.parametrize(
        "config",
        [
            {
                "api_type": "openai",
                "model": "test-embedding",
                # Stream is forbidden
                "stream": True,
            },
            {
                "api_type": "openai",
                "model": "test-embedding",
                # n is forbidden
                "n": 10,
            },
        ],
    )
    def test_openai_embedding_cannot_be_instantiated_with_forbidden_keys(
        self,
        config: dict,
        monkeypatch: MonkeyPatch,
    ):
        with pytest.raises(ValueError), structlog.testing.capture_logs() as caplog:
            OpenAILLMClient.from_config(config)

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
                    "model": "test-gpt",
                    "api_type": "openai",
                    "timeout": 7,
                },
                False,
            ),
            (
                {
                    "model": "test-gpt",
                    "api_type": "openai",
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
            client = OpenAILLMClient.from_config(config)

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
