from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from pytest import MonkeyPatch

from rasa.shared.constants import OPENAI_API_BASE_ENV_VAR, OPENAI_API_KEY_ENV_VAR
from rasa.shared.exceptions import ProviderClientValidationError
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
                    "provider": "openai",
                    "model": "test_model",
                    "api_type": "openai",
                    "temperature": 0.2,
                    "max_completion_tokens": 1000,
                },
                "test_model",
                None,
                {"temperature": 0.2, "max_completion_tokens": 1000},
            ),
            # Use deprecated aliases for provider
            (
                {
                    "type": "openai",
                    "model": "test_model",
                },
                "test_model",
                None,
                {},
            ),
            (
                {
                    "_type": "openai",
                    "model": "test_model",
                },
                "test_model",
                None,
                {},
            ),
            # Use deprecated alias for model
            (
                {
                    "provider": "openai",
                    "model_name": "test_model",
                },
                "test_model",
                None,
                {},
            ),
            # Use api base
            (
                {
                    "provider": "openai",
                    "model": "test_model",
                    "api_base": "https://my.api.base.com/my_model",
                },
                "test_model",
                "https://my.api.base.com/my_model",
                {},
            ),
            # Deprecated alias to api base
            (
                {
                    "provider": "openai",
                    "model": "test_model",
                    "openai_api_base": "https://my.api.base.com/my_model",
                },
                "test_model",
                "https://my.api.base.com/my_model",
                {},
            ),
            # Deprecated alias to api type
            (
                {
                    "provider": "openai",
                    "model": "test_model",
                    "openai_api_type": "openai",
                },
                "test_model",
                None,
                {},
            ),
            # Deprecated alias for max tokens
            (
                {
                    "provider": "openai",
                    "model": "test_model",
                    "api_type": "openai",
                    "temperature": 0.2,
                    "max_tokens": 1000,
                },
                "test_model",
                None,
                {"temperature": 0.2, "max_completion_tokens": 1000},
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

    def test_completion_api_key_set_as_env(
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

    @pytest.fixture
    def mocked_lite_llm_completion(self, monkeypatch: MonkeyPatch) -> MagicMock:
        _mock = MagicMock()
        monkeypatch.setattr(
            "rasa.shared.providers.llm._base_litellm_client.completion", _mock
        )
        return _mock

    def test_completion_call_arguments(
        self,
        client: OpenAILLMClient,
        monkeypatch: MonkeyPatch,
        mocked_lite_llm_completion: MagicMock,
    ) -> None:
        """Tests that the API base, version and key is resolved properly."""
        # Given
        api_key = "my key"
        monkeypatch.setenv("RASA_OPENAI_API_KEY", api_key)
        test_prompt = "Hello, this is a test prompt."

        client._extra_parameters = {"api_key": "${RASA_OPENAI_API_KEY}"}
        client._api_base = "https://my.api.base.com/my_model"
        client._api_version = "v1"

        # When
        client.completion([test_prompt])

        # Then
        mocked_lite_llm_completion.assert_called_once_with(
            messages=[{"content": "Hello, this is a test prompt.", "role": "user"}],
            **{
                "api_base": client._api_base,
                "api_version": client._api_version,
                "api_key": api_key,
                "drop_params": False,
                "model": "openai/test_model",
            },
        )

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

    @pytest.fixture
    def mocked_lite_llm_acompletion(self, monkeypatch: MonkeyPatch) -> AsyncMock:
        _mock = AsyncMock()
        monkeypatch.setattr(
            "rasa.shared.providers.llm._base_litellm_client.acompletion", _mock
        )
        return _mock

    async def test_acompletion_call_arguments(
        self,
        client: OpenAILLMClient,
        monkeypatch: MonkeyPatch,
        mocked_lite_llm_acompletion: MagicMock,
    ) -> None:
        """Tests that the API base, version and key is resolved properly for acompletion."""  # noqa: E501
        # Given
        api_key = "my key"
        monkeypatch.setenv("RASA_OPENAI_API_KEY", api_key)
        test_prompt = "Hello, this is a test prompt."

        #  We are not testing for the response here, so we can mock the response
        mocked_lite_llm_acompletion.return_value = MagicMock()

        client._extra_parameters = {"api_key": "${RASA_OPENAI_API_KEY}"}
        client._api_base = "https://my.api.base.com/my_model"
        client._api_version = "v1"

        # When
        await client.acompletion([test_prompt])

        # Then
        mocked_lite_llm_acompletion.assert_called_once_with(
            messages=[{"content": "Hello, this is a test prompt.", "role": "user"}],
            **{
                "api_base": client._api_base,
                "api_version": client._api_version,
                "api_key": api_key,
                "drop_params": False,
                "model": "openai/test_model",
            },
        )

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
    def test_from_config_raises_error_for_using_forbidden_keys(
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
                    "provider": "openai",
                    "model": "test-gpt",
                    "timeout": 7,
                },
                False,
            ),
            (
                {
                    "provider": "openai",
                    "model": "test-gpt",
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

    @pytest.mark.parametrize(
        "config, expected_model, api_key_env_var, expected_failure",
        [
            (
                {
                    "provider": "openai",
                    "model": "test_model",
                    "temperature": 0.2,
                    "api_key": "openai llm client validation key",
                },
                "test_model",
                None,
                False,
            ),
            # api key is set in environment variable
            (
                {
                    "provider": "openai",
                    "model": "test_model",
                    "temperature": 0.2,
                },
                "test_model",
                OPENAI_API_KEY_ENV_VAR,
                False,
            ),
            # api key is not set
            (
                {
                    "provider": "openai",
                    "model": "test_model",
                    "temperature": 0.2,
                },
                "test_model",
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
        expected_failure: bool,
        monkeypatch: MonkeyPatch,
    ) -> None:
        # Given
        monkeypatch.delenv(OPENAI_API_KEY_ENV_VAR, raising=False)
        if api_key_env_var:
            monkeypatch.setenv(api_key_env_var, "openai llm client validation key")

        # When
        if expected_failure:
            with pytest.raises(ProviderClientValidationError):
                OpenAILLMClient.from_config(config)
        else:
            client = OpenAILLMClient.from_config(config)

            # Then
            assert client.model == expected_model

        # Cleanup
        if api_key_env_var:
            monkeypatch.delenv(api_key_env_var, raising=False)
