import pytest
import structlog
from pytest import MonkeyPatch

from rasa.shared.constants import OPENAI_API_BASE_ENV_VAR
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.providers.llm.self_hosted_llm_client import (
    SelfHostedLLMClient,
)


class TestSelfHostedLLMClient:
    @pytest.fixture
    def client(self) -> SelfHostedLLMClient:
        return SelfHostedLLMClient(
            model="test_model",
            api_type="openai",
            api_base="https://my.api.base.com/my_model",
            provider="self-hosted",
        )

    def test_conforms_to_protocol(self, client: SelfHostedLLMClient) -> None:
        assert isinstance(client, LLMClient)

    def test_completion_fn_args(self, client: SelfHostedLLMClient) -> None:
        assert client._completion_fn_args == {
            "model": "openai/test_model",
            "drop_params": False,
            "api_base": "https://my.api.base.com/my_model",
            "api_version": None,
        }

    def test_init_does_not_fetches_from_environment_variables(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # Given

        # Set the environment variables
        monkeypatch.setenv(OPENAI_API_BASE_ENV_VAR, "https://my.api.base.com/my_model")

        # When
        client = SelfHostedLLMClient(
            model="test_model", api_base=None, provider="self-hosted"
        )

        # Then, does not fetch from environment variable.
        assert client.api_base is None

    @pytest.mark.parametrize(
        "config, expected_model, expected_api_base, expected_extra_parameters",
        [
            (
                {
                    "provider": "self-hosted",
                    "model": "test_model",
                    "api_type": "openai",
                    "api_base": "https://my.api.base.com/my_model",
                    "temperature": 0.2,
                    "max_tokens": 1000,
                },
                "test_model",
                "https://my.api.base.com/my_model",
                {"temperature": 0.2, "max_tokens": 1000},
            ),
            # Use deprecated aliases for provider
            (
                {
                    "type": "self-hosted",
                    "model": "test_model",
                    "api_base": "https://my.api.base.com/my_model",
                },
                "test_model",
                "https://my.api.base.com/my_model",
                {},
            ),
            (
                {
                    "_type": "self-hosted",
                    "model": "test_model",
                    "api_base": "https://my.api.base.com/my_model",
                },
                "test_model",
                "https://my.api.base.com/my_model",
                {},
            ),
            # Use deprecated alias for model
            (
                {
                    "provider": "self-hosted",
                    "model_name": "test_model",
                    "api_base": "https://my.api.base.com/my_model",
                },
                "test_model",
                "https://my.api.base.com/my_model",
                {},
            ),
            # Deprecated alias to api base
            (
                {
                    "provider": "self-hosted",
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
                    "provider": "self-hosted",
                    "model": "test_model",
                    "openai_api_type": "openai",
                    "api_base": "https://my.api.base.com/my_model",
                },
                "test_model",
                "https://my.api.base.com/my_model",
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
    ) -> None:
        # When
        client = SelfHostedLLMClient.from_config(config)

        # Then
        assert client.provider == "self-hosted"
        assert client.api_type == "openai"
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
                "provider": "self-hosted",
                "api_base": "https://my.api.base.com/my_model",
            },
        ],
    )
    def test_from_config_fails_if_invalid_api_type_is_present(
        self,
        invalid_config: dict,
    ) -> None:
        # Expected to fail, due to invalid value for `api_type`
        with pytest.raises(ValueError):
            SelfHostedLLMClient.from_config(invalid_config)

    def test_client_for_required_params(self) -> None:
        # Given
        client = SelfHostedLLMClient(
            model="test_model",
            api_base="https://my.api.base.com/my_model",
            api_type="openai",
            provider="self-hosted",
        )

        # Then
        assert client.model == "test_model"
        assert client.provider == "self-hosted"
        assert client.api_base == "https://my.api.base.com/my_model"
        assert client.api_type == "openai"
        assert client.api_version is None
        assert client._use_chat_completions_endpoint is True

    def test_completion(self, client: SelfHostedLLMClient) -> None:
        # Given
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

    async def test_acompletion(self, client: SelfHostedLLMClient) -> None:
        # Given
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
                "provider": "self-hosted",
                "model": "test-model",
                "api_base": "https://my.api.base.com/my_model",
                # Stream is forbidden
                "stream": True,
            },
            {
                "provider": "openai",
                "model": "test-model",
                "api_base": "https://my.api.base.com/my_model",
                # n is forbidden
                "n": 10,
            },
        ],
    )
    def test_from_config_raises_error_for_using_forbidden_keys(
        self, config: dict
    ) -> None:
        with pytest.raises(ValueError), structlog.testing.capture_logs() as caplog:
            SelfHostedLLMClient.from_config(config)

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
                    "api_base": "https://my.api.base.com/my_model",
                    "timeout": 7,
                },
                False,
            ),
            (
                {
                    "provider": "openai",
                    "model": "test-gpt",
                    "api_base": "https://my.api.base.com/my_model",
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
    ) -> None:
        # When
        with pytest.warns(None) as record:
            client = SelfHostedLLMClient.from_config(config)

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

    def test_litellm_model_name(self, client: SelfHostedLLMClient) -> None:
        assert client._litellm_model_name == "openai/test_model"

    def test_from_config_uses_api_version(self) -> None:
        # Given
        config = {
            "provider": "self-hosted",
            "model": "test_model",
            "api_base": "https://my.api.base.com/my_model",
            "api_version": "v1",
        }

        # When
        client = SelfHostedLLMClient.from_config(config)

        # Then
        assert client.api_version == "v1"
        assert client.api_base == "https://my.api.base.com/my_model"
        assert client.model == "test_model"
        assert client.provider == "self-hosted"
        assert client.api_type == "openai"
        assert client._use_chat_completions_endpoint is True
        assert client._extra_parameters == {}

    @pytest.mark.parametrize(
        "config, expected_missing_key",
        [
            (
                {  # missing provider
                    "model": "test-model",
                    "api_base": "https://my.api.base.com/my_model",
                },
                "provider",
            ),
            (
                {  # missing model
                    "provider": "self-hosted",
                    "api_base": "https://my.api.base.com/my_model",
                },
                "model",
            ),
            (
                {  # missing api_base
                    "provider": "self-hosted",
                    "model": "test-model",
                },
                "api_base",
            ),
        ],
    )
    def test_from_config_raises_error_for_missing_required_keys(
        self,
        config: dict,
        expected_missing_key: str,
    ) -> None:
        with pytest.raises(ValueError), structlog.testing.capture_logs() as caplog:
            SelfHostedLLMClient.from_config(config)

        found_validation_log = False
        for record in caplog:
            if record["event"] == "validate_required_keys":
                assert record["missing_keys"][0] == expected_missing_key
                found_validation_log = True
                break

        assert found_validation_log

    async def test_atext_completion(self) -> None:
        # Given
        test_prompt = "Hello, this is a test prompt."
        test_response = "Hello, this is mocked response!"

        client = SelfHostedLLMClient(
            model="test_model",
            api_base="https://my.api.base.com/my_model",
            api_type="openai",
            provider="self-hosted",
            use_chat_completions_endpoint=False,  # Ensures atext_completion is used.
        )
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
