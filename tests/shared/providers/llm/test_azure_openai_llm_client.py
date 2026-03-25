from unittest.mock import MagicMock, patch

import pytest
import structlog
from pytest import MonkeyPatch

from rasa.shared.constants import (
    API_BASE_CONFIG_KEY,
    API_VERSION_CONFIG_KEY,
    AZURE_API_BASE_ENV_VAR,
    AZURE_API_KEY_ENV_VAR,
    AZURE_API_TYPE_ENV_VAR,
    AZURE_API_VERSION_ENV_VAR,
    AZURE_OPENAI_PROVIDER,
    OPENAI_API_BASE_ENV_VAR,
    OPENAI_API_KEY_ENV_VAR,
    OPENAI_API_TYPE_ENV_VAR,
    OPENAI_API_VERSION_ENV_VAR,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers.llm._base_litellm_client import _BaseLiteLLMClient
from rasa.shared.providers.llm.azure_openai_llm_client import (
    AzureOpenAILLMClient,
)
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.llm import (
    REASONING_EFFORT_CONFIG_KEY,
    REASONING_EFFORT_HIGH,
    REASONING_EFFORT_MINIMAL,
    REASONING_EFFORT_NONE,
)
from tests.utilities import filter_logs


class TestAzureOpenAILLMClient:
    def test_conforms_to_protocol(self, monkeypatch: MonkeyPatch):
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")
        client = AzureOpenAILLMClient(
            deployment="test_deployment",
            api_base="https://my.api.base.com/my_model",
            api_version="2023-01-01",
            api_type="azure",
        )
        assert isinstance(client, LLMClient)

        monkeypatch.delenv(AZURE_API_KEY_ENV_VAR, False)

    def test_init_fetches_from_environment_variables(self, monkeypatch: MonkeyPatch):
        # Given

        # Set the environment variables
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")
        monkeypatch.setenv(AZURE_API_BASE_ENV_VAR, "https://my.api.base.com/my_model")
        monkeypatch.setenv(AZURE_API_VERSION_ENV_VAR, "2023-01-01")
        monkeypatch.setenv(AZURE_API_TYPE_ENV_VAR, "test api type")

        # When
        client = AzureOpenAILLMClient(deployment="test_deployment")

        # Then
        assert client.deployment == "test_deployment"
        assert client.model is None
        assert client.api_base == "https://my.api.base.com/my_model"
        assert client.api_version == "2023-01-01"
        assert client._api_key_env_var == "${AZURE_API_KEY}"
        assert client.api_type == "test api type"

        # Clean up
        monkeypatch.delenv(OPENAI_API_KEY_ENV_VAR, False)
        monkeypatch.delenv(OPENAI_API_BASE_ENV_VAR, False)
        monkeypatch.delenv(OPENAI_API_VERSION_ENV_VAR, False)
        monkeypatch.delenv(OPENAI_API_TYPE_ENV_VAR, False)

    def test_init_fetches_from_deprecated_environment_variables(
        self, monkeypatch: MonkeyPatch
    ):
        # Given

        # Set the environment variables
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        monkeypatch.setenv(OPENAI_API_BASE_ENV_VAR, "https://my.api.base.com/my_model")
        monkeypatch.setenv(OPENAI_API_VERSION_ENV_VAR, "2023-01-01")
        monkeypatch.setenv(OPENAI_API_TYPE_ENV_VAR, "test api type")

        # When
        client = AzureOpenAILLMClient(deployment="test_deployment")

        # Then
        assert client.deployment == "test_deployment"
        assert client.model is None
        assert client.api_base == "https://my.api.base.com/my_model"
        assert client.api_version == "2023-01-01"
        assert client._api_key_env_var == "${OPENAI_API_KEY}"
        assert client.api_type == "test api type"

        # Clean up
        monkeypatch.delenv(OPENAI_API_KEY_ENV_VAR, False)
        monkeypatch.delenv(OPENAI_API_BASE_ENV_VAR, False)
        monkeypatch.delenv(OPENAI_API_VERSION_ENV_VAR, False)
        monkeypatch.delenv(OPENAI_API_TYPE_ENV_VAR, False)

    def test_validate_client_setup(self, monkeypatch: MonkeyPatch):
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")
        monkeypatch.delenv(AZURE_API_BASE_ENV_VAR, False)
        monkeypatch.delenv(AZURE_API_VERSION_ENV_VAR, False)
        monkeypatch.delenv(OPENAI_API_BASE_ENV_VAR, False)
        monkeypatch.delenv(OPENAI_API_VERSION_ENV_VAR, False)

        # Given
        expected_event = "azure_openai_llm_client.not_configured"
        expected_log_level = "error"
        expected_log_message_parts = [
            "Set API Base",
            AZURE_API_BASE_ENV_VAR,
            API_BASE_CONFIG_KEY,
            "Set API Version",
            AZURE_API_VERSION_ENV_VAR,
            API_VERSION_CONFIG_KEY,
        ]

        with structlog.testing.capture_logs() as caplog:
            with pytest.raises(ProviderClientValidationError):
                AzureOpenAILLMClient(deployment="test_deployment")

            # Then
            logs = filter_logs(
                caplog, expected_event, expected_log_level, expected_log_message_parts
            )

            assert len(logs) == 1

        # Clean up
        monkeypatch.delenv(AZURE_API_KEY_ENV_VAR, False)

    @pytest.mark.parametrize(
        "config,"
        "expected_deployment,"
        "expected_api_base,"
        "expected_api_version,"
        "expected_extra_parameters",
        [
            (
                {
                    "provider": "azure",
                    "deployment": "test_deployment_name",
                    "api_type": "azure",
                    "api_base": "https://my.api.base.com/my_model",
                    "api_version": "2023-01-01",
                    "temperature": 0.2,
                    "max_completion_tokens": 1000,
                },
                "test_deployment_name",
                "https://my.api.base.com/my_model",
                "2023-01-01",
                {"temperature": 0.2, "max_completion_tokens": 1000},
            ),
            # Use deprecated aliases for keys
            (
                {
                    "_type": "azure",
                    "deployment_name": "test_deployment_name",
                    "openai_api_type": "azure",
                    "openai_api_base": "https://my.api.base.com/my_model",
                    "openai_api_version": "2023-01-01",
                    "max_tokens": 256,
                },
                "test_deployment_name",
                "https://my.api.base.com/my_model",
                "2023-01-01",
                {"max_completion_tokens": 256},
            ),
            (
                {
                    "provider": "azure",
                    "engine": "test_deployment_name",
                    "openai_api_type": "azure",
                    "openai_api_base": "https://my.api.base.com/my_model",
                    "openai_api_version": "2023-01-01",
                    "max_tokens": 256,
                },
                "test_deployment_name",
                "https://my.api.base.com/my_model",
                "2023-01-01",
                {"max_completion_tokens": 256},
            ),
            (
                {
                    # Missing `provider`
                    "deployment": "test_deployment_name",
                    "api_base": "https://my.api.base.com/my_model",
                    "api_version": "2023-01-01",
                },
                "test_deployment_name",
                "https://my.api.base.com/my_model",
                "2023-01-01",
                {},
            ),
        ],
    )
    def test_from_config(
        self,
        config: dict,
        expected_deployment: str,
        expected_api_base: str,
        expected_api_version: str,
        expected_extra_parameters: dict,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")

        # When
        client = AzureOpenAILLMClient.from_config(config)

        # Then
        assert client.deployment == expected_deployment
        assert client.api_base == expected_api_base
        assert client.api_version == expected_api_version
        assert len(client._extra_parameters) == len(expected_extra_parameters)
        for parameter_key, parameter_value in expected_extra_parameters.items():
            assert parameter_key in client._extra_parameters
            assert client._extra_parameters[parameter_key] == parameter_value

        # Clean up
        monkeypatch.delenv(AZURE_API_KEY_ENV_VAR, False)

    @pytest.mark.parametrize(
        "invalid_config",
        [
            {
                # Bypassing with LiteLLM only approach
                "model": "azure/test_deployment_name",
                "api_base": "https://my.api.base.com/my_model",
                "api_version": "2023-01-01",
            },
            {
                # Invalid value for `provider`
                "provider": "invalid",
                "deployment": "test_deployment_name",
                "api_base": "https://my.api.base.com/my_model",
                "api_version": "2023-01-01",
            },
        ],
    )
    def test_from_config_fails_if_required_keys_are_not_present(
        self, invalid_config: dict, monkeypatch: MonkeyPatch
    ):
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")
        with pytest.raises(ValueError):
            AzureOpenAILLMClient.from_config(invalid_config)

        # Clean up
        monkeypatch.delenv(AZURE_API_KEY_ENV_VAR, False)

    def test_completion(self, monkeypatch: MonkeyPatch):
        # Given
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")
        test_prompt = "Hello, this is a test prompt."
        test_response = "Hello, this is mocked response!"

        client = AzureOpenAILLMClient(
            deployment="test_deployment",
            api_base="https://my.api.base.com/my_model",
            api_version="2023-01-01",
            mock_response=test_response,
        )

        # When
        response = client.completion([test_prompt])

        # Then
        assert response.choices == [test_response]
        assert response.model == client.deployment
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

        # Clean up
        monkeypatch.delenv(AZURE_API_KEY_ENV_VAR, False)

    async def test_acompletion(self, monkeypatch: MonkeyPatch):
        # Given
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")
        test_prompt = "Hello, this is a test prompt."
        test_response = "Hello, this is mocked response!"

        client = AzureOpenAILLMClient(
            deployment="test_deployment",
            api_base="https://my.api.base.com/my_model",
            api_version="2023-01-01",
            mock_response=test_response,
        )

        # When
        response = await client.acompletion([test_prompt])

        # Then
        assert response.choices == [test_response]
        assert response.model == client.deployment
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

        # Clean up
        monkeypatch.delenv(AZURE_API_KEY_ENV_VAR, False)

    @pytest.mark.parametrize(
        "config",
        [
            {
                "provider": "azure",
                "deployment": "some_azure_deployment",
                "api_base": "https://test",
                "api_version": "v1",
                # Stream is forbidden
                "stream": True,
            },
            {
                "provider": "azure",
                "deployment": "some_azure_deployment",
                "api_base": "https://test",
                "api_version": "v1",
                # n is forbidden
                "n": 10,
            },
        ],
    )
    def test_azure_openai_embedding_cannot_be_instantiated_with_forbidden_keys(
        self,
        config: dict,
        monkeypatch: MonkeyPatch,
    ):
        with pytest.raises(ValueError), structlog.testing.capture_logs() as caplog:
            AzureOpenAILLMClient.from_config(config)

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
                    "provider": "azure",
                    "deployment": "some_azure_deployment",
                    "model": "test-gpt",
                    "api_base": "https://test",
                    "api_version": "2023-05-15",
                    "timeout": 7,
                },
                False,
            ),
            (
                {
                    "provider": "azure",
                    "deployment": "some_azure_deployment",
                    "model": "test-gpt",
                    "api_base": "https://test",
                    "api_version": "2023-05-15",
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
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "some_key")

        # When
        with pytest.warns(None) as record:
            client = AzureOpenAILLMClient.from_config(config)

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

        # Clean up
        monkeypatch.delenv(AZURE_API_KEY_ENV_VAR, False)

    def test_resolve_api_key_env_var_from_extra_parameters(self):
        client = AzureOpenAILLMClient(
            deployment="test_deployment",
            api_base="https://my.api.base.com/my_model",
            api_version="2023-01-01",
            api_type="azure",
            api_key="${API_KEY}",
        )

        assert client._resolve_api_key_env_var() == "${API_KEY}"

    def test_resolve_api_key_env_var_from_azure_env_var(self, monkeypatch: MonkeyPatch):
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "azure_api_key")
        client = AzureOpenAILLMClient(
            deployment="test_deployment",
            api_base="https://my.api.base.com/my_model",
            api_version="2023-01-01",
            api_type="azure",
        )
        client._extra_parameters = {}
        assert client._resolve_api_key_env_var() == "${AZURE_API_KEY}"

    def test_resolve_api_key_env_var_from_openai_env_var(
        self, monkeypatch: MonkeyPatch
    ):
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "openai_api_key")
        client = AzureOpenAILLMClient(
            deployment="test_deployment",
            api_base="https://my.api.base.com/my_model",
            api_version="2023-01-01",
            api_type="azure",
        )
        client._extra_parameters = {}
        with pytest.warns(FutureWarning):
            assert client._resolve_api_key_env_var() == "${OPENAI_API_KEY}"

    def test_resolve_api_key_env_var_not_set(
        self,
        monkeypatch: MonkeyPatch,
    ):
        monkeypatch.setenv(AZURE_API_KEY_ENV_VAR, "my key")

        client = AzureOpenAILLMClient(
            deployment="test_deployment",
            api_base="https://my.api.base.com/my_model",
            api_version="2023-01-01",
            api_type="azure",
        )

        monkeypatch.delenv(AZURE_API_KEY_ENV_VAR, False)
        monkeypatch.delenv(OPENAI_API_KEY_ENV_VAR, False)

        with pytest.raises(ProviderClientValidationError):
            client._resolve_api_key_env_var()


# ============================================================================
# Deployment-only model resolution tests
# ============================================================================


def _make_deployment_only_client() -> AzureOpenAILLMClient:
    """Return a deployment-only AzureOpenAILLMClient (no `model` set)."""
    return AzureOpenAILLMClient(
        deployment="my-deployment",
        api_base="https://my.api.base.com",
        api_version="2025-01-01",
        api_key="${AZURE_API_KEY}",
    )


def _make_client_with_model(model: str = "gpt-5.1-2025-11-13") -> AzureOpenAILLMClient:
    """Return an AzureOpenAILLMClient with an explicit model name."""
    return AzureOpenAILLMClient(
        deployment="my-deployment",
        model=model,
        api_base="https://my.api.base.com",
        api_version="2025-01-01",
        api_key="${AZURE_API_KEY}",
    )


class TestAzureDeploymentModelResolution:
    """Tests for lazy model resolution in deployment-only Azure configs."""

    def test_is_deployment_only_true_when_no_model(self) -> None:
        client = _make_deployment_only_client()
        assert client._is_deployment_only is True

    def test_is_deployment_only_false_when_model_set(self) -> None:
        client = _make_client_with_model()
        assert client._is_deployment_only is False

    def test_on_model_resolved_sets_reasoning_effort_for_gpt51(self) -> None:
        client = _make_deployment_only_client()
        with patch(
            "rasa.shared.utils.llm._get_litellm_reasoning_effort_capability",
            return_value=(None, None),
        ):
            client._on_model_resolved("gpt-5.1-2025-11-13")

        assert client._resolved_model == "gpt-5.1-2025-11-13"
        assert (
            client._extra_parameters[REASONING_EFFORT_CONFIG_KEY]
            == REASONING_EFFORT_NONE
        )
        # allowed_openai_params is NOT stored in _extra_parameters; it is
        # injected on every call via _completion_fn_args instead.
        assert "allowed_openai_params" not in client._extra_parameters

    def test_on_model_resolved_sets_reasoning_effort_for_gpt5_mini(self) -> None:
        client = _make_deployment_only_client()
        with patch(
            "rasa.shared.utils.llm._get_litellm_reasoning_effort_capability",
            return_value=(None, None),
        ):
            client._on_model_resolved("gpt-5-mini-2025-08-07")

        assert client._resolved_model == "gpt-5-mini-2025-08-07"
        assert (
            client._extra_parameters[REASONING_EFFORT_CONFIG_KEY]
            == REASONING_EFFORT_MINIMAL
        )

    def test_on_model_resolved_no_reasoning_effort_for_gpt4o(self) -> None:
        # gpt-4o is not in the fallback map -> no reasoning_effort injected
        client = _make_deployment_only_client()
        with patch(
            "rasa.shared.utils.llm._get_litellm_reasoning_effort_capability",
            return_value=(None, None),
        ):
            client._on_model_resolved("gpt-4o-2024-11-20")

        assert client._resolved_model == "gpt-4o-2024-11-20"
        assert REASONING_EFFORT_CONFIG_KEY not in client._extra_parameters

    def test_on_model_resolved_does_not_override_user_set_reasoning_effort(
        self,
    ) -> None:
        # User explicitly set reasoning_effort; _on_model_resolved must not
        # change it.  allowed_openai_params is handled by _completion_fn_args
        # on every call, so it is not stored in _extra_parameters here.
        client = _make_deployment_only_client()
        client._extra_parameters[REASONING_EFFORT_CONFIG_KEY] = REASONING_EFFORT_HIGH

        with patch(
            "rasa.shared.utils.llm._get_litellm_reasoning_effort_capability",
            return_value=(None, None),
        ):
            client._on_model_resolved("gpt-5.1-2025-11-13")

        assert (
            client._extra_parameters[REASONING_EFFORT_CONFIG_KEY]
            == REASONING_EFFORT_HIGH
        )
        assert "allowed_openai_params" not in client._extra_parameters

    def test_on_model_resolved_is_idempotent(self) -> None:
        # Calling twice with the same model must not change anything.
        client = _make_deployment_only_client()
        client._on_model_resolved("gpt-5.1-2025-11-13")
        effort_after_first = client._extra_parameters.get(REASONING_EFFORT_CONFIG_KEY)

        client._on_model_resolved("gpt-5.1-2025-11-13")

        assert client._resolved_model == "gpt-5.1-2025-11-13"
        assert (
            client._extra_parameters.get(REASONING_EFFORT_CONFIG_KEY)
            == effort_after_first
        )

    def test_on_model_resolved_second_call_is_ignored(self) -> None:
        # A second call with a different model name should be silently ignored.
        client = _make_deployment_only_client()
        client._on_model_resolved("gpt-5.1-2025-11-13")
        first_effort = client._extra_parameters.get(REASONING_EFFORT_CONFIG_KEY)

        client._on_model_resolved("gpt-5-mini-2025-08-07")

        # Should still have the value from the first call
        assert client._extra_parameters.get(REASONING_EFFORT_CONFIG_KEY) == first_effort

    def test_format_response_triggers_model_resolution(self) -> None:
        client = _make_deployment_only_client()
        assert client._resolved_model is None

        fake_response = LLMResponse(
            id="r1",
            created=0,
            choices=["hello"],
            model="gpt-5.1-2025-11-13",
        )
        with patch.object(
            _BaseLiteLLMClient,
            "_format_response",
            return_value=fake_response,
        ):
            result = client._format_response(MagicMock())

        assert result.model == "gpt-5.1-2025-11-13"
        assert client._resolved_model == "gpt-5.1-2025-11-13"
        assert (
            client._extra_parameters.get(REASONING_EFFORT_CONFIG_KEY)
            == REASONING_EFFORT_NONE
        )

    def test_format_response_skips_resolution_when_model_set(self) -> None:
        # When the client already has an explicit model, _on_model_resolved
        # must NOT be triggered (it's not a deployment-only config).
        client = _make_client_with_model("gpt-5.1-2025-11-13")

        fake_response = LLMResponse(
            id="r2",
            created=0,
            choices=["hello"],
            model="gpt-5.1-2025-11-13",
        )
        with patch.object(
            _BaseLiteLLMClient,
            "_format_response",
            return_value=fake_response,
        ):
            client._format_response(MagicMock())

        # _resolved_model should remain None because _is_deployment_only is False
        assert client._resolved_model is None

    def test_completion_fn_args_adds_allowed_openai_params_for_deployment_only(
        self,
    ) -> None:
        # When reasoning_effort is present for a deployment-only config,
        # _completion_fn_args must add it to allowed_openai_params so LiteLLM
        # forwards it without raising UnsupportedParamsError — including on the
        # very first call, before _on_model_resolved has fired.
        client = _make_deployment_only_client()
        client._extra_parameters[REASONING_EFFORT_CONFIG_KEY] = REASONING_EFFORT_HIGH

        fn_args = client._completion_fn_args

        assert REASONING_EFFORT_CONFIG_KEY in fn_args.get("allowed_openai_params", [])

    def test_completion_fn_args_merges_with_existing_allowed_params(self) -> None:
        # Pre-existing allowed_openai_params entries must be preserved.
        client = _make_deployment_only_client()
        client._extra_parameters[REASONING_EFFORT_CONFIG_KEY] = REASONING_EFFORT_NONE
        client._extra_parameters["allowed_openai_params"] = ["stream"]

        fn_args = client._completion_fn_args

        allowed = fn_args.get("allowed_openai_params", [])
        assert "stream" in allowed
        assert REASONING_EFFORT_CONFIG_KEY in allowed

    def test_completion_fn_args_no_allowed_params_when_no_reasoning_effort(
        self,
    ) -> None:
        # If reasoning_effort is absent, allowed_openai_params must not be
        # injected for it.
        client = _make_deployment_only_client()

        fn_args = client._completion_fn_args

        assert REASONING_EFFORT_CONFIG_KEY not in fn_args.get(
            "allowed_openai_params", []
        )

    def test_completion_fn_args_no_allowed_params_when_model_set(self) -> None:
        # Non-deployment-only configs must not get allowed_openai_params
        # injected for reasoning_effort.
        client = _make_client_with_model()
        client._extra_parameters[REASONING_EFFORT_CONFIG_KEY] = REASONING_EFFORT_NONE

        fn_args = client._completion_fn_args

        assert REASONING_EFFORT_CONFIG_KEY not in fn_args.get(
            "allowed_openai_params", []
        )

    def test_completion_fn_args_includes_allowed_params_after_auto_injection(
        self,
    ) -> None:
        # End-to-end probe path: _on_model_resolved injects reasoning_effort, then
        # _completion_fn_args must include allowed_openai_params on the next call.
        client = _make_deployment_only_client()
        client._on_model_resolved("gpt-5.1-2025-11-13")

        fn_args = client._completion_fn_args

        assert REASONING_EFFORT_CONFIG_KEY in fn_args
        assert REASONING_EFFORT_CONFIG_KEY in fn_args.get("allowed_openai_params", [])

    def test_format_response_stream_triggers_model_resolution(self) -> None:
        """Streaming path must also resolve the model name."""
        client = _make_deployment_only_client()
        assert client._resolved_model is None

        fake_chunk = LLMResponse(
            id="r-stream",
            created=0,
            choices=["hi"],
            model="gpt-5.1-2025-11-13",
        )
        with patch.object(
            _BaseLiteLLMClient,
            "_format_response_stream",
            return_value=fake_chunk,
        ):
            result = client._format_response_stream(MagicMock())

        assert result.model == "gpt-5.1-2025-11-13"
        assert client._resolved_model == "gpt-5.1-2025-11-13"
        assert (
            client._extra_parameters.get(REASONING_EFFORT_CONFIG_KEY)
            == REASONING_EFFORT_NONE
        )

    def test_format_response_stream_skips_resolution_when_model_set(self) -> None:
        """Streaming path must not resolve when an explicit model is set."""
        client = _make_client_with_model("gpt-5.1-2025-11-13")

        fake_chunk = LLMResponse(
            id="r-stream-2",
            created=0,
            choices=["hi"],
            model="gpt-5.1-2025-11-13",
        )
        with patch.object(
            _BaseLiteLLMClient,
            "_format_response_stream",
            return_value=fake_chunk,
        ):
            client._format_response_stream(MagicMock())

        assert client._resolved_model is None

    def test_format_response_skips_resolution_when_model_is_empty(self) -> None:
        # If the API response contains an empty model string, _on_model_resolved
        # must not be called (guard against partial/malformed responses).
        client = _make_deployment_only_client()

        fake_response = LLMResponse(
            id="r3",
            created=0,
            choices=["hello"],
            model="",  # empty
        )
        with patch.object(
            _BaseLiteLLMClient,
            "_format_response",
            return_value=fake_response,
        ):
            client._format_response(MagicMock())

        assert client._resolved_model is None
        assert REASONING_EFFORT_CONFIG_KEY not in client._extra_parameters

    def test_on_model_resolved_provider_used_is_azure(self) -> None:
        # The fallback map is queried with AZURE_OPENAI_PROVIDER.
        # Verify the correct provider constant is passed by checking that
        # gpt-5.1 (which is in the map for both openai and azure) resolves.
        from rasa.shared.utils.llm import _get_fallback_reasoning_effort_default

        effort = _get_fallback_reasoning_effort_default(
            AZURE_OPENAI_PROVIDER, "gpt-5.1-2025-11-13"
        )
        assert effort == REASONING_EFFORT_NONE
