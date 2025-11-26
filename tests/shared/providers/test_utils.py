from unittest.mock import Mock

import pytest
from google.auth.environment_vars import AWS_DEFAULT_REGION
from pytest import MonkeyPatch

from rasa.shared.constants import (
    AWS_ACCESS_KEY_ID_CONFIG_KEY,
    AWS_ACCESS_KEY_ID_ENV_VAR,
    AWS_REGION_NAME_CONFIG_KEY,
    AWS_REGION_NAME_ENV_VAR,
    AWS_SECRET_ACCESS_KEY_CONFIG_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_VAR,
    AWS_SESSION_TOKEN_CONFIG_KEY,
    AWS_SESSION_TOKEN_ENV_VAR,
    LLM_API_HEALTH_CHECK_ENV_VAR,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers._utils import validate_aws_setup_for_litellm_clients


@pytest.mark.parametrize(
    "available_call_kwargs, available_env_vars, should_raise_error",
    [
        # All settings are provided through config keys
        (
            {
                AWS_ACCESS_KEY_ID_CONFIG_KEY: "key_id",
                AWS_SECRET_ACCESS_KEY_CONFIG_KEY: "secret_key",
                AWS_REGION_NAME_CONFIG_KEY: "us-east-1",
                AWS_SESSION_TOKEN_CONFIG_KEY: "token",
            },
            {},
            False,
        ),
        # All settings are provided through config keys and env vars
        (
            {
                AWS_REGION_NAME_CONFIG_KEY: "us-east-1",
                AWS_SESSION_TOKEN_CONFIG_KEY: "token",
            },
            {
                AWS_ACCESS_KEY_ID_ENV_VAR: "key_id",
                AWS_SECRET_ACCESS_KEY_ENV_VAR: "secret_key",
            },
            False,
        ),
        # All settings are provided through the IAM role
        (
            {
                "model_id": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # noqa: E501
                AWS_REGION_NAME_CONFIG_KEY: "us-east-1",
            },
            {},
            False,
        ),
        # Missing one setting (access key id)
        (
            {
                AWS_REGION_NAME_CONFIG_KEY: "us-east-1",
                AWS_SESSION_TOKEN_CONFIG_KEY: "token",
            },
            {
                AWS_SECRET_ACCESS_KEY_ENV_VAR: "secret_key",
            },
            True,
        ),
        # Missing multiple settings (access key id and aws secret access key)
        (
            {
                AWS_REGION_NAME_CONFIG_KEY: "us-east-1",
                AWS_SESSION_TOKEN_CONFIG_KEY: "token",
            },
            {},
            True,
        ),
        # Missing region name
        (
            {
                "model_id": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"  # noqa: E501
            },
            {},
            True,
        ),
    ],
)
def test_validate_aws_setup_for_litellm_clients(
    available_call_kwargs: dict,
    available_env_vars: dict,
    should_raise_error: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    # Explicitly clear all relevant AWS-related env vars before setting new ones
    for env_var in [
        AWS_ACCESS_KEY_ID_ENV_VAR,
        AWS_SECRET_ACCESS_KEY_ENV_VAR,
        AWS_SESSION_TOKEN_ENV_VAR,
        AWS_REGION_NAME_ENV_VAR,
        AWS_DEFAULT_REGION,
        LLM_API_HEALTH_CHECK_ENV_VAR,
    ]:
        monkeypatch.delenv(env_var, raising=False)

    litellm_model_name = "bedrock/anthropic.claude-test"
    for env_var, value in available_env_vars.items():
        monkeypatch.setenv(env_var, value)

    if should_raise_error:
        with pytest.raises(ProviderClientValidationError):
            validate_aws_setup_for_litellm_clients(
                litellm_model_name, available_call_kwargs, "test", "bedrock"
            )
    else:
        validate_aws_setup_for_litellm_clients(
            litellm_model_name, available_call_kwargs, "test", "bedrock"
        )


def test_validate_credentials_with_aws_client_healthcheck_enabled(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(LLM_API_HEALTH_CHECK_ENV_VAR, "true")
    mock_validate_credentials_with_aws_client = Mock()
    monkeypatch.setattr(
        "rasa.shared.providers._utils._validate_credentials_with_aws_client",
        mock_validate_credentials_with_aws_client,
    )

    call_kwargs = {
        "model_id": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # noqa: E501
        AWS_REGION_NAME_CONFIG_KEY: "us-east-1",
    }
    litellm_model_name = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    validate_aws_setup_for_litellm_clients(
        litellm_model_name=litellm_model_name,
        litellm_call_kwargs=call_kwargs,
        source_log="test_function",
        provider="bedrock",
    )

    additional_kwargs = {"region_name": "us-east-1"}
    mock_validate_credentials_with_aws_client.assert_called_once_with(
        "bedrock", additional_kwargs, litellm_model_name, "test_function"
    )


def test_validate_credentials_with_aws_client_healthcheck_disabled(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(LLM_API_HEALTH_CHECK_ENV_VAR, "false")
    mock_validate_credentials_with_aws_client = Mock()
    monkeypatch.setattr(
        "rasa.shared.providers._utils._validate_credentials_with_aws_client",
        mock_validate_credentials_with_aws_client,
    )

    call_kwargs = {
        "model_id": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # noqa: E501
        AWS_REGION_NAME_CONFIG_KEY: "us-east-1",
    }
    litellm_model_name = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    validate_aws_setup_for_litellm_clients(
        litellm_model_name=litellm_model_name,
        litellm_call_kwargs=call_kwargs,
        source_log="test_function",
        provider="bedrock",
    )
    mock_validate_credentials_with_aws_client.assert_not_called()
