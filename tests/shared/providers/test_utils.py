import pytest
from moto import mock_aws

from rasa.shared.constants import (
    AWS_ACCESS_KEY_ID_CONFIG_KEY,
    AWS_ACCESS_KEY_ID_ENV_VAR,
    AWS_REGION_NAME_CONFIG_KEY,
    AWS_SECRET_ACCESS_KEY_CONFIG_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_VAR,
    AWS_SESSION_TOKEN_CONFIG_KEY,
    AWS_SESSION_TOKEN_ENV_VAR,
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
        with mock_aws():
            validate_aws_setup_for_litellm_clients(
                litellm_model_name, available_call_kwargs, "test", "bedrock"
            )
