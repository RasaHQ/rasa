import pytest

from rasa.shared.constants import (
    AWS_ACCESS_KEY_ID_ENV_VAR,
    AWS_SECRET_ACCESS_KEY_CONFIG_KEY,
    AWS_ACCESS_KEY_ID_CONFIG_KEY,
    AWS_REGION_NAME_CONFIG_KEY,
    AWS_SESSION_TOKEN_CONFIG_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_VAR,
)
from rasa.shared.exceptions import ProviderClientValidationError
from rasa.shared.providers._configs.utils import resolve_aliases
from rasa.shared.providers._utils import validate_aws_setup_for_litellm_clients


def test_resolve_aliases_with_single_alias() -> None:
    config = {"old_key": "value1", "another_key": "value2"}
    alias_mapping = {"old_key": "new_key"}

    result = resolve_aliases(config, alias_mapping)

    assert result == {"new_key": "value1", "another_key": "value2"}
    assert "old_key" not in result


def test_resolve_aliases_with_multiple_aliases() -> None:
    config = {"old_key1": "value1", "old_key2": "value2"}
    alias_mapping = {"old_key1": "new_key1", "old_key2": "new_key2"}

    result = resolve_aliases(config, alias_mapping)

    assert result == {"new_key1": "value1", "new_key2": "value2"}
    assert "old_key1" not in result
    assert "old_key2" not in result


def test_resolve_aliases_with_no_aliases() -> None:
    config = {"key1": "value1", "key2": "value2"}
    alias_mapping = {"non_existent_key": "new_key"}

    result = resolve_aliases(config, alias_mapping)

    assert result == config


def test_resolve_aliases_with_conflicting_keys() -> None:
    config = {"old_key": "value1", "new_key": "value2"}
    alias_mapping = {"old_key": "new_key"}

    result = resolve_aliases(config, alias_mapping)

    assert result == {"new_key": "value1"}
    assert result["new_key"] == "value1"
    assert "old_key" not in result


def test_resolve_aliases_with_empty_config() -> None:
    config = {}
    alias_mapping = {"old_key": "new_key"}

    result = resolve_aliases(config, alias_mapping)

    assert result == {}


@pytest.mark.parametrize(
    "available_call_kwargs, available_env_vars, should_raise_error",
    [
        # All settings are provided through config keys
        (
            {
                AWS_ACCESS_KEY_ID_CONFIG_KEY: "key_id",
                AWS_SECRET_ACCESS_KEY_CONFIG_KEY: "secret_key",
                AWS_REGION_NAME_CONFIG_KEY: "region",
                AWS_SESSION_TOKEN_CONFIG_KEY: "token",
            },
            {},
            False,
        ),
        # All settings are provided through config keys and env vars
        (
            {
                AWS_REGION_NAME_CONFIG_KEY: "region",
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
                AWS_REGION_NAME_CONFIG_KEY: "region",
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
                AWS_REGION_NAME_CONFIG_KEY: "region",
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
    litellm_model_name = "bedrock/anthropic.claude-test"
    for env_var, value in available_env_vars.items():
        monkeypatch.setenv(env_var, value)

    if should_raise_error:
        with pytest.raises(ProviderClientValidationError):
            validate_aws_setup_for_litellm_clients(
                litellm_model_name, available_call_kwargs, "test"
            )
    else:
        validate_aws_setup_for_litellm_clients(
            litellm_model_name, available_call_kwargs, "test"
        )
