"""Utilities for agent authentication and secret validation."""

import re
from typing import Any, Dict

from rasa.exceptions import ValidationError
from rasa.shared.agents.auth.constants import (
    CONFIG_API_KEY_KEY,
    CONFIG_CLIENT_SECRET_KEY,
    CONFIG_OAUTH_KEY,
    CONFIG_TOKEN_KEY,
)
from rasa.shared.constants import SECRET_DATA_FORMAT_PATTERN

AUTH_SECRETS = [CONFIG_API_KEY_KEY, CONFIG_TOKEN_KEY, CONFIG_CLIENT_SECRET_KEY]


def _is_valid_secret_data_format(value: str) -> bool:
    """Check if a value is in the correct environment variable format.

    Args:
        value: The value to check

    Returns:
        True if the value is in format "${env_var}", False otherwise
    """
    if not isinstance(value, str):
        return False

    # Use the common regex pattern for environment variable validation
    return bool(re.match(SECRET_DATA_FORMAT_PATTERN, value))


def _validate_secret_value(value: Any, key: str, context: str) -> None:
    """Generic function to validate a single secret value.

    Args:
        value: The value to validate
        key: The key name for error messages
        context: Context for error messages
            (e.g., "agent 'my_agent'", "MCP server 'my_server'")
    """
    if isinstance(value, str):
        if not _is_valid_secret_data_format(value):
            raise ValidationError(
                code="validation.sensitive_key_string_value_must_be_set_as_env_var",
                event_info=(
                    f"You defined the '{key}' in {context} as a string. The '{key}' "
                    f"must be set as an environment variable. Please update your "
                    f"config."
                ),
                key=key,
            )
    else:
        raise ValidationError(
            code="validation.sensitive_key_must_be_set_as_env_var",
            event_info=(
                f"You should define the '{key}' in {context} using the environment "
                f"variable syntax - ${{ENV_VARIABLE_NAME}}. Please update your config."
            ),
            key=key,
        )


def validate_secrets_in_params(
    params: Dict[str, Any], context_name: str = "configuration"
) -> None:
    """Validate that secrets in params are in environment variable format.

    Args:
        params: The parameters dictionary to validate
        context_name: Name of the context for error messages
            (e.g., "agent", "MCP server")
    """
    for key, value in params.items():
        if key in AUTH_SECRETS:
            _validate_secret_value(value, key, context_name)
        elif key == CONFIG_OAUTH_KEY and isinstance(value, dict):
            # Handle oauth object specifically - we know it contains client_secret
            if CONFIG_CLIENT_SECRET_KEY in value:
                _validate_secret_value(
                    value[CONFIG_CLIENT_SECRET_KEY],
                    CONFIG_CLIENT_SECRET_KEY,
                    context_name,
                )
