from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Text, Union

import structlog

from rasa.shared.utils.io import resolve_environment_variables
from rasa.shared.utils.yaml import read_config_file
from rasa.tracing.constants import (
    ENDPOINTS_TRACING_KEY,
    LANGFUSE_CONFIG_BASE_URL_KEY,
    LANGFUSE_CONFIG_DEBUG_KEY,
    LANGFUSE_CONFIG_ENVIRONMENT_KEY,
    LANGFUSE_CONFIG_MEDIA_UPLOAD_THREAD_COUNT_KEY,
    LANGFUSE_CONFIG_PRIVATE_KEY_KEY,
    LANGFUSE_CONFIG_PUBLIC_KEY_KEY,
    LANGFUSE_CONFIG_RELEASE_KEY,
    LANGFUSE_CONFIG_SAMPLE_RATE_KEY,
    LANGFUSE_CONFIG_TIMEOUT_KEY,
    TRACING_TYPE_LANGFUSE,
)
from rasa.tracing.exceptions import (
    DuplicateTracingConfigException,
    InvalidLangfuseConfigException,
)
from rasa.utils.endpoints import EndpointConfig

structlogger = structlog.get_logger()


def configure_langfuse(endpoints_file: str) -> None:
    """Configure Langfuse callback for litellm from endpoints file."""
    langfuse_config = _get_langfuse_config(endpoints_file)

    if not langfuse_config:
        structlogger.debug(
            "langfuse_configuration.langfuse_config_not_found",
            event_info=(
                f"No Langfuse configuration found in {endpoints_file}, "
                f"Langfuse will not be configured."
            ),
        )
        return

    if not _is_langfuse_available():
        return

    config_values = _extract_langfuse_config_values(langfuse_config)

    _validate_langfuse_config(config_values)

    public_key = config_values[LANGFUSE_CONFIG_PUBLIC_KEY_KEY]
    private_key = config_values[LANGFUSE_CONFIG_PRIVATE_KEY_KEY]

    resolved_keys = _resolve_environment_variables(public_key, private_key)
    _set_langfuse_environment_variables(config_values, resolved_keys)
    _configure_litellm_callback()

    structlogger.info(
        "langfuse_configuration.langfuse_config_configured",
        event_info=(f"Langfuse has been configured from {endpoints_file}."),
    )


def _is_langfuse_available() -> bool:
    """Check if Langfuse is available."""
    try:
        import langfuse  # noqa: F401
    except ImportError as e:
        structlogger.error(
            "langfuse_configuration.langfuse_not_available",
            event_info=(
                "Langfuse is not available. "
                "Please install langfuse via `pip install rasa-pro[monitoring]`. "
                "Langfuse will not be configured."
            ),
            error=str(e),
        )
        return False
    return True


def _get_langfuse_config(endpoints_file: str) -> Optional[EndpointConfig]:
    """Read Langfuse configuration from endpoints file."""
    if not endpoints_file:
        return None

    try:
        content = read_config_file(endpoints_file)
    except FileNotFoundError:
        structlogger.debug(
            "langfuse_configuration.endpoints_file_not_found",
            event_info=(
                f"Endpoints file not found: {endpoints_file}, "
                f"Langfuse will not be configured."
            ),
        )
        return None
    except Exception:
        structlogger.error(
            "langfuse_configuration.endpoints_file_read_error",
            event_info=(
                f"Error reading endpoints file: {endpoints_file}, "
                f"Langfuse will not be configured."
            ),
        )
        return None

    if content.get(ENDPOINTS_TRACING_KEY) is None:
        return None

    tracing_config: Union[List[Dict[Text, Any]], Dict[Text, Any]] = content.get(
        ENDPOINTS_TRACING_KEY, []
    )

    tracing_configs = _parse_tracing_configs(tracing_config)

    langfuse_configs = [
        config for config in tracing_configs if config.type == TRACING_TYPE_LANGFUSE
    ]

    if len(langfuse_configs) == 0:
        return None

    if len(langfuse_configs) > 1:
        _log_multiple_langfuse_configs_error(endpoints_file)
        raise DuplicateTracingConfigException(
            f"Multiple Langfuse configs found in {endpoints_file}, "
            "which is not supported. Only one Langfuse config is allowed."
        )

    return langfuse_configs[0]


def _parse_tracing_configs(
    tracing_config: Union[List[Dict[Text, Any]], Dict[Text, Any]],
) -> List[EndpointConfig]:
    """Parse tracing configuration into list of EndpointConfig objects."""
    if isinstance(tracing_config, list):
        return [EndpointConfig.from_dict(item) for item in tracing_config]
    else:
        return [EndpointConfig.from_dict(tracing_config)]


def _log_multiple_langfuse_configs_error(endpoints_file: str) -> None:
    """Log error when multiple Langfuse configs are found."""
    structlogger.error(
        "langfuse_configuration.multiple_langfuse_configs",
        event_info=(
            f"Multiple Langfuse configs found in {endpoints_file}, "
            f"which is not supported. Only one Langfuse config is allowed."
        ),
    )


def _extract_langfuse_config_values(
    langfuse_config: EndpointConfig,
) -> Dict[str, Any]:
    """Extract all configuration values from Langfuse config."""
    return {
        LANGFUSE_CONFIG_PUBLIC_KEY_KEY: langfuse_config.kwargs.get(
            LANGFUSE_CONFIG_PUBLIC_KEY_KEY
        ),
        LANGFUSE_CONFIG_PRIVATE_KEY_KEY: langfuse_config.kwargs.get(
            LANGFUSE_CONFIG_PRIVATE_KEY_KEY
        ),
        LANGFUSE_CONFIG_BASE_URL_KEY: langfuse_config.kwargs.get(
            LANGFUSE_CONFIG_BASE_URL_KEY
        ),
        LANGFUSE_CONFIG_TIMEOUT_KEY: langfuse_config.kwargs.get(
            LANGFUSE_CONFIG_TIMEOUT_KEY
        ),
        LANGFUSE_CONFIG_DEBUG_KEY: langfuse_config.kwargs.get(
            LANGFUSE_CONFIG_DEBUG_KEY
        ),
        LANGFUSE_CONFIG_ENVIRONMENT_KEY: langfuse_config.kwargs.get(
            LANGFUSE_CONFIG_ENVIRONMENT_KEY
        ),
        LANGFUSE_CONFIG_RELEASE_KEY: langfuse_config.kwargs.get(
            LANGFUSE_CONFIG_RELEASE_KEY
        ),
        LANGFUSE_CONFIG_MEDIA_UPLOAD_THREAD_COUNT_KEY: langfuse_config.kwargs.get(
            LANGFUSE_CONFIG_MEDIA_UPLOAD_THREAD_COUNT_KEY
        ),
        LANGFUSE_CONFIG_SAMPLE_RATE_KEY: langfuse_config.kwargs.get(
            LANGFUSE_CONFIG_SAMPLE_RATE_KEY
        ),
    }


def _resolve_environment_variables(
    public_key: str, private_key: str
) -> Dict[str, Union[str, List[Any], Dict[str, Any]]]:
    """Resolve environment variables in public and secret keys."""
    return {
        LANGFUSE_CONFIG_PUBLIC_KEY_KEY: resolve_environment_variables(public_key),
        LANGFUSE_CONFIG_PRIVATE_KEY_KEY: resolve_environment_variables(private_key),
    }


def _set_langfuse_environment_variables(
    config_values: Dict[str, Any], resolved_keys: Dict[str, Optional[str]]
) -> None:
    """Set Langfuse environment variables for LiteLLM integration."""
    if resolved_keys[LANGFUSE_CONFIG_PUBLIC_KEY_KEY] is not None:
        os.environ["LANGFUSE_PUBLIC_KEY"] = resolved_keys[
            LANGFUSE_CONFIG_PUBLIC_KEY_KEY
        ]  # type: ignore[assignment]
    if resolved_keys[LANGFUSE_CONFIG_PRIVATE_KEY_KEY] is not None:
        os.environ["LANGFUSE_SECRET_KEY"] = resolved_keys[
            LANGFUSE_CONFIG_PRIVATE_KEY_KEY
        ]  # type: ignore[assignment]
    if config_values[LANGFUSE_CONFIG_BASE_URL_KEY] is not None:
        os.environ["LANGFUSE_OTEL_HOST"] = config_values[LANGFUSE_CONFIG_BASE_URL_KEY]
    if config_values[LANGFUSE_CONFIG_TIMEOUT_KEY] is not None:
        os.environ["LANGFUSE_TIMEOUT"] = config_values[LANGFUSE_CONFIG_TIMEOUT_KEY]
    if config_values[LANGFUSE_CONFIG_DEBUG_KEY] is not None:
        os.environ["LANGFUSE_DEBUG"] = config_values[LANGFUSE_CONFIG_DEBUG_KEY]
    if config_values[LANGFUSE_CONFIG_ENVIRONMENT_KEY] is not None:
        os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = config_values[
            LANGFUSE_CONFIG_ENVIRONMENT_KEY
        ]
    if config_values[LANGFUSE_CONFIG_RELEASE_KEY] is not None:
        os.environ["LANGFUSE_RELEASE"] = config_values[LANGFUSE_CONFIG_RELEASE_KEY]
    if config_values[LANGFUSE_CONFIG_MEDIA_UPLOAD_THREAD_COUNT_KEY] is not None:
        os.environ["LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT"] = config_values[
            LANGFUSE_CONFIG_MEDIA_UPLOAD_THREAD_COUNT_KEY
        ]
    if config_values[LANGFUSE_CONFIG_SAMPLE_RATE_KEY] is not None:
        os.environ["LANGFUSE_SAMPLE_RATE"] = config_values[
            LANGFUSE_CONFIG_SAMPLE_RATE_KEY
        ]


def _configure_litellm_callback() -> None:
    """Configure LiteLLM to use Langfuse as a callback."""
    import litellm

    # LiteLLM will automatically trace LLM calls to langfuse if these environment
    # variables are set and the callback is configured
    litellm.success_callback = ["langfuse_otel"]


def _validate_langfuse_config(config_values: Dict[str, Any]) -> None:
    """Validate Langfuse configuration."""
    _validate_required_keys(
        config_values,
        [
            LANGFUSE_CONFIG_PUBLIC_KEY_KEY,
            LANGFUSE_CONFIG_PRIVATE_KEY_KEY,
            LANGFUSE_CONFIG_BASE_URL_KEY,
        ],
    )

    public_key = config_values[LANGFUSE_CONFIG_PUBLIC_KEY_KEY]
    private_key = config_values[LANGFUSE_CONFIG_PRIVATE_KEY_KEY]
    _validate_key_syntax(public_key, private_key)


def _validate_required_keys(
    config_values: Dict[str, Any], required_keys: List[str]
) -> None:
    """Validate that required keys are provided."""
    missing_keys: List[str] = []
    for key in required_keys:
        if not config_values[key]:
            missing_keys.append(key)

    if missing_keys:
        structlogger.error(
            "tracing.langfuse_config.invalid_config",
            event_info=(
                f"Langfuse config is invalid. The following keys are required and "
                f"currently not configured: {missing_keys}."
            ),
        )
        raise InvalidLangfuseConfigException(
            f"Langfuse config is invalid. The following keys are required and "
            f"currently not configured: {missing_keys}."
        )


def _validate_key_syntax(public_key: str, secret_key: str) -> None:
    """Validate that keys use the correct $syntax for environment variables."""
    if not public_key.startswith("$") or not secret_key.startswith("$"):
        structlogger.error(
            "langfuse_configuration.langfuse_config_invalid",
            event_info=(
                f"Langfuse config is invalid: {LANGFUSE_CONFIG_PUBLIC_KEY_KEY} and "
                f"{LANGFUSE_CONFIG_PRIVATE_KEY_KEY} need to be provided via the "
                "$syntax."
            ),
        )
        raise InvalidLangfuseConfigException(
            f"Langfuse config is invalid: {LANGFUSE_CONFIG_PUBLIC_KEY_KEY} and "
            f"{LANGFUSE_CONFIG_PRIVATE_KEY_KEY} need to be provided via the "
            "$syntax."
        )
