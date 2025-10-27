from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import structlog

from rasa.shared.constants import (
    API_BASE_CONFIG_KEY,
    API_TYPE_CONFIG_KEY,
    API_VERSION_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY,
    MAX_COMPLETION_TOKENS_CONFIG_KEY,
    MAX_TOKENS_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_NAME_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
    OPENAI_API_BASE_CONFIG_KEY,
    OPENAI_API_TYPE,
    OPENAI_API_TYPE_CONFIG_KEY,
    OPENAI_API_VERSION_CONFIG_KEY,
    OPENAI_PROVIDER,
    PROVIDER_CONFIG_KEY,
    RASA_TYPE_CONFIG_KEY,
    REQUEST_TIMEOUT_CONFIG_KEY,
    STREAM_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
)
from rasa.shared.utils.configs import (
    raise_deprecation_warnings,
    resolve_aliases,
    validate_forbidden_keys,
    validate_required_keys,
)

structlogger = structlog.get_logger()


DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING = {
    # Model name aliases
    MODEL_NAME_CONFIG_KEY: MODEL_CONFIG_KEY,
    # Provider aliases
    RASA_TYPE_CONFIG_KEY: PROVIDER_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY: PROVIDER_CONFIG_KEY,
    # API type aliases
    OPENAI_API_TYPE_CONFIG_KEY: API_TYPE_CONFIG_KEY,
    # API base aliases
    OPENAI_API_BASE_CONFIG_KEY: API_BASE_CONFIG_KEY,
    # API version aliases
    OPENAI_API_VERSION_CONFIG_KEY: API_VERSION_CONFIG_KEY,
    # Timeout aliases
    REQUEST_TIMEOUT_CONFIG_KEY: TIMEOUT_CONFIG_KEY,
    # Max tokens aliases
    MAX_TOKENS_CONFIG_KEY: MAX_COMPLETION_TOKENS_CONFIG_KEY,
}

REQUIRED_KEYS = [MODEL_CONFIG_KEY]

FORBIDDEN_KEYS = [
    STREAM_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
]


@dataclass
class OpenAIClientConfig:
    """Parses configuration for OpenAI client.

    Resolves aliases and raises deprecation warnings.

    Raises:
        ValueError: Raised in cases of invalid configuration:
            - If any of the required configuration keys are missing.
            - If `api_type` has a value different from `openai`.
    """

    model: str
    api_base: Optional[str]
    api_version: Optional[str]

    # API Type is not actually used by LiteLLM backend, but we define
    # it here for backward compatibility.
    api_type: str = OPENAI_API_TYPE

    # Provider is not used by LiteLLM backend, but we define
    # it here since it's used as switch between different
    # clients
    provider: str = OPENAI_PROVIDER

    extra_parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # In case of OpenAI hosting, it doesn't make sense
        # for API type to be anything else that 'openai'
        if self.api_type != OPENAI_API_TYPE:
            message = f"API type must be set to '{OPENAI_API_TYPE}'."
            structlogger.error(
                "openai_client_config.validation_error",
                message=message,
                api_type=self.api_type,
            )
            raise ValueError(message)
        if self.provider != OPENAI_PROVIDER:
            message = f"Provider must be set to '{OPENAI_PROVIDER}'."
            structlogger.error(
                "openai_client_config.validation_error",
                message=message,
                provider=self.provider,
            )
            raise ValueError(message)
        if self.model is None:
            message = "Model cannot be set to None."
            structlogger.error(
                "openai_client_config.validation_error",
                message=message,
                model=self.model,
            )
            raise ValueError(message)

    @classmethod
    def from_dict(cls, config: dict) -> OpenAIClientConfig:
        """Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Config is missing required keys.

        Returns:
            AzureOpenAIClientConfig
        """
        # Check for deprecated keys
        raise_deprecation_warnings(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)
        # Resolve any potential aliases
        config = cls.resolve_config_aliases(config)
        # Validate that the required keys are present
        validate_required_keys(config, REQUIRED_KEYS)
        # Validate that the forbidden keys are not present
        validate_forbidden_keys(config, FORBIDDEN_KEYS)
        this = OpenAIClientConfig(
            # Required parameters
            model=config.pop(MODEL_CONFIG_KEY),
            # Pop the 'provider' key. Currently, it's *optional* because of
            # backward compatibility with older versions.
            provider=config.pop(PROVIDER_CONFIG_KEY, OPENAI_PROVIDER),
            # Optional parameters
            api_base=config.pop(API_BASE_CONFIG_KEY, None),
            api_version=config.pop(API_VERSION_CONFIG_KEY, None),
            api_type=config.pop(API_TYPE_CONFIG_KEY, OPENAI_API_TYPE),
            # The rest of parameters (e.g. model parameters) are considered
            # as extra parameters (this also includes timeout).
            extra_parameters=config,
        )
        return this

    def to_dict(self) -> dict:
        """Converts the config instance into a dictionary."""
        d = asdict(self)
        # Extra parameters should also be on the top level
        d.pop("extra_parameters", None)
        d.update(self.extra_parameters)
        return d

    @staticmethod
    def resolve_config_aliases(config: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_aliases(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)


def is_openai_config(config: dict) -> bool:
    """Check whether the configuration is meant to configure an OpenAI client."""
    # Process the config to handle all the aliases
    config = OpenAIClientConfig.resolve_config_aliases(config)

    # Case: Configuration contains `provider: openai`
    if config.get(PROVIDER_CONFIG_KEY) == OPENAI_PROVIDER:
        return True

    return False
