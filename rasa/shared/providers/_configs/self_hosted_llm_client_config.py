from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import structlog

from rasa.shared.constants import (
    MODEL_CONFIG_KEY,
    MODEL_NAME_CONFIG_KEY,
    OPENAI_API_BASE_CONFIG_KEY,
    API_BASE_CONFIG_KEY,
    OPENAI_API_TYPE_CONFIG_KEY,
    API_TYPE_CONFIG_KEY,
    OPENAI_API_VERSION_CONFIG_KEY,
    API_VERSION_CONFIG_KEY,
    RASA_TYPE_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY,
    STREAM_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
    REQUEST_TIMEOUT_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    OPENAI_PROVIDER,
    SELF_HOSTED_PROVIDER,
    USE_CHAT_COMPLETIONS_ENDPOINT_CONFIG_KEY,
)
from rasa.shared.providers._configs.utils import (
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
}

REQUIRED_KEYS = [API_BASE_CONFIG_KEY, MODEL_CONFIG_KEY, PROVIDER_CONFIG_KEY]

FORBIDDEN_KEYS = [
    STREAM_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
]


@dataclass
class SelfHostedLLMClientConfig:
    """Parses configuration for Self Hosted LiteLLM client, resolves aliases and
    raises deprecation warnings.

    Raises:
        ValueError: Raised in cases of invalid configuration:
            - If any of the required configuration keys are missing.
    """

    model: str
    provider: str
    api_base: str
    api_version: Optional[str] = None
    api_type: Optional[str] = OPENAI_PROVIDER
    use_chat_completions_endpoint: Optional[bool] = True
    extra_parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model is None:
            message = "Model cannot be set to None."
            structlogger.error(
                "self_hosted_llm_client_config.validation_error",
                message=message,
                model=self.model,
            )
            raise ValueError(message)
        if self.provider is None:
            message = "Provider cannot be set to None."
            structlogger.error(
                "self_hosted_llm_client_config.validation_error",
                message=message,
                provider=self.provider,
            )
            raise ValueError(message)
        if self.api_base is None:
            message = "API base cannot be set to None."
            structlogger.error(
                "self_hosted_llm_client_config.validation_error",
                message=message,
                provider=self.provider,
            )
            raise ValueError(message)
        if self.api_type != OPENAI_PROVIDER:
            message = (
                f"Currently supports only {OPENAI_PROVIDER} endpoints. "
                f"API type must be set to '{OPENAI_PROVIDER}'."
            )
            structlogger.error(
                "self_hosted_llm_client_config.validation_error",
                message=message,
                api_type=self.api_type,
            )
            raise ValueError(message)

    @classmethod
    def from_dict(cls, config: dict) -> "SelfHostedLLMClientConfig":
        """
        Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Config is missing required keys.

        Returns:
            DefaultLiteLLMClientConfig
        """
        # Check for deprecated keys
        raise_deprecation_warnings(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)
        # Resolve any potential aliases
        config = cls.resolve_config_aliases(config)
        # Validate that the required keys are present
        validate_required_keys(config, REQUIRED_KEYS)
        # Validate that the forbidden keys are not present
        validate_forbidden_keys(config, FORBIDDEN_KEYS)
        this = SelfHostedLLMClientConfig(
            # Required parameters
            model=config.pop(MODEL_CONFIG_KEY),
            provider=config.pop(PROVIDER_CONFIG_KEY),
            api_base=config.pop(API_BASE_CONFIG_KEY),
            # Optional parameters
            api_type=config.pop(API_TYPE_CONFIG_KEY, OPENAI_PROVIDER),
            api_version=config.pop(API_VERSION_CONFIG_KEY, None),
            use_chat_completions_endpoint=config.pop(
                USE_CHAT_COMPLETIONS_ENDPOINT_CONFIG_KEY, True
            ),
            # The rest of parameters (e.g. model parameters) are considered
            # as extra parameters
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


def is_self_hosted_config(config: dict) -> bool:
    """Check whether the configuration is meant to configure an self-hosted client."""
    # Process the config to handle all the aliases
    config = SelfHostedLLMClientConfig.resolve_config_aliases(config)

    # Case: Configuration contains `provider: self-hosted`
    if config.get(PROVIDER_CONFIG_KEY) == SELF_HOSTED_PROVIDER:
        return True

    return False
