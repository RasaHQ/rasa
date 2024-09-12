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
    DEPLOYMENT_CONFIG_KEY,
    DEPLOYMENT_NAME_CONFIG_KEY,
    ENGINE_CONFIG_KEY,
    RASA_TYPE_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY,
    STREAM_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
    REQUEST_TIMEOUT_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    AZURE_OPENAI_PROVIDER,
    AZURE_API_TYPE,
)
from rasa.shared.providers._configs.utils import (
    resolve_aliases,
    raise_deprecation_warnings,
    validate_required_keys,
    validate_forbidden_keys,
)

structlogger = structlog.get_logger()

DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING = {
    # Deployment name aliases
    DEPLOYMENT_NAME_CONFIG_KEY: DEPLOYMENT_CONFIG_KEY,
    ENGINE_CONFIG_KEY: DEPLOYMENT_CONFIG_KEY,
    # Provider aliases
    RASA_TYPE_CONFIG_KEY: PROVIDER_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY: PROVIDER_CONFIG_KEY,
    # API type aliases
    OPENAI_API_TYPE_CONFIG_KEY: API_TYPE_CONFIG_KEY,
    # API base aliases
    OPENAI_API_BASE_CONFIG_KEY: API_BASE_CONFIG_KEY,
    # API version aliases
    OPENAI_API_VERSION_CONFIG_KEY: API_VERSION_CONFIG_KEY,
    # Model name aliases
    MODEL_NAME_CONFIG_KEY: MODEL_CONFIG_KEY,
    # Timeout aliases
    REQUEST_TIMEOUT_CONFIG_KEY: TIMEOUT_CONFIG_KEY,
}

REQUIRED_KEYS = [DEPLOYMENT_CONFIG_KEY]

FORBIDDEN_KEYS = [
    STREAM_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
]


@dataclass
class AzureOpenAIClientConfig:
    """Parses configuration for Azure OpenAI client, resolves aliases and
    raises deprecation warnings.

    Raises:
        ValueError: Raised in cases of invalid configuration:
            - If any of the required configuration keys are missing.
            - If `api_type` has a value different from `azure`.
    """

    deployment: str

    model: Optional[str]
    api_base: Optional[str]
    api_version: Optional[str]
    # API Type is not used by LiteLLM backend, but we define
    # it here for backward compatibility.
    api_type: Optional[str] = AZURE_API_TYPE

    # Provider is not used by LiteLLM backend, but we define it here since it's
    # used as switch between different clients.
    provider: str = AZURE_OPENAI_PROVIDER

    extra_parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider != AZURE_OPENAI_PROVIDER:
            message = f"Provider must be set to '{AZURE_OPENAI_PROVIDER}'."
            structlogger.error(
                "azure_openai_client_config.validation_error",
                message=message,
                provider=self.provider,
            )
            raise ValueError(message)
        if self.deployment is None:
            message = "Deployment cannot be set to None."
            structlogger.error(
                "azure_openai_client_config.validation_error",
                message=message,
                deployment=self.deployment,
            )
            raise ValueError(message)

    @classmethod
    def from_dict(cls, config: dict) -> "AzureOpenAIClientConfig":
        """
        Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Raised in cases of invalid configuration:
                - If any of the required configuration keys are missing.
                - If `api_type` has a value different from `azure`.

        Returns:
            AzureOpenAIClientConfig
        """
        # Check for deprecated keys
        raise_deprecation_warnings(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)
        # Resolve any potential aliases
        config = cls.resolve_config_aliases(config)
        # Validate that required keys are set
        validate_required_keys(config, REQUIRED_KEYS)
        # Validate that the forbidden keys are not present
        validate_forbidden_keys(config, FORBIDDEN_KEYS)
        # Init client config
        this = AzureOpenAIClientConfig(
            # Required parameters
            deployment=config.pop(DEPLOYMENT_CONFIG_KEY),
            # Pop the 'provider' key. Currently, it's *optional* because of
            # backward compatibility with older versions.
            provider=config.pop(PROVIDER_CONFIG_KEY, AZURE_OPENAI_PROVIDER),
            # Optional
            api_type=config.pop(API_TYPE_CONFIG_KEY, AZURE_API_TYPE),
            model=config.pop(MODEL_CONFIG_KEY, None),
            # Optional, can also be set through environment variables
            # in clients.
            api_base=config.pop(API_BASE_CONFIG_KEY, None),
            api_version=config.pop(API_VERSION_CONFIG_KEY, None),
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


def is_azure_openai_config(config: dict) -> bool:
    """Check whether the configuration is meant to configure
    an Azure OpenAI client.
    """
    # Resolve any aliases that are specific to Azure OpenAI configuration
    config = AzureOpenAIClientConfig.resolve_config_aliases(config)

    # Case: Configuration contains `provider: azure`.
    if config.get(PROVIDER_CONFIG_KEY) == AZURE_OPENAI_PROVIDER:
        return True

    # Case: Configuration contains `deployment` key
    # (specific to Azure OpenAI configuration)
    if config.get(DEPLOYMENT_CONFIG_KEY) is not None:
        return True

    return False
