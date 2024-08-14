from dataclasses import asdict, dataclass, field
from typing import Optional

import structlog

from rasa.shared.constants import (
    MODEL_KEY,
    MODEL_NAME_KEY,
    OPENAI_API_BASE_CONFIG_KEY,
    OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY,
    OPENAI_API_TYPE_CONFIG_KEY,
    OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY,
    OPENAI_API_VERSION_CONFIG_KEY,
    OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY,
    OPENAI_DEPLOYMENT_CONFIG_KEY,
    OPENAI_DEPLOYMENT_NAME_CONFIG_KEY,
    OPENAI_ENGINE_CONFIG_KEY,
    RASA_TYPE_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY,
)
from rasa.shared.utils.io import raise_deprecation_warning

structlogger = structlog.get_logger()
AZURE_API_TYPE = "azure"


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

    # API Type is not actually used by LiteLLM backend, but we define
    # it here for:
    # 1. Backward compatibility.
    # 2. Because it's used as a switch denominator for Azure OpenAI clients.
    api_type: str

    model: Optional[str]
    api_base: Optional[str]
    api_version: Optional[str]
    extra_parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.api_type != AZURE_API_TYPE:
            message = f"API type must be set to '{AZURE_API_TYPE}'."
            structlogger.error(
                "azure_openai_client_config.validation_error",
                message=message,
                api_type=self.api_type,
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
        _raise_deprecation_warnings(config)
        # Resolve any potential aliases
        config = _resolve_aliases(config)
        # Validate that required keys are set
        cls._validate_required_keys(config)
        this = AzureOpenAIClientConfig(
            # Required parameters
            deployment=config.pop(OPENAI_DEPLOYMENT_CONFIG_KEY),
            api_type=config.pop(OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY),
            # Optional
            model=config.pop(MODEL_KEY, None),
            # Optional, can also be set through environment variables
            # in clients.
            api_base=config.pop(OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY, None),
            api_version=config.pop(OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY, None),
            # The rest of parameters (e.g. model parameters) are considered
            # as extra parameters.
            extra_parameters=config,
        )
        return this

    def to_dict(self) -> dict:
        """Converts the config instance into a dictionary."""
        return asdict(self)

    @staticmethod
    def _validate_required_keys(config: dict) -> None:
        """Validates that the passed config is containing
        all the required keys.

        Raises:
            ValueError: The config does not contain required key.
        """
        required_keys = [
            OPENAI_DEPLOYMENT_CONFIG_KEY,
            OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY,
        ]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            message = (
                f"Missing required keys '{missing_keys}' for Azure OpenAI "
                f"client configuration."
            )
            structlogger.error(
                "azure_openai_client_config.validate_required_keys",
                message=message,
                missing_keys=missing_keys,
            )
            raise ValueError(message)


def _resolve_aliases(config: dict) -> dict:
    """
    Resolve aliases in the Azure OpenAI configuration to standard keys for
    LLM/embedding client.

    This function ensures that all configuration keys are standardized by
    replacing any aliases with their corresponding primary keys. It helps in
    maintaining backward compatibility and avoids modifying the original
    dictionary to ensure consistency across multiple usages.

    It does not add new keys if the keys were not previously defined.

    Args:
        config: Dictionary containing the configuration.
    Returns:
        New dictionary containing the processed configuration.

    """
    # Create a new or copied dictionary to avoid modifying the original
    # config, as it's used in multiple places (e.g. command generators).
    config = config.copy()

    # Use `deployment` and if there are any aliases replace them
    deployment = (
        config.get(OPENAI_DEPLOYMENT_NAME_CONFIG_KEY)
        or config.get(OPENAI_DEPLOYMENT_CONFIG_KEY)
        or config.get(OPENAI_ENGINE_CONFIG_KEY)
    )
    if deployment is not None:
        config[OPENAI_DEPLOYMENT_CONFIG_KEY] = deployment

    # Use `api_type` and if there are any aliases replace them
    # In reality, LiteLLM is not using this at all
    # It's here for backward compatibility
    api_type = (
        config.get(OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY)
        or config.get(OPENAI_API_TYPE_CONFIG_KEY)
        or config.get(RASA_TYPE_CONFIG_KEY)
        or config.get(LANGCHAIN_TYPE_CONFIG_KEY)
    )
    if api_type is not None:
        config[OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY] = api_type

    # Use `model` and if there are any aliases replace them
    # In reality, LiteLLM is not using this at all
    # It's here for backward compatibility
    model = config.get(MODEL_NAME_KEY) or config.get(MODEL_KEY)
    if model is not None:
        config[MODEL_KEY] = model

    # Use `api_base` and if there are any aliases replace them
    api_base = config.get(OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY) or config.get(
        OPENAI_API_BASE_CONFIG_KEY
    )
    if api_base is not None:
        config[OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY] = api_base

    # Use `api_version` and if there are any aliases replace them
    api_version = config.get(OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY) or config.get(
        OPENAI_API_VERSION_CONFIG_KEY
    )
    if api_version is not None:
        config[OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY] = api_version

    # Pop all aliases from the config
    for key in [
        OPENAI_API_BASE_CONFIG_KEY,
        OPENAI_API_TYPE_CONFIG_KEY,
        OPENAI_API_VERSION_CONFIG_KEY,
        OPENAI_DEPLOYMENT_NAME_CONFIG_KEY,
        OPENAI_ENGINE_CONFIG_KEY,
        MODEL_NAME_KEY,
        RASA_TYPE_CONFIG_KEY,
        LANGCHAIN_TYPE_CONFIG_KEY,
    ]:
        config.pop(key, None)

    return config


def _raise_deprecation_warnings(config: dict) -> None:
    # Check for `deployment`, `api_base`, `api_type`, `api_version` aliases and
    # raise deprecation warnings.
    _mapper_deprecated_keys_to_new_keys = {
        OPENAI_DEPLOYMENT_NAME_CONFIG_KEY: OPENAI_DEPLOYMENT_CONFIG_KEY,
        OPENAI_ENGINE_CONFIG_KEY: OPENAI_DEPLOYMENT_CONFIG_KEY,
        OPENAI_API_BASE_CONFIG_KEY: OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY,
        OPENAI_API_TYPE_CONFIG_KEY: OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY,
        RASA_TYPE_CONFIG_KEY: OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY,
        LANGCHAIN_TYPE_CONFIG_KEY: OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY,
        OPENAI_API_VERSION_CONFIG_KEY: OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY,
    }
    for deprecated_key, new_key in _mapper_deprecated_keys_to_new_keys.items():
        if deprecated_key in config:
            raise_deprecation_warning(
                message=(
                    f"'{deprecated_key}' is deprecated and will be removed in "
                    f"version 4.0.0. Use '{new_key}' instead."
                )
            )


def is_azure_openai_config(config: dict) -> bool:
    """Check whether the configuration is meant to configure
    an Azure OpenAI client.
    """
    # Resolve any aliases that are specific to Azure OpenAI configuration
    config = _resolve_aliases(config)

    # Case: Configuration contains `api_type: azure`.
    if config.get(OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY) == AZURE_API_TYPE:
        return True

    # Case: Configuration contains `deployment` key
    # (specific to Azure OpenAI configuration)
    if config.get(OPENAI_DEPLOYMENT_CONFIG_KEY) is not None:
        return True

    # Case: Azure OpenAI is defined through the LiteLLM way:
    # `model: azure/deployment_name`.
    #
    # This case would bypass the Rasa's Azure OpenAI client and
    # instantiate the client through the default litellm clients.
    # This expression will recognize this attempt and return
    # `true` if this is the case. However, this config is not
    # valid config to be used within Rasa. We want to avoid having
    # multiple ways to do the same thing. This configuration will
    # result in an error.
    if (model := config.get(MODEL_KEY)) is not None:
        if model.startswith(f"{AZURE_API_TYPE}/"):
            return True

    return False
