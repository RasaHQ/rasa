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
    RASA_TYPE_CONFIG_KEY,
    LANGCHAIN_TYPE_CONFIG_KEY,
)
from rasa.shared.utils.io import raise_deprecation_warning

structlogger = structlog.get_logger()
OPENAI_API_TYPE = "openai"


@dataclass
class OpenAIClientConfig:
    """Parses configuration for Azure OpenAI client, resolves aliases and
    raises deprecation warnings.

    Raises:
        ValueError: Raised in cases of invalid configuration:
            - If any of the required configuration keys are missing.
            - If `api_type` has a value different from `openai`.
    """

    model: str

    # API Type is not actually used by LiteLLM backend, but we define
    # it here for:
    # 1. Backward compatibility.
    # 2. Because it's used as a switch denominator for Azure OpenAI clients.
    api_type: str

    api_base: Optional[str]
    api_version: Optional[str]
    extra_parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.api_type != OPENAI_API_TYPE:
            message = f"API type must be set to '{OPENAI_API_TYPE}'."
            structlogger.error(
                "openai_client_config.validation_error",
                message=message,
                api_type=self.api_type,
            )
            raise ValueError(message)

    @classmethod
    def from_dict(cls, config: dict) -> "OpenAIClientConfig":
        """
        Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Config is missing required keys.

        Returns:
            AzureOpenAIClientConfig
        """
        config = cls._process_config(config)
        cls._validate_required_keys(config)
        this = OpenAIClientConfig(
            # Required parameters
            model=config.pop(MODEL_KEY),
            api_type=config.pop(OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY),
            # Optional parameters
            api_base=config.pop(OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY, None),
            api_version=config.pop(OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY, None),
            # The rest of parameters (e.g. model parameters) are considered
            # as extra parameters
            extra_parameters=config,
        )
        return this

    def to_dict(self) -> dict:
        """Converts the config instance into a dictionary."""
        return asdict(self)

    @staticmethod
    def _process_config(config: dict) -> dict:
        # Check for deprecated keys
        _raise_deprecation_warnings(config)
        # Resolve any potential aliases
        config = _resolve_aliases(config)
        return config

    @staticmethod
    def _validate_required_keys(config: dict) -> None:
        """Validates that the passed config is containing
        all the required keys.

        Raises:
            ValueError: The config does not contain required key.
        """
        required_keys = [MODEL_KEY, OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            message = (
                f"Missing required keys '{missing_keys}' for OpenAI "
                f"client configuration."
            )
            structlogger.error(
                "openai_client_config.validate_required_keys",
                message=message,
                missing_keys=missing_keys,
            )
            raise ValueError(message)


def _resolve_aliases(config: dict) -> dict:
    """
    Process the configuration for the OpenAI llm/embedding client.

    Args:
        config: Dictionary containing the configuration.
    Returns:
        New dictionary containing the processed configuration.
    """
    # Create a new or copied dictionary to avoid modifying the original
    # config, as it's used in multiple places (e.g. command generators).
    config = config.copy()

    # Use `model` and if there are any aliases replace them
    config[MODEL_KEY] = config.get(MODEL_NAME_KEY) or config.get(MODEL_KEY)

    # Use `api_type` and if there are any aliases replace them
    # In reality, LiteLLM is not using this at all
    # It's here for backward compatibility
    config[OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY] = (
        config.get(OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY)
        or config.get(OPENAI_API_TYPE_CONFIG_KEY)
        or config.get(RASA_TYPE_CONFIG_KEY)
        or config.get(LANGCHAIN_TYPE_CONFIG_KEY)
    )

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

    # Pop the keys so there are no duplicates
    for key in [
        MODEL_NAME_KEY,
        OPENAI_API_BASE_CONFIG_KEY,
        OPENAI_API_TYPE_CONFIG_KEY,
        OPENAI_API_VERSION_CONFIG_KEY,
        RASA_TYPE_CONFIG_KEY,
        LANGCHAIN_TYPE_CONFIG_KEY,
    ]:
        config.pop(key, None)

    return config


def _raise_deprecation_warnings(config: dict) -> None:
    # Check for `model`, `api_base`, `api_type`, `api_version` aliases and
    # raise deprecation warnings.
    _mapper_deprecated_keys_to_new_keys = {
        MODEL_NAME_KEY: MODEL_KEY,
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


def is_openai_config(config: dict) -> bool:
    """Check whether the configuration is meant to configure
    an OpenAI client.
    """

    from litellm.utils import get_llm_provider

    # Process the config to handle all the aliases
    config = _resolve_aliases(config)

    # Case: Configuration contains `api_type: openai`
    if config.get(OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY) == OPENAI_API_TYPE:
        return True

    # Case: Configuration contains `model: openai/gpt-4` (litellm approach)
    #
    # This case would bypass the Rasa's Azure OpenAI client and
    # instantiate the client through the default litellm clients.
    # This expression will recognize this attempt and return
    # `true` if this is the case. However, this config is not
    # valid config to be used within Rasa. We want to avoid having
    # multiple ways to do the same thing. This configuration will
    # result in an error.
    if (model := config.get(MODEL_KEY)) is not None:
        if model.startswith(f"{OPENAI_API_TYPE}/"):
            return True

    # Case: Configuration contains "known" models of openai (litellm approach)
    #
    # Similar to the case above.
    try:
        _, provider, _, _ = get_llm_provider(config.get(MODEL_KEY))
        if provider == OPENAI_API_TYPE:
            return True
    except Exception:
        pass

    return False
