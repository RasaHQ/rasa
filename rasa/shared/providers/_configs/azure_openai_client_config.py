from dataclasses import dataclass, field, asdict
from typing import Optional

from rasa.shared.constants import (
    OPENAI_API_BASE_CONFIG_KEY,
    OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY,
    OPENAI_API_VERSION_CONFIG_KEY,
    OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY,
    OPENAI_DEPLOYMENT_CONFIG_KEY,
    OPENAI_DEPLOYMENT_NAME_CONFIG_KEY,
    OPENAI_ENGINE_CONFIG_KEY,
    MODEL_KEY,
    OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY,
    MODEL_NAME_KEY,
    OPENAI_API_TYPE_CONFIG_KEY,
)
from rasa.shared.utils.io import raise_deprecation_warning


@dataclass
class AzureOpenAIClientConfig:
    deployment: str
    api_base: str
    api_version: str
    api_type: Optional[str] = None
    model: Optional[str] = None
    model_parameters: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict) -> "AzureOpenAIClientConfig":
        config = cls._process_config(config)
        this = AzureOpenAIClientConfig(
            deployment=config.pop(OPENAI_DEPLOYMENT_CONFIG_KEY),
            model=config.pop(MODEL_KEY, None),
            api_base=config.pop(OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY),
            api_version=config.pop(OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY),
            api_type=config.pop(OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY, None),
            # The rest of parameters are considered as model parameters
            model_parameters=config,
        )
        return this

    def to_dict(self) -> dict:
        """Converts the config instance into a dictionary."""
        return asdict(self)

    @staticmethod
    def _process_config(config: dict) -> dict:
        """Process the configuration for the Azure OpenAI llm/embedding client.

        Args:
            config: Dictionary containing the configuration.
        Returns:
            Dictionary containing the processed configuration.
        """
        # Check for deprecated keys
        AzureOpenAIClientConfig._raise_deprecation_warnings(config)

        # Create a new or copied dictionary to avoid modifying the original
        # config, as it's used in multiple places (e.g. command generators).
        config = config.copy()

        # Use `deployment` and if there are any aliases replace them
        config[OPENAI_DEPLOYMENT_CONFIG_KEY] = (
            config.get(OPENAI_DEPLOYMENT_NAME_CONFIG_KEY)
            or config.get(OPENAI_DEPLOYMENT_CONFIG_KEY)
            or config.get(OPENAI_ENGINE_CONFIG_KEY)
        )

        # Use `model` and if there are any aliases replace them
        # In reality, LiteLLM is not using this at all
        # It's here for backward compatibility
        config[MODEL_KEY] = config.get(MODEL_NAME_KEY) or config.get(MODEL_KEY)

        # Use `api_base` and if there are any aliases replace them
        config[OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY] = config.get(
            OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY
        ) or config.get(OPENAI_API_BASE_CONFIG_KEY)

        # Use `api_version` and if there are any aliases replace them
        config[OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY] = config.get(
            OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY
        ) or config.get(OPENAI_API_VERSION_CONFIG_KEY)

        # Use `api_type` and if there are any aliases replace them
        config[OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY] = config.get(
            OPENAI_API_TYPE_CONFIG_KEY
        ) or config.get(OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY)

        # Pop all aliases from the config
        for key in [
            OPENAI_DEPLOYMENT_NAME_CONFIG_KEY,
            OPENAI_ENGINE_CONFIG_KEY,
            OPENAI_API_BASE_CONFIG_KEY,
            OPENAI_API_VERSION_CONFIG_KEY,
            OPENAI_API_TYPE_CONFIG_KEY,
            MODEL_NAME_KEY,
        ]:
            config.pop(key, None)

        return config

    @staticmethod
    def _raise_deprecation_warnings(config: dict) -> None:
        # Check for `deployment` aliases and raise deprecation warnings
        for deprecated_deployment_alias in [
            OPENAI_DEPLOYMENT_NAME_CONFIG_KEY,
            OPENAI_ENGINE_CONFIG_KEY,
        ]:
            if deprecated_deployment_alias in config:
                raise_deprecation_warning(
                    message=(
                        f"'{deprecated_deployment_alias}' is deprecated and "
                        f"will be removed in version 4.0.0. Use "
                        f"'{OPENAI_DEPLOYMENT_CONFIG_KEY}' instead."
                    )
                )

        # Check for `api_base` aliases and raise deprecation warnings
        if OPENAI_API_BASE_CONFIG_KEY in config:
            raise_deprecation_warning(
                message=(
                    f"'{OPENAI_API_BASE_CONFIG_KEY}' is deprecated and will be"
                    f"removed in version 4.0.0. Use "
                    f"'{OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY}' instead."
                )
            )

        # Check for `api_version` aliases and raise deprecation warnings
        if OPENAI_API_VERSION_CONFIG_KEY in config:
            raise_deprecation_warning(
                message=(
                    f"'{OPENAI_API_VERSION_CONFIG_KEY}' is deprecated and will be"
                    f"removed in version 4.0.0. Use "
                    f"'{OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY}' instead."
                )
            )

        # Check for `api_type` aliases and raise deprecation warnings
        if OPENAI_API_TYPE_CONFIG_KEY in config:
            raise_deprecation_warning(
                message=(
                    f"'{OPENAI_API_TYPE_CONFIG_KEY}' is deprecated and will be"
                    f"removed in version 4.0.0. Use "
                    f"'{OPENAI_API_TYPE_NO_PREFIX_CONFIG_KEY}' instead."
                )
            )
