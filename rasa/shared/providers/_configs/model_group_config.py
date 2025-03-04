from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import structlog

from rasa.shared.constants import (
    API_BASE_CONFIG_KEY,
    API_KEY,
    API_TYPE_CONFIG_KEY,
    API_VERSION_CONFIG_KEY,
    DEPLOYMENT_CONFIG_KEY,
    EXTRA_PARAMETERS_KEY,
    MODEL_CONFIG_KEY,
    MODEL_GROUP_ID_CONFIG_KEY,
    MODEL_GROUPS_CONFIG_KEY,
    MODELS_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
)
from rasa.shared.providers.mappings import get_client_config_class_from_provider

structlogger = structlog.get_logger()


@dataclass
class ModelConfig:
    """Parses the model config.

    Raises:
        ValueError: If the provider config key is missing in the config.
    """

    provider: str
    model: Optional[str] = None
    deployment: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    extra_parameters: dict = field(default_factory=dict)
    # Retained for backward compatibility with older configurations,
    # but intentionally not included in extra_parameters
    api_type: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict) -> ModelConfig:
        """Initializes a dataclass from the passed config. The provider config param is
        used to determine the client config class to use. The client config class takes
        care of resolving config aliases and throwing deprecation warnings.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Config is missing required keys.

        Returns:
            ModelConfig
        """
        from rasa.shared.utils.llm import get_provider_from_config

        # Get the provider from config, this also inferring the provider from
        # deprecated configurations
        provider = get_provider_from_config(config)

        # Retrieve the client configuration class for the specified provider.
        client_config_clazz = get_client_config_class_from_provider(provider)

        # Try to instantiate the config object in order to resolve deprecated
        # aliases and throw deprecation warnings.
        client_config_obj = client_config_clazz.from_dict(config)

        # Convert back to dictionary and instantiate the ModelConfig object.
        client_config = client_config_obj.to_dict()

        # Check for provider after resolving all aliases
        if PROVIDER_CONFIG_KEY not in client_config:
            raise ValueError(
                f"Missing required key '{PROVIDER_CONFIG_KEY}' in "
                f"'{MODELS_CONFIG_KEY}' config."
            )

        return ModelConfig(
            provider=client_config.pop(PROVIDER_CONFIG_KEY, None),
            model=client_config.pop(MODEL_CONFIG_KEY, None),
            deployment=client_config.pop(DEPLOYMENT_CONFIG_KEY, None),
            api_type=client_config.pop(API_TYPE_CONFIG_KEY, None),
            api_base=client_config.pop(API_BASE_CONFIG_KEY, None),
            api_key=client_config.pop(API_KEY, None),
            api_version=client_config.pop(API_VERSION_CONFIG_KEY, None),
            extra_parameters=client_config,
        )

    def to_dict(self) -> dict:
        """Converts the config instance into a dictionary."""
        d = asdict(self)

        # Extra parameters should also be on the top level
        d.pop(EXTRA_PARAMETERS_KEY, None)
        d.update(self.extra_parameters)

        # Remove keys with None values
        return {key: value for key, value in d.items() if value is not None}


@dataclass
class ModelGroupConfig:
    """Parses the models config. The models config is a list of model configs.

    Raises:
        ValueError: If the model group ID is None or if the models list is empty.
    """

    model_group_id: str
    models: List[ModelConfig]

    def __post_init__(self) -> None:
        if self.model_group_id is None:
            message = "Model group ID cannot be set to None."
            structlogger.error(
                "model_group_config.validation_error",
                message=message,
                model_group_id=self.model_group_id,
            )
            raise ValueError(message)
        if not self.models:
            message = "Models cannot be empty."
            structlogger.error(
                "model_group_config.validation_error",
                message=message,
                model_group_id=self.model_group_id,
            )
            raise ValueError(message)

    @classmethod
    def from_dict(cls, config: dict) -> ModelGroupConfig:
        """Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Config is missing required keys.

        Returns:
            ModelGroupConfig
        """
        if MODELS_CONFIG_KEY not in config:
            raise ValueError(
                f"Missing required key '{MODELS_CONFIG_KEY}' in "
                f"'{MODEL_GROUPS_CONFIG_KEY}' config."
            )

        models_config = [
            ModelConfig.from_dict(model_config)
            for model_config in config[MODELS_CONFIG_KEY]
        ]

        return cls(
            model_group_id=config.get(MODEL_GROUP_ID_CONFIG_KEY),
            models=models_config,
        )

    def to_dict(self) -> dict:
        """Converts the config instance into a dictionary."""
        d = {
            MODEL_GROUP_ID_CONFIG_KEY: self.model_group_id,
            MODELS_CONFIG_KEY: [model.to_dict() for model in self.models],
        }
        return d
