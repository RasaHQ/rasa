from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import structlog

from rasa.exceptions import ValidationError
from rasa.shared.constants import (
    MODEL_CONFIG_KEY,
    MODEL_NAME_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
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
    # Timeout aliases
    REQUEST_TIMEOUT_CONFIG_KEY: TIMEOUT_CONFIG_KEY,
}

REQUIRED_KEYS = [MODEL_CONFIG_KEY, PROVIDER_CONFIG_KEY]

FORBIDDEN_KEYS = [
    STREAM_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
]


@dataclass
class DefaultLiteLLMClientConfig:
    """Parses configuration for default LiteLLM client.

    Resolves aliases and raises deprecation warnings.

    Raises:
        ValueError: Raised in cases of invalid configuration:
            - If any of the required configuration keys are missing.
    """

    model: str
    provider: str
    extra_parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model is None:
            message = "Model cannot be set to None."
            structlogger.error(
                "default_litellm_client_config.validation_error",
                message=message,
                model=self.model,
            )
            raise ValueError(message)
        if self.provider is None:
            message = "Provider cannot be set to None."
            structlogger.error(
                "default_litellm_client_config.validation_error",
                message=message,
                provider=self.provider,
            )
            raise ValueError(message)

    @classmethod
    def from_dict(cls, config: dict) -> DefaultLiteLLMClientConfig:
        """Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Config is missing required keys.

        Returns:
            DefaultLiteLLMClientConfig
        """
        # Check for deprecated keys
        raise_deprecation_warnings(config, DEPRECATED_ALIASES_TO_STANDARD_KEY_MAPPING)
        # Raise error for using `model_name` instead instead of `model`
        cls.check_and_error_for_model_name_in_config(config)
        # Resolve any potential aliases.
        config = cls.resolve_config_aliases(config)
        # Validate that the required keys are present
        validate_required_keys(config, REQUIRED_KEYS)
        # Validate that the forbidden keys are not present
        validate_forbidden_keys(config, FORBIDDEN_KEYS)
        this = DefaultLiteLLMClientConfig(
            # Required parameters
            model=config.pop(MODEL_CONFIG_KEY),
            provider=config.pop(PROVIDER_CONFIG_KEY),
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

    @staticmethod
    def check_and_error_for_model_name_in_config(config: Dict[str, Any]) -> None:
        """Check for usage of deprecated model_name and raise an error if found."""
        if config.get(MODEL_NAME_CONFIG_KEY) and not config.get(MODEL_CONFIG_KEY):
            event_info = (
                f"Unsupported parameter - {MODEL_NAME_CONFIG_KEY} is set. Please use "
                f"{MODEL_CONFIG_KEY} instead."
            )
            error_code = "default_litellm_client_config.unsupported_parameter_in_config"
            raise ValidationError(code=error_code, event_info=event_info, config=config)
