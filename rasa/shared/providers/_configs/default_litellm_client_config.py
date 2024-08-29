from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import structlog

from rasa.shared.constants import (
    MODEL_CONFIG_KEY,
    MODEL_NAME_CONFIG_KEY,
    STREAM_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
)
from rasa.shared.providers._configs.utils import (
    validate_required_keys,
    validate_forbidden_keys,
)

structlogger = structlog.get_logger()


REQUIRED_KEYS = [MODEL_CONFIG_KEY]

FORBIDDEN_KEYS = [
    STREAM_CONFIG_KEY,
    N_REPHRASES_CONFIG_KEY,
]


@dataclass
class DefaultLiteLLMClientConfig:
    """Parses configuration for default LiteLLM client, resolves aliases and
    raises deprecation warnings.

    Raises:
        ValueError: Raised in cases of invalid configuration:
            - If any of the required configuration keys are missing.
    """

    model: str
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

    @classmethod
    def from_dict(cls, config: dict) -> "DefaultLiteLLMClientConfig":
        """
        Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Config is missing required keys.

        Returns:
            DefaultLiteLLMClientConfig
        """
        # Validate that the required keys are present
        validate_required_keys(config, REQUIRED_KEYS)
        # Validate that the forbidden keys are not present
        validate_forbidden_keys(config, FORBIDDEN_KEYS)
        this = DefaultLiteLLMClientConfig(
            # Required parameters
            model=config.pop(MODEL_CONFIG_KEY),
            # The rest of parameters (e.g. model parameters) are considered
            # as extra parameters
            extra_parameters=config,
        )
        return this

    def to_dict(self) -> dict:
        """Converts the config instance into a dictionary."""
        return asdict(self)


def check_and_error_for_model_name_in_config(config: Dict[str, Any]) -> None:
    """Check for usage of deprecated model_name and raise an error if found."""
    if config.get(MODEL_NAME_CONFIG_KEY) and not config.get(MODEL_CONFIG_KEY):
        event_info = (
            f"Unsupported parameter - {MODEL_NAME_CONFIG_KEY} is set. Please use "
            f"{MODEL_CONFIG_KEY} instead."
        )
        structlogger.error(
            "default_litellm_client_config.unsupported_parameter_in_config",
            event_info=event_info,
        )
        raise KeyError(event_info)
