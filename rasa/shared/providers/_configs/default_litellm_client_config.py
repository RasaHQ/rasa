from dataclasses import asdict, dataclass, field

import structlog

from rasa.shared.constants import (
    MODEL_CONFIG_KEY,
)

structlogger = structlog.get_logger()
OPENAI_API_TYPE = "openai"


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
        cls._validate_required_keys(config)
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

    @staticmethod
    def _validate_required_keys(config: dict) -> None:
        """Validates that the passed config is containing
        all the required keys.

        Raises:
            ValueError: The config does not contain required key.
        """
        if MODEL_CONFIG_KEY not in config:
            message = (
                f"Missing required key '{MODEL_CONFIG_KEY}' for "
                f"client configuration."
            )
            structlogger.error(
                "default_litellm_client_config.validate_required_keys",
                message=message,
            )
            raise ValueError(message)
