from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import structlog

from rasa.shared.constants import (
    MODEL_CONFIG_KEY,
    RASA_PROVIDER,
    PROVIDER_CONFIG_KEY,
    API_BASE_CONFIG_KEY,
)
from rasa.shared.providers._configs.utils import (
    validate_required_keys,
)

REQUIRED_KEYS = [MODEL_CONFIG_KEY, PROVIDER_CONFIG_KEY, API_BASE_CONFIG_KEY]

structlogger = structlog.get_logger()

@dataclass
class RasaLLMClientConfig:
    """Parses configuration for a Rasa Hosted LiteLLM client, checks required keys present.

    Raises:
        ValueError: Raised in cases of invalid configuration:
            - If any of the required configuration keys are missing.
    """

    model: Optional[str]
    api_base: Optional[str]
    # Provider is not used by LiteLLM backend, but we define it here since it's
    # used as switch between different clients.
    provider: str = RASA_PROVIDER

    extra_parameters: dict = field(default_factory=dict)


    @classmethod
    def from_dict(cls, config: dict) -> "RasaLLMClientConfig":
        """
        Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Raises:
            ValueError: Raised in cases of invalid configuration:
                - If any of the required configuration keys are missing.
                - If `api_type` has a value different from `azure`.

        Returns:
            RasaLLMClientConfig
        """
        # Validate that required keys are set
        validate_required_keys(config, REQUIRED_KEYS)

        # Init client config
        this = RasaLLMClientConfig(
            model=config.pop(MODEL_CONFIG_KEY, None),
            api_base=config.pop(API_BASE_CONFIG_KEY, None),
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
