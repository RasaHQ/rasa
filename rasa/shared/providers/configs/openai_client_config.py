from dataclasses import dataclass, field
from typing import Optional

from rasa.shared.constants import (
    OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY,
    OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY,
    MODEL_KEY,
    MODEL_NAME_KEY,
    OPENAI_API_BASE_CONFIG_KEY,
    OPENAI_API_VERSION_CONFIG_KEY,
)


@dataclass
class OpenAIClientConfig:
    model: str
    api_base: Optional[str]
    api_version: Optional[str]
    client_parameters: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict) -> "OpenAIClientConfig":
        config = cls._process_config(config)
        this = OpenAIClientConfig(
            api_base=config.pop(OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY),
            api_version=config.pop(OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY),
            client_parameters=config,
        )
        return this

    @staticmethod
    def _process_config(config: dict) -> dict:
        """
        Process the configuration for the OpenAI llm/embedding client.

        Args:
            config: Dictionary containing the configuration.
        Returns:
            Dictionary containing the processed configuration.
        """
        # base config processing
        config = config.copy()

        # use `model` and if there are any aliases replace them
        config[MODEL_KEY] = config.get(MODEL_NAME_KEY) or config.get(MODEL_KEY)

        # use `api_base` and if there are any aliases replace them
        config[OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY] = config.get(
            OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY
        ) or config.get(OPENAI_API_BASE_CONFIG_KEY)

        # use `api_version` and if there are any aliases replace them
        config[OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY] = config.get(
            OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY
        ) or config.get(OPENAI_API_VERSION_CONFIG_KEY)

        # pop the keys so there are no duplicates
        for key in [
            MODEL_NAME_KEY,
            OPENAI_API_BASE_CONFIG_KEY,
            OPENAI_API_VERSION_CONFIG_KEY,
        ]:
            config.pop(key, None)

        return config
