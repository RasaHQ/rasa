from dataclasses import dataclass, field

from rasa.shared.constants import (
    OPENAI_API_BASE_CONFIG_KEY,
    OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY,
    OPENAI_API_VERSION_CONFIG_KEY,
    OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY,
    OPENAI_DEPLOYMENT_CONFIG_KEY,
    OPENAI_DEPLOYMENT_NAME_CONFIG_KEY,
    OPENAI_ENGINE_CONFIG_KEY,
)


@dataclass
class AzureOpenAIClientConfig:
    deployment: str
    api_base: str
    api_version: str
    client_parameters: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict) -> "AzureOpenAIClientConfig":
        config = cls._process_config(config)
        this = AzureOpenAIClientConfig(
            deployment=config.pop(OPENAI_DEPLOYMENT_CONFIG_KEY),
            api_base=config.pop(OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY),
            api_version=config.pop(OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY),
            client_parameters=config,
        )
        return this

    @staticmethod
    def _process_config(config: dict) -> dict:
        """Process the configuration for the Azure OpenAI llm/embedding client.

        Args:
            config: Dictionary containing the configuration.
        Returns:
            Dictionary containing the processed configuration.
        """
        # use `deployment` and if there are any aliases replace them
        config[OPENAI_DEPLOYMENT_CONFIG_KEY] = (
            config.get(OPENAI_DEPLOYMENT_NAME_CONFIG_KEY)
            or config.get(OPENAI_DEPLOYMENT_CONFIG_KEY)
            or config.get(OPENAI_ENGINE_CONFIG_KEY)
        )

        # use `api_base` and if there are any aliases replace them
        config[OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY] = config.get(
            OPENAI_API_BASE_NO_PREFIX_CONFIG_KEY
        ) or config.get(OPENAI_API_BASE_CONFIG_KEY)

        # use `api_version` and if there are any aliases replace them
        config[OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY] = config.get(
            OPENAI_API_VERSION_NO_PREFIX_CONFIG_KEY
        ) or config.get(OPENAI_API_VERSION_CONFIG_KEY)

        # pop all aliases from the config
        for key in [
            OPENAI_DEPLOYMENT_NAME_CONFIG_KEY,
            OPENAI_ENGINE_CONFIG_KEY,
            OPENAI_API_BASE_CONFIG_KEY,
            OPENAI_API_VERSION_CONFIG_KEY,
        ]:
            config.pop(key, None)

        return config
