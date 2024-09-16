from typing import Dict, Any

from rasa.shared.providers._configs.default_litellm_client_config import (
    DefaultLiteLLMClientConfig,
)
from rasa.shared.providers.llm._base_litellm_client import _BaseLiteLLMClient


class DefaultLiteLLMClient(_BaseLiteLLMClient):
    """A default client for interfacing with LiteLLM LLM endpoints.

    Parameters:
        model (str): The model or deployment name.
        kwargs: Any: Additional configuration parameters that can include, but
            are not limited to model parameters and lite-llm specific
            parameters. These parameters will be passed to the
            completion/acompletion calls. To see what it can include, visit:

            https://docs.litellm.ai/docs/completion/input
    Raises:
        ProviderClientValidationError: If validation of the client setup fails.
        ProviderClientAPIException: If the API request fails.
    """

    def __init__(self, provider: str, model: str, **kwargs: Any):
        super().__init__()  # type: ignore
        self._provider = provider
        self._model = model
        self._extra_parameters = kwargs
        self.validate_client_setup()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DefaultLiteLLMClient":
        default_config = DefaultLiteLLMClientConfig.from_dict(config)
        return cls(
            model=default_config.model,
            provider=default_config.provider,
            # Pass the rest of the configuration as extra parameters
            **default_config.extra_parameters,
        )

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        """

        Returns:
        """
        return self._model

    @property
    def config(self) -> Dict:
        """
        Returns the configuration for the openai embedding client.
        Returns:
            Dictionary containing the configuration.
        """
        config = DefaultLiteLLMClientConfig(
            model=self._model,
            provider=self._provider,
            extra_parameters=self._extra_parameters,
        )
        return config.to_dict()

    @property
    def _litellm_model_name(self) -> str:
        """Returns the value of LiteLLM's model parameter to be used in
        completion/acompletion in LiteLLM format:

        <provider>/<model or deployment name>
        """
        if self.model and f"{self.provider}/" not in self.model:
            return f"{self.provider}/{self.model}"
        return self.model

    @property
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        """Returns optional configuration parameters specific
        to the client provider and deployed model.
        """
        return self._extra_parameters
