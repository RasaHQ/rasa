from typing import Any, Dict, Optional
import os

from rasa.shared.constants import (
    OPENAI_API_BASE_ENV_VAR,
    OPENAI_API_TYPE_ENV_VAR,
    OPENAI_API_VERSION_ENV_VAR,
)
from rasa.shared.providers._configs.openai_client_config import OpenAIClientConfig
from rasa.shared.providers.embedding._base_litellm_embedding_client import (
    _BaseLiteLLMEmbeddingClient,
)

OPENAI_PROVIDER = "openai"


class OpenAIEmbeddingClient(_BaseLiteLLMEmbeddingClient):
    """A client for interfacing with OpenAI Embeddings.

    Parameters:
        model (str): The OpenAI model name.
        api_base (Optional[str]): Optional, the base URL for the API endpoints.
            If not provided, it will be set via environment variables.
        api_type (Optional[str]): Optional, the type of the API to use.
            If not provided, it will be set via environment variable.
        api_version (Optional[str]): Optional, the version of the API to use.
            If not provided, it will be set via environment variable.
        model_parameters (Optional[Dict[str, Any]]): Configuration parameters specific
            to the model deployment.

    Raises:
        ProviderClientValidationError: If validation of the client setup fails.
    """

    def __init__(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_type: Optional[str] = None,
        api_version: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ):
        self._model = model
        self._api_base = api_base or os.environ.get(OPENAI_API_BASE_ENV_VAR)
        self._api_type = api_type or os.environ.get(OPENAI_API_TYPE_ENV_VAR)
        self._api_version = api_version or os.environ.get(OPENAI_API_VERSION_ENV_VAR)
        self._model_parameters = model_parameters or {}
        self.validate_client_setup()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OpenAIEmbeddingClient":
        openai_config = OpenAIClientConfig.from_dict(config)
        return cls(
            model=openai_config.model,
            api_base=openai_config.api_base,
            api_type=openai_config.api_type,
            api_version=openai_config.api_version,
            model_parameters=openai_config.model_parameters,
        )

    @property
    def config(self) -> Dict:
        """
        Returns the configuration for the openai embedding client.

        Returns:
            Dictionary containing the configuration.
        """
        config = OpenAIClientConfig(
            model=self.model,
            api_base=self.api_base,
            api_type=self.api_type,
            api_version=self.api_version,
            model_parameters=self.model_parameters,
        )
        return config.to_dict()

    @property
    def model(self) -> str:
        """
        Returns the model name for the openai embedding client.

        Returns:
            String representing the model name.
        """
        return self._model

    @property
    def provider(self) -> str:
        """Return the provider name."""
        return OPENAI_PROVIDER

    @property
    def api_base(self) -> Optional[str]:
        """
        Returns the base API URL for the openai embedding client.

        Returns:
            String representing the base API URL.
        """
        return self._api_base

    @property
    def api_type(self) -> Optional[str]:
        """
        Returns the API type for the openai embedding client.

        Returns:
            String representing the API version.
        """
        return self._api_type

    @property
    def api_version(self) -> Optional[str]:
        """
        Returns the API version for the openai embedding client.

        Returns:
            String representing the API version.
        """
        return self._api_version

    @property
    def model_parameters(self) -> Dict[str, Any]:
        """
        Returns the model parameters for the openai embedding client.

        Returns:
            Dictionary containing the model parameters.
        """
        return self._model_parameters

    @property
    def _embedding_fn_args(self) -> Dict[str, Any]:
        return {
            "model": f"{self.provider}/{self.model}",
            "api_base": self.api_base,
            "api_type": self.api_type,
            "api_version": self.api_version,
            **self.model_parameters,
        }
