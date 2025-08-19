from typing import Any, Dict

from rasa.shared.constants import (
    AWS_BEDROCK_PROVIDER,
    AWS_SAGEMAKER_PROVIDER,
)
from rasa.shared.providers._configs.default_litellm_client_config import (
    DefaultLiteLLMClientConfig,
)
from rasa.shared.providers._utils import validate_aws_setup_for_litellm_clients
from rasa.shared.providers.embedding._base_litellm_embedding_client import (
    _BaseLiteLLMEmbeddingClient,
)


class DefaultLiteLLMEmbeddingClient(_BaseLiteLLMEmbeddingClient):
    """A default client for interfacing with LiteLLM Embedding endpoints.

    Parameters:
        model (str): The OpenAI model name.
        kwargs: Any: Additional configuration parameters that can include, but
            are not limited to model parameters and lite-llm specific
            parameters. These parameters will be passed to the
            embedding/aembedding calls.

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
    def from_config(cls, config: Dict[str, Any]) -> "DefaultLiteLLMEmbeddingClient":
        """Creates a DefaultLiteLLMEmbeddingClient instance from a config dict."""
        default_config = DefaultLiteLLMClientConfig.from_dict(config)
        return cls(
            model=default_config.model,
            provider=default_config.provider,
            # Pass the rest of the configuration as extra parameters
            **default_config.extra_parameters,
        )

    @property
    def config(self) -> Dict:
        """
        Returns the configuration for the default litellm embedding client.

        Returns:
            Dictionary containing the configuration.
        """
        config = DefaultLiteLLMClientConfig(
            model=self.model,
            provider=self.provider,
            extra_parameters=self._extra_parameters,
        )
        return config.to_dict()

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        """
        Returns the model name for the default litellm embedding client.

        Returns:
            String representing the model name.
        """
        return self._model

    @property
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        """
        Returns optional configuration parameters specific to the client provider
        and deployed model.

        Returns:
            Dictionary containing the model parameters.
        """
        return self._extra_parameters

    @property
    def _litellm_model_name(self) -> str:
        """
        Returns the value of LiteLLM's model parameter to be used in
        embedding/aembedding in LiteLLM format:
        <provider>/<model or deployment name>

        Returns:
            String representing the model name in LiteLLM format.
        """
        if self.model and f"{self.provider}/" not in self.model:
            return f"{self.provider}/{self.model}"
        return self.model

    @property
    def _embedding_fn_args(self) -> Dict[str, Any]:
        return {
            "model": self._litellm_model_name,
            **self._litellm_extra_parameters,
        }

    def validate_client_setup(self) -> None:
        # TODO: Temporarily disable environment variable validation for AWS setup
        #       (Bedrock and SageMaker) until resolved by either:
        #       1. An update from the LiteLLM package addressing the issue.
        #       2. The implementation of a Bedrock client on our end.
        #       ---
        #       This fix ensures a consistent user experience for Bedrock (and
        #       SageMaker) in Rasa by allowing AWS secrets to be provided as extra
        #       parameters without triggering validation errors due to missing AWS
        #       environment variables.
        if self.provider.lower() in [AWS_BEDROCK_PROVIDER, AWS_SAGEMAKER_PROVIDER]:
            validate_aws_setup_for_litellm_clients(
                self._litellm_model_name,
                self._litellm_extra_parameters,
                "default_litellm_embedding_client",
                provider=self.provider,
            )
        else:
            super().validate_client_setup()
