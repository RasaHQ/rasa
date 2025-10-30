from __future__ import annotations

import logging
from typing import Any, Dict, List

import structlog

from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.providers._configs.litellm_router_client_config import (
    LiteLLMRouterClientConfig,
)
from rasa.shared.providers.embedding._base_litellm_embedding_client import (
    _BaseLiteLLMEmbeddingClient,
)
from rasa.shared.providers.embedding.embedding_response import EmbeddingResponse
from rasa.shared.providers.router._base_litellm_router_client import (
    _BaseLiteLLMRouterClient,
)
from rasa.shared.utils.io import suppress_logs

structlogger = structlog.get_logger()


class LiteLLMRouterEmbeddingClient(
    _BaseLiteLLMRouterClient, _BaseLiteLLMEmbeddingClient
):
    """A client for interfacing with LiteLLM Router Embedding endpoints.

    Parameters:
        model_group_id (str): The model group ID.
        model_configurations (List[Dict[str, Any]]): The list of model configurations.
        router_settings (Dict[str, Any]): The router settings.
        kwargs (Optional[Dict[str, Any]]): Additional configuration parameters.

    Raises:
        ProviderClientValidationError: If validation of the client setup fails.
    """

    def __init__(
        self,
        model_group_id: str,
        model_configurations: List[Dict[str, Any]],
        router_settings: Dict[str, Any],
        **kwargs: Any,
    ):
        super().__init__(
            model_group_id, model_configurations, router_settings, **kwargs
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> LiteLLMRouterEmbeddingClient:
        """Instantiates a LiteLLM Router Embedding client from a configuration dict.

        Args:
            config: (Dict[str, Any]) The configuration dictionary.

        Returns:
            LiteLLMRouterLLMClient: The instantiated LiteLLM Router LLM client.

        Raises:
            ValueError: If the configuration is invalid.
        """
        try:
            client_config = LiteLLMRouterClientConfig.from_dict(config)
        except (KeyError, ValueError) as e:
            message = "Cannot instantiate a client from the passed configuration."
            structlogger.error(
                "litellm_router_llm_client.from_config.error",
                message=message,
                config=config,
                original_error=e,
            )
            raise

        return cls(
            model_group_id=client_config.model_group_id,
            model_configurations=client_config.litellm_model_list,
            router_settings=client_config.litellm_router_settings,
            **client_config.extra_parameters,
        )

    @suppress_logs(log_level=logging.WARNING)
    def embed(self, documents: List[str], **kwargs: Any) -> EmbeddingResponse:
        """
        Embeds a list of documents synchronously.

        Args:
            documents: List of documents to be embedded.

        Returns:
            List of embedding vectors.

        Raises:
            ProviderClientAPIException: If API calls raised an error.
        """
        self.validate_documents(documents)
        try:
            response = self.router_client.embedding(
                input=documents, **{**self._embedding_fn_args, **kwargs}
            )
            return self._format_response(response)
        except Exception as e:
            raise ProviderClientAPIException(
                message="Failed to embed documents", original_exception=e
            )

    @suppress_logs(log_level=logging.WARNING)
    async def aembed(self, documents: List[str], **kwargs: Any) -> EmbeddingResponse:
        """
        Embeds a list of documents asynchronously.

        Args:
            documents: List of documents to be embedded.

        Returns:
            List of embedding vectors.

        Raises:
            ProviderClientAPIException: If API calls raised an error.
        """
        self.validate_documents(documents)
        try:
            response = await self.router_client.aembedding(
                input=documents, **{**self._embedding_fn_args, **kwargs}
            )
            return self._format_response(response)
        except Exception as e:
            raise ProviderClientAPIException(
                message="Failed to embed documents", original_exception=e
            )

    @property
    def _embedding_fn_args(self) -> Dict[str, Any]:
        """Returns the arguments to be passed to the embedding function."""
        return {
            **self._litellm_extra_parameters,
            "model": self._model_group_id,
        }
