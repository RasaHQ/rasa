from abc import abstractmethod
from typing import Any, Dict, List

import litellm
import structlog
from litellm import aembedding, embedding, validate_environment

from rasa.shared.exceptions import (
    ProviderClientAPIException,
    ProviderClientValidationError,
)
from rasa.shared.providers.embedding.embedding_response import (
    EmbeddingResponse,
    EmbeddingUsage,
)

structlogger = structlog.get_logger()

_VALIDATE_ENVIRONMENT_MISSING_KEYS_KEY = "missing_keys"


class _BaseLiteLLMEmbeddingClient:
    """
    An abstract base class for LiteLLM embedding clients.

    This class defines the interface and common functionality for all clients
    based on LiteLLM.

    The class is made private to prevent it from being part of the
    public-facing interface, as it serves as an internal base class
    for specific implementations of clients that are currently based on
    LiteLLM.

    By keeping it private, we ensure that only the derived, concrete
    implementations are exposed to users, maintaining a cleaner and
    more controlled API surface.
    """

    @property
    @abstractmethod
    def config(self) -> dict:
        """Returns the configuration for that the embedding client in dict form."""
        pass

    @property
    @abstractmethod
    def _litellm_model_name(self) -> str:
        """Returns the model name in LiteLLM format based on the Provider/API type."""
        pass

    @property
    @abstractmethod
    def _litellm_extra_parameters(self) -> Dict[str, Any]:
        """Returns a dictionary of extra parameters which include model
        parameters as well as LiteLLM specific input parameters.
        By default, this returns an empty dictionary (no extra parameters).
        """
        return {}

    @property
    @abstractmethod
    def _embedding_fn_args(self) -> Dict[str, Any]:
        """Returns the arguments to be passed to the embedding function."""
        return {
            **self._litellm_extra_parameters,
            "model": self._litellm_model_name,
        }

    def validate_client_setup(self) -> None:
        """Perform client validation. By default only environment variables
        are validated. Override this method to add more validation steps.

        Raises:
            ProviderClientValidationError if validation fails.
        """
        self._validate_environment_variables()
        self._validate_api_key_not_in_config()

    def _validate_environment_variables(self) -> None:
        """Validate that the required environment variables are set."""
        validation_info = validate_environment(self._litellm_model_name)
        if missing_environment_variables := validation_info.get(
            _VALIDATE_ENVIRONMENT_MISSING_KEYS_KEY
        ):
            event_info = (
                f"Environment variables: {missing_environment_variables} "
                f"not set. Required for API calls."
            )
            structlogger.error(
                "base_litellm_embedding_client.validate_environment_variables",
                event_info=event_info,
                missing_environment_variables=missing_environment_variables,
            )
            raise ProviderClientValidationError(event_info)

    def _validate_api_key_not_in_config(self) -> None:
        if "api_key" in self._litellm_extra_parameters:
            event_info = (
                "API Key is set through `api_key` extra parameter."
                "Set API keys through environment variables."
            )
            structlogger.error(
                "base_litellm_client.validate_api_key_not_in_config",
                event_info=event_info,
            )
            raise ProviderClientValidationError(event_info)

    def validate_documents(self, documents: List[str]) -> None:
        """
        Validates a list of documents to ensure they are suitable for embedding.

        Args:
            documents: List of documents to be validated.

        Raises:
            ValueError: If any document is invalid.
        """
        for doc in documents:
            if not isinstance(doc, str):
                raise ValueError("All documents must be strings.")
            if not doc.strip():
                raise ValueError("Documents cannot be empty or whitespace.")

    def embed(self, documents: List[str]) -> EmbeddingResponse:
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
            response = embedding(input=documents, **self._embedding_fn_args)
            return self._format_response(response)
        except Exception as e:
            raise ProviderClientAPIException(
                message="Failed to embed documents", original_exception=e
            )

    async def aembed(self, documents: List[str]) -> EmbeddingResponse:
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
            response = await aembedding(input=documents, **self._embedding_fn_args)
            return self._format_response(response)
        except Exception as e:
            raise ProviderClientAPIException(
                message="Failed to embed documents", original_exception=e
            )

    def _format_response(
        self, response: litellm.EmbeddingResponse
    ) -> EmbeddingResponse:
        """Parses the LiteLLM EmbeddingResponse to Rasa format.

        Raises:
            ValueError: If any response data is None.
        """

        # If data is not available (None), raise a ValueError
        if response.data is None:
            message = (
                "Failed to embed documents. Received 'None' " "instead of embeddings."
            )
            structlogger.error(
                "base_litellm_client.format_response.data_is_none",
                message=message,
                response=response.to_dict(),
            )
            raise ValueError(message)

        # Sort the embeddings by the "index" key
        response.data.sort(key=lambda x: x["index"])
        # Extract the embedding vectors
        embeddings = [data["embedding"] for data in response.data]
        formatted_response = EmbeddingResponse(
            data=embeddings,
            model=response.model,
        )

        # Process additional usage information if available
        if response.usage:
            completion_tokens = (
                response.usage.completion_tokens
                if hasattr(response.usage, "completion_tokens")
                else 0
            )
            prompt_tokens = (
                response.usage.prompt_tokens
                if hasattr(response.usage, "prompt_tokens")
                else 0
            )
            total_tokens = (
                response.usage.total_tokens
                if hasattr(response.usage, "total_tokens")
                else 0
            )

            formatted_response.usage = EmbeddingUsage(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
            )

        # Log the response with masked data for brevity
        log_response = formatted_response.to_dict()
        log_response["data"] = "Embedding response data not shown here for brevity."
        structlogger.debug(
            "base_litellm_client.formatted_response",
            formatted_response=log_response,
        )
        return formatted_response
