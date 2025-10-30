from typing import TYPE_CHECKING, Any, Dict, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rasa.shared.providers.embedding.embedding_response import EmbeddingResponse


@runtime_checkable
class EmbeddingClient(Protocol):
    @classmethod
    def from_config(cls, config: dict) -> "EmbeddingClient":
        """
        Initializes the embedding client with the given configuration.

        This class method should be implemented to parse the given
        configuration and create an instance of llm client.
        """
        ...

    @property
    def config(self) -> Dict:
        """
        Returns the configuration for the embedding client.

        This property should be implemented to return a dictionary containing
        the configuration settings for the embedding client.
        """
        ...

    def embed(self, documents: List[str], **kwargs: Any) -> "EmbeddingResponse":
        """
        Embeds a list of documents synchronously.

        This method should be implemented to take a list of documents (as strings)
        and return a list of embedding vectors (each represented as a list of floats).

        Args:
            documents: List of documents to be embedded.
            **kwargs: Additional parameters forwarded to the underlying provider call
                (e.g. tracing metadata or provider-specific options).

        Returns:
            List of embedding vectors.

        Raises:
            ProviderClientAPIException: If API calls raised an error.
        """
        ...

    async def aembed(self, documents: List[str], **kwargs: Any) -> "EmbeddingResponse":
        """
        Embeds a list of documents asynchronously.

        This method should be implemented to take a list of documents (as strings)
        and return a list of embedding vectors (each represented as a list of floats).

        Args:
            documents: List of documents to be embedded.
            **kwargs: Additional parameters forwarded to the underlying provider call
                (e.g. tracing metadata or provider-specific options).

        Returns:
            List of embedding vectors.

        Raises:
            ProviderClientAPIException: If API calls raised an error.
        """
        ...

    def validate_documents(self, documents: List[str]) -> None:
        """
        Validates a list of documents to ensure they are suitable for embedding.

        This method should be implemented for document validation. It should raise
        a ValueError if any document is invalid.

        Args:
            documents: List of documents to be validated.

        Raises:
            ValueError: If any document is invalid.
        """
        ...

    def validate_client_setup(self, *args, **kwargs) -> None:  # type: ignore
        """
        Perform client setup validation.

        This method should be implemented to validate whether the client can be used
        with the parameters provided through configuration or environment variables.

        If there are any issues, the client should raise a ValidationError.
        If no validation is needed, this check can simply pass.
        """
        ...
