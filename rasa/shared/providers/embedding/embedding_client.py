from typing import Protocol, Dict, List


class EmbeddingClient(Protocol):
    @property
    def config(self) -> Dict:
        """
        Returns the configuration for the embedding client.

        This property should be implemented to return a dictionary containing
        the configuration settings for the embedding client.
        """
        ...

    @property
    def embedding_size(self) -> int:
        """
        Returns the size of the embedding vector.

        This property should be implemented to return an integer representing
        the size of the embedding vector.
        """
        ...

    @classmethod
    def get_default_config(cls) -> Dict:
        """
        Returns the default configuration for the embedding client.

        This class method should be implemented to return a dictionary containing
        the default configuration settings for the embedding client.
        """
        ...

    def embed(self, documents: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents synchronously.

        This method should be implemented to take a list of documents (as strings)
        and return a list of embedding vectors (each represented as a list of floats).

        Args:
            documents: List of documents to be embedded.

        Returns:
            List of embedding vectors.
        """
        ...

    async def aembed(self, documents: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents asynchronously.

        This method should be implemented to take a list of documents (as strings)
        and return a list of embedding vectors (each represented as a list of floats).

        Args:
            documents: List of documents to be embedded.

        Returns:
            List of embedding vectors.
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
