from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.embeddings.embeddings import Embeddings

from rasa.shared.providers.embedding.embedding_client import EmbeddingClient

# Context variable to store embedding metadata for Langfuse tracing
# This allows metadata to be set per-call without changing the LangChain interface
_embedding_metadata: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "embedding_metadata", default=None
)


class _LangchainEmbeddingClientAdapter(Embeddings):
    """Temporary adapter to bridge differences between LiteLLM and LangChain.

    Clients instantiated with `embedder_factory` follow our new EmbeddingClient
    protocol, but `langchain`'s vector stores require an `Embeddings` type
    client. This adapter extracts and returns the necessary part of the output
    from our LiteLLM-based clients.

    This adapter will be removed in ticket:
    https://rasahq.atlassian.net/browse/ENG-1220
    """

    def __init__(self, client: EmbeddingClient):
        self._client = client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        # Extract metadata from context if available
        metadata = get_embedding_metadata()
        kwargs = {"metadata": metadata} if metadata else {}
        response = self._client.embed(documents=texts, **kwargs)
        embedding_vector = response.data
        return embedding_vector

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        # Extract metadata from context if available
        metadata = get_embedding_metadata()
        kwargs = {"metadata": metadata} if metadata else {}
        response = self._client.embed(documents=[text], **kwargs)
        embedding_vector = response.data[0]
        return embedding_vector

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        # Extract metadata from context if available
        metadata = get_embedding_metadata()
        kwargs = {"metadata": metadata} if metadata else {}
        response = await self._client.aembed(documents=texts, **kwargs)
        embedding_vector = response.data
        return embedding_vector

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        # Extract metadata from context if available
        metadata = get_embedding_metadata()
        kwargs = {"metadata": metadata} if metadata else {}
        response = await self._client.aembed(documents=[text], **kwargs)
        embedding_vector = response.data[0]
        return embedding_vector


def set_embedding_metadata(metadata: Optional[Dict[str, Any]]) -> None:
    """Set metadata for embedding calls in the current context.

    This function allows you to set metadata that will be passed to embedding
    calls made through the LangChain adapter. The metadata is stored in a
    context variable, so it works correctly with async code and can be set
    per-call without modifying the LangChain interface.

    Args:
        metadata: Dictionary of metadata to pass to embedding calls.
                 Set to None to clear the metadata.
    """
    _embedding_metadata.set(metadata)


def get_embedding_metadata() -> Optional[Dict[str, Any]]:
    """Get the current embedding metadata from context.

    Returns:
        The current metadata dictionary, or None if no metadata is set.
    """
    return _embedding_metadata.get()


@contextmanager
def embedding_metadata_context(metadata: Dict[str, Any]) -> Iterator[None]:
    """Context manager to set embedding metadata for a block of code.

    This context manager automatically sets and clears metadata, ensuring
    it's properly cleaned up even if an exception occurs.

    Args:
        metadata: Dictionary of metadata to pass to embedding calls.

    Example:
        ```python
        from rasa.shared.providers.embedding._langchain_embedding_client_adapter
          import (
            embedding_metadata_context
        )

        # Metadata is automatically set and cleared
        with embedding_metadata_context({
            "session_id": "user123",
            "component_name": "EnterpriseSearchPolicy"
        }):
            embeddings = embedder.embed_documents(["text"])
            # All embedding calls within this block will include the metadata
        ```
    """
    token = _embedding_metadata.set(metadata)
    try:
        yield
    finally:
        _embedding_metadata.reset(token)
