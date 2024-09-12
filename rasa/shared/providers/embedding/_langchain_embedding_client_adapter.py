from typing import List

from langchain_core.embeddings.embeddings import Embeddings

from rasa.shared.providers.embedding.embedding_client import EmbeddingClient


class _LangchainEmbeddingClientAdapter(Embeddings):
    """
    Temporary adapter to bridge differences between LiteLLM and LangChain.

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
        response = self._client.embed(documents=texts)
        embedding_vector = response.data
        return embedding_vector

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        response = self._client.embed(documents=[text])
        embedding_vector = response.data[0]
        return embedding_vector

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        response = await self._client.aembed(documents=texts)
        embedding_vector = response.data
        return embedding_vector

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        response = await self._client.aembed(documents=[text])
        embedding_vector = response.data[0]
        return embedding_vector
