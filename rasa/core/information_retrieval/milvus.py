from typing import TYPE_CHECKING, List, Text

import structlog
from langchain.vectorstores.milvus import Milvus
from rasa.utils.endpoints import EndpointConfig

from rasa.core.information_retrieval.information_retrieval import (
    InformationRetrieval,
    InformationRetrievalException,
)

if TYPE_CHECKING:
    from langchain.schema import Document
    from langchain.schema.embeddings import Embeddings

logger = structlog.get_logger()


class Milvus_Store(InformationRetrieval):
    """Milvus Store implementation."""

    def __init__(
        self,
        embeddings: "Embeddings",
    ):
        self.embeddings = embeddings

    def connect(self, config: EndpointConfig) -> None:
        """Connect to the Milvus system."""
        params = config.kwargs
        self.client = Milvus(
            self.embeddings,
            connection_args={
                "host": str(params.get("host")),
                "port": str(params.get("port")),
                "user": str(params.get("user")),
                "password": str(params.get("password")),
            },
            collection_name=str(params.get("collection")),
        )

    async def search(self, query: Text, threshold: float = 0.0) -> List["Document"]:
        """Search for documents in the Milvus store.

        Args:
            query: The query to search for.
            threshold: minimum similarity score to consider a document a match.

        Returns:
        A list of documents that match the query.
        """
        logger.debug("information_retrieval.milvus_store.search", query=query)
        try:
            hits = await self.client.asimilarity_search_with_score(query, k=4)
        except Exception as exc:
            raise InformationRetrievalException from exc

        filtered_hits = [doc for doc, score in hits if score >= threshold]
        return filtered_hits
