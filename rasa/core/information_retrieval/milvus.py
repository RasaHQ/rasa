from typing import Any, Dict, Text

import structlog
from langchain_community.vectorstores.milvus import Milvus

from rasa.core.information_retrieval import (
    InformationRetrieval,
    InformationRetrievalException,
    SearchResultList,
)
from rasa.utils.endpoints import EndpointConfig

logger = structlog.get_logger()


class Milvus_Store(InformationRetrieval):
    """Milvus Store implementation."""

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

    async def search(
        self, query: Text, tracker_state: Dict[str, Any], threshold: float = 0.0
    ) -> SearchResultList:
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

        scores = [score for _, score in hits]
        logger.debug(
            "information_retrieval.milvus_store.search_results_before_threshold",
            scores=scores,
        )
        # Milvus uses Euclidean distance metric by default
        # so the lower the score, the better the match.
        filtered_hits = [doc for doc, score in hits if score <= threshold]
        return SearchResultList.from_document_list(filtered_hits)
