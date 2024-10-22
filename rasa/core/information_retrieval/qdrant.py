from typing import Text, Any, Dict

import structlog
from langchain_community.vectorstores.qdrant import Qdrant
from pydantic import ValidationError
from qdrant_client import QdrantClient
from rasa.utils.endpoints import EndpointConfig

from rasa.core.information_retrieval import (
    SearchResultList,
    InformationRetrieval,
    InformationRetrievalException,
)

logger = structlog.get_logger()


class PayloadNotFoundException(InformationRetrievalException):
    """Exception raised for errors in missing payloads."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__()

    def __str__(self) -> str:
        return self.base_message + self.message + f"{self.__cause__}"


class QdrantInformationRetrievalException(InformationRetrievalException):
    """Exception raised for errors in the Qdrant vector store."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__()

    def __str__(self) -> str:
        return self.base_message + self.message + f"{self.__cause__}"


class Qdrant_Store(InformationRetrieval):
    def connect(
        self,
        config: EndpointConfig,
    ) -> None:
        """Connect to the Qdrant system."""
        params = config.kwargs
        self.client = Qdrant(
            client=QdrantClient(
                location=params.get("location"),
                url=params.get("url"),
                port=int(params.get("port", 6333)),
                grpc_port=int(params.get("grpc_port", 6334)),
                prefer_grpc=bool(params.get("prefer_grpc", False)),
                https=bool(params.get("https")),
                api_key=params.get("api_key"),
                prefix=params.get("prefix"),
                timeout=int(params.get("timeout", 5)),
                host=params.get("host"),
                path=params.get("path"),
            ),
            collection_name=str(params.get("collection")),
            embeddings=self.embeddings,
            content_payload_key=params.get("content_payload_key", "text"),
            metadata_payload_key=params.get("metadata_payload_key", "metadata"),
        )

    async def search(
        self, query: Text, tracker_state: Dict[str, Any], threshold: float = 0.0
    ) -> SearchResultList:
        """Search for a document in the Qdrant vector store.

        Args:
            query: The query to search for.
            threshold: minimum similarity score to consider a document a match.

        Returns:
        A list of documents that match the query.
        """
        logger.debug("information_retrieval.qdrant_store.search", query=query)
        try:
            hits = await self.client.asimilarity_search(
                query, k=4, score_threshold=threshold
            )
        except ValidationError as e:
            raise PayloadNotFoundException(
                "Payload not found in the Qdrant response. Please make sure "
                "the `content_payload_key`and `metadata_payload_key` are correct in "
                f"the Qdrant configuration. Error: {e}"
                ""
            ) from e
        except Exception as e:
            raise QdrantInformationRetrievalException(
                f"Failed to search the Qdrant vector store. Encountered error: {e}"
            ) from e
        return SearchResultList.from_document_list(hits)
