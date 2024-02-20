from typing import TYPE_CHECKING, List, Text

import structlog
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from rasa.utils.endpoints import EndpointConfig

from rasa.core.information_retrieval.information_retrieval import InformationRetrieval

if TYPE_CHECKING:
    from langchain.schema import Document
    from langchain.schema.embeddings import Embeddings

structlogger = structlog.get_logger()


class Qdrant_Store(InformationRetrieval):
    def __init__(
        self,
        embeddings: "Embeddings",
    ) -> None:
        self.embeddings = embeddings

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
        )

    def search(self, query: Text) -> List["Document"]:
        """Search for a document in the vector store."""
        structlogger.info("information_retrieval.qdrant_store.search", query=query)
        return self.client.similarity_search(query, k=4)
