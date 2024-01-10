from typing import TYPE_CHECKING, List, Text

import structlog
from langchain.vectorstores.milvus import Milvus
from rasa.utils.endpoints import EndpointConfig

from rasa.core.information_retrieval.information_retrieval import InformationRetrieval

if TYPE_CHECKING:
    from langchain.schema import Document
    from langchain.schema.embeddings import Embeddings

structlogger = structlog.get_logger()


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

    def search(self, query: Text) -> List["Document"]:
        structlogger.debug("information_retrieval.milvus_store.search", query=query)
        return self.client.similarity_search(query, k=4)
