from __future__ import annotations

from typing import TYPE_CHECKING, List, Text

import structlog
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from langchain.schema import Document
    from langchain.schema.embeddings import Embeddings


structlogger = structlog.get_logger()


class InformationRetrieval:
    """Base class for any InformationRetrieval implementation."""

    def connect(
        self,
        config: EndpointConfig,
    ) -> None:
        """Connect to the InformationRetrieval system."""
        raise NotImplementedError(
            "InformationRetrieval must implement the `connect` method."
        )

    def search(
        self,
        query: Text,
    ) -> List["Document"]:
        """Search for a document in the InformationRetrieval system."""
        raise NotImplementedError(
            "InformationRetrieval must implement the `search` method."
        )


def create_from_endpoint_config(
    config_type: Text,
    embeddings: "Embeddings",
) -> InformationRetrieval:
    """Instantiate a vector store based on its configuration."""
    structlogger.debug(
        "information_retrieval.create_from_endpoint_config", config_type=config_type
    )
    if config_type == "milvus":
        from rasa.core.information_retrieval.milvus import Milvus_Store

        return Milvus_Store(embeddings=embeddings)
    elif config_type == "qdrant":
        from rasa.core.information_retrieval.qdrant import Qdrant_Store

        return Qdrant_Store(embeddings=embeddings)
    else:
        structlogger.error(
            "information_retrieval.create_from_endpoint_config.unknown_type",
            config_type=config_type,
        )
        raise ValueError(f"Unknown vector store type '{config_type}'.")
