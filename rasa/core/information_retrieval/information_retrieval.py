from __future__ import annotations

from typing import TYPE_CHECKING, List, Text

import structlog

from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from langchain.schema import Document
    from langchain.schema.embeddings import Embeddings


logger = structlog.get_logger()


class InformationRetrievalException(RasaException):
    """Base class for exceptions raised by InformationRetrieval operations."""

    def __init__(self) -> None:
        self.base_message = "An error occurred while searching for documents: "

    def __str__(self) -> str:
        return self.base_message + f"{self.__cause__}"


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

    async def search(
        self,
        query: Text,
        threshold: float = 0.0,
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
    logger.debug(
        "information_retrieval.create_from_endpoint_config", config_type=config_type
    )
    if config_type == "milvus":
        from rasa.core.information_retrieval.milvus import Milvus_Store

        return Milvus_Store(embeddings=embeddings)
    elif config_type == "qdrant":
        from rasa.core.information_retrieval.qdrant import Qdrant_Store

        return Qdrant_Store(embeddings=embeddings)
    else:
        logger.error(
            "information_retrieval.create_from_endpoint_config.unknown_type",
            config_type=config_type,
        )
        raise ValueError(f"Unknown vector store type '{config_type}'.")
