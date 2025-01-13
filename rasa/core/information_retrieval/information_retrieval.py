from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Text

import structlog

from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from langchain.schema import Document
    from langchain.schema.embeddings import Embeddings


logger = structlog.get_logger()


@dataclass
class SearchResult:
    """A search result object.

    Attributes:
        text: The text content of the retrieved document result.
        metadata: The metadata associated with the document result.
        score: The score of the search result.
    """

    text: str
    metadata: dict
    score: Optional[float] = None

    @classmethod
    def from_document(cls, document: Document) -> "SearchResult":
        """Construct a SearchResult object from Langchain Document object."""
        return cls(text=document.page_content, metadata=document.metadata)


@dataclass
class SearchResultList:
    results: List[SearchResult]
    metadata: dict

    @classmethod
    def from_document_list(cls, documents: List["Document"]) -> "SearchResultList":
        """
        Convert a list of Langchain Documents to a SearchResultList object.

        Args:
            documents: List of Langchain Documents.

        Returns:
            SearchResultList object.
        """
        return cls(
            results=[SearchResult.from_document(doc) for doc in documents],
            metadata={"total_results": len(documents)},
        )


class InformationRetrievalException(RasaException):
    """Base class for exceptions raised by InformationRetrieval operations."""

    def __init__(self) -> None:
        self.base_message = "An error occurred while searching for documents: "

    def __str__(self) -> str:
        return self.base_message + f"{self.__cause__}"


class InformationRetrieval:
    """Base class for any InformationRetrieval implementation."""

    def __init__(self, embeddings: "Embeddings") -> None:
        self.embeddings = embeddings

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
        tracker_state: dict[str, Any],
        threshold: float = 0.0,
    ) -> SearchResultList:
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
        # Import the module dynamically
        try:
            module_name, class_name = config_type.rsplit(".", 1)
            module = importlib.import_module(module_name)
        except ValueError:
            logger.error(
                "information_retrieval.create_from_endpoint_config.invalid_config",
                config_type=config_type,
            )
            raise ValueError(
                f"Invalid configuration for vector store: '{config_type}'. "
                f"Expected a module path and a class name separated by a dot."
            )
        except ModuleNotFoundError:
            logger.error(
                "information_retrieval.create_from_endpoint_config.unknown_type",
                config_type=config_type,
            )
            raise ImportError(f"Cannot retrieve class from path {config_type}.")

        external_class = getattr(module, class_name)
        return external_class(embeddings=embeddings)
