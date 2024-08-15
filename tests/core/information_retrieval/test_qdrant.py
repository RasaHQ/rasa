from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch
from langchain.schema.embeddings import Embeddings
from langchain_community.vectorstores.qdrant import Qdrant

from rasa.core.information_retrieval import InformationRetrievalException
from rasa.utils.endpoints import EndpointConfig
from rasa.core.information_retrieval.qdrant import (
    PayloadNotFoundException,
    QdrantInformationRetrievalException,
    Qdrant_Store,
)


def test_qdrant_store(embeddings: Embeddings) -> None:
    client = Qdrant_Store(embeddings)
    assert client is not None
    assert isinstance(client, Qdrant_Store)
    assert client.embeddings is not None
    assert isinstance(client.embeddings, Embeddings)


def test_qdrant_store_connect(embeddings: Embeddings) -> None:
    client = Qdrant_Store(embeddings)
    client.connect(
        EndpointConfig(
            location=":memory:",
            collection_name="test",
            content_payload_key="content",
            metadata_payload_key="extra",
        )
    )
    assert client.client is not None
    assert isinstance(client.client, Qdrant)
    assert client.client.content_payload_key == "content"
    assert client.client.metadata_payload_key == "extra"


async def test_qdrant_search_raises_PayloadNotFoundException(
    monkeypatch: MonkeyPatch,
    embeddings: Embeddings,
) -> None:
    def mock_init(self, embeddings: Any):
        self.embeddings = embeddings
        self.client = MagicMock()

    monkeypatch.setattr(
        "rasa.core.information_retrieval.qdrant.Qdrant_Store.__init__",
        mock_init,
    )
    qdrant_store = Qdrant_Store(embeddings=embeddings)

    base_exception_msg = "An error occurred"
    monkeypatch.setattr(
        qdrant_store.client,
        "asimilarity_search",
        AsyncMock(
            side_effect=ValidationError.from_exception_data(
                base_exception_msg, line_errors=[]
            )
        ),
    )

    with pytest.raises(PayloadNotFoundException) as e:
        await qdrant_store.search("test", {})

    assert issubclass(e.type, InformationRetrievalException)
    assert (
        "An error occurred while searching for documents: "
        "Payload not found in the Qdrant response. Please "
        "make sure the `content_payload_key`and "
        "`metadata_payload_key` are correct in the Qdrant "
        f"configuration. Error: 0 validation errors for {base_exception_msg}\n"
    ) in str(e.value)


async def test_qdrant_search_raises_QdrantInformationRetrievalException(
    monkeypatch: MonkeyPatch,
    embeddings: Embeddings,
) -> None:
    def mock_init(self, embeddings: Any):
        self.embeddings = embeddings
        self.client = MagicMock()

    monkeypatch.setattr(
        "rasa.core.information_retrieval.qdrant.Qdrant_Store.__init__",
        mock_init,
    )
    qdrant_store = Qdrant_Store(embeddings=embeddings)

    base_exception_msg = "An error occurred"
    monkeypatch.setattr(
        qdrant_store.client,
        "asimilarity_search",
        AsyncMock(side_effect=Exception(base_exception_msg)),
    )

    with pytest.raises(QdrantInformationRetrievalException) as e:
        await qdrant_store.search("test", {})

    assert issubclass(e.type, InformationRetrievalException)
    assert (
        "An error occurred while searching for documents: "
        "Failed to search the Qdrant vector store. "
        f"Encountered error: {base_exception_msg}"
    ) in str(e.value)
