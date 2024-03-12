from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.qdrant import Qdrant

from rasa.core.information_retrieval.information_retrieval import (
    InformationRetrievalException,
)
from rasa.utils.endpoints import EndpointConfig
from rasa.core.information_retrieval.qdrant import (
    PayloadNotFoundException,
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


# TODO: Update this test to use ValidationError.from_exception_data() when using pydantic 2.x  # noqa: E501
def test_qdrant_search_raises_custom_exception(
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

    def mock_str(self):
        return base_exception_msg

    monkeypatch.setattr(ValidationError, "__str__", mock_str)

    monkeypatch.setattr(
        qdrant_store.client,
        "similarity_search",
        Mock(side_effect=ValidationError([], Mock())),
    )

    with pytest.raises(PayloadNotFoundException) as e:
        qdrant_store.search("test")

    assert issubclass(e.type, InformationRetrievalException)
    assert (
        f"An error occurred while searching for documents: "
        f"Payload not found in the Qdrant response. "
        f"Please make sure the `content_payload_key`and `metadata_payload_key` "
        f"are correct in the Qdrant configuration. "
        f"Error: {base_exception_msg}"
    ) in str(e.value)
