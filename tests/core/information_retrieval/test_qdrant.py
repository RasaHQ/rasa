import pytest
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.qdrant import Qdrant
from pytest import MonkeyPatch
from rasa.utils.endpoints import EndpointConfig

from rasa.core.information_retrieval.qdrant import Qdrant_Store


@pytest.fixture
def embeddings(monkeypatch: MonkeyPatch) -> Embeddings:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    return OpenAIEmbeddings()


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


# TODO: Uncomment this test when using pydantic 2.x
# pydantic 1.x does not support ValidationError.from_exception_data()
# def test_qdrant_store_validation_error(embeddings):
#     qdrant = Qdrant_Store(embeddings)
#     qdrant.connect(
#         EndpointConfig(
#             location=":memory:",
#             collection_name="test",
#             content_payload_key="content",
#             metadata_payload_key="extra",
#         )
#     )
#     # Mock the similarity_search method to raise an exception
#     with patch.object(
#         qdrant.client,
#         "similarity_search",
#         side_effect=ValidationError.from_exception_data("test", line_errors=[])
#     ):
#          with pytest.raises(PayloadNotFoundException, match="Payload not found"):
#              qdrant.search("test")
