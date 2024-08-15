from typing import Any

from pytest import MonkeyPatch

from rasa.core.information_retrieval import InformationRetrievalException
from rasa.core.information_retrieval.milvus import Milvus_Store
from langchain.schema.embeddings import Embeddings
from langchain.schema import Document
from unittest.mock import AsyncMock, patch
import pytest
from unittest.mock import MagicMock
from langchain_community.vectorstores.milvus import Milvus


def test_milvus_store(embeddings: Embeddings):
    client = Milvus_Store(embeddings)
    assert client is not None
    assert isinstance(client, Milvus_Store)
    assert client.embeddings is not None
    assert isinstance(client.embeddings, Embeddings)


@pytest.mark.parametrize(
    "threshold, expected_count, expected_id",
    [
        (0.5, 5, "doc1"),
        (0.9, 1, "doc1"),
        (0.0, 9, "doc1"),
        (1.0, 0, None),
    ],
)
async def test_milvus_store_search(
    embeddings: Embeddings, threshold: float, expected_count: int, expected_id: str
):
    milvus_store = Milvus_Store(embeddings)
    milvus_store.client = MagicMock(spec=Milvus)

    search_results = [
        (Document(page_content="hello", metadata={"id": "doc1"}), 0.9),
        (Document(page_content="world", metadata={"id": "doc2"}), 0.8),
        (Document(page_content="this", metadata={"id": "doc3"}), 0.7),
        (Document(page_content="is a test", metadata={"id": "doc4"}), 0.6),
        (Document(page_content="sample", metadata={"id": "doc5"}), 0.5),
        (Document(page_content="document", metadata={"id": "doc6"}), 0.4),
        (Document(page_content="for", metadata={"id": "doc7"}), 0.3),
        (Document(page_content="testing", metadata={"id": "doc8"}), 0.2),
        (Document(page_content="purposes", metadata={"id": "doc9"}), 0.1),
    ]
    # patch the client to return a list of tuples
    with patch.object(
        milvus_store.client,
        "asimilarity_search_with_score",
        return_value=search_results,
    ):
        hits = await milvus_store.search("test", {}, threshold=threshold)
        assert len(hits.results) == expected_count
        if hits.results:
            assert hits.results[0].metadata["id"] == expected_id


async def test_milvus_search_raises_custom_exception(
    monkeypatch: MonkeyPatch,
    embeddings: Embeddings,
) -> None:
    def mock_init(self, embeddings: Any):
        self.embeddings = embeddings
        self.client = MagicMock()

    monkeypatch.setattr(
        "rasa.core.information_retrieval.milvus.Milvus_Store.__init__",
        mock_init,
    )
    milvus_store = Milvus_Store(embeddings=embeddings)

    base_exception_msg = "An error occurred"

    monkeypatch.setattr(
        milvus_store.client,
        "asimilarity_search_with_score",
        AsyncMock(side_effect=Exception(base_exception_msg)),
    )

    with pytest.raises(InformationRetrievalException) as e:
        await milvus_store.search("test", {})

    assert (
        f"An error occurred while searching for documents: {base_exception_msg}"
        in str(e.value)
    )
