from unittest.mock import Mock

import pytest
from langchain.schema.embeddings import Embeddings
from pytest import MonkeyPatch

from rasa.core.information_retrieval.faiss import FAISS_Store
from rasa.core.information_retrieval.information_retrieval import (
    InformationRetrievalException,
)


def test_faiss_search_raises_custom_exception(
    monkeypatch: MonkeyPatch,
    embeddings: Embeddings,
) -> None:
    def mock_init(self, *args, **kwargs):
        self.index = Mock()

    monkeypatch.setattr(
        "rasa.core.information_retrieval.faiss.FAISS_Store.__init__",
        mock_init,
    )
    faiss_store = FAISS_Store(
        embeddings=embeddings, index_path="test", docs_folder="test"
    )

    base_exception_msg = "An error occurred"

    monkeypatch.setattr(
        faiss_store.index,
        "as_retriever",
        Mock(side_effect=Exception(base_exception_msg)),
    )

    with pytest.raises(InformationRetrievalException) as e:
        faiss_store.search("test")

    assert (
        f"An error occurred while searching for documents: {base_exception_msg}"
        in str(e.value)
    )
