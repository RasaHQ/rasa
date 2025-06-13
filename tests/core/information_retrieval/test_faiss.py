from unittest.mock import Mock, patch

import pytest
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from pytest import MonkeyPatch

from rasa.core.information_retrieval import InformationRetrievalException
from rasa.core.information_retrieval.faiss import FAISS_Store


async def test_faiss_search_raises_custom_exception(
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
        await faiss_store.search("test", {})

    assert (
        f"An error occurred while searching for documents: {base_exception_msg}"
        in str(e.value)
    )


@patch("rasa.core.information_retrieval.faiss.FAISS.from_texts")
@patch("rasa.core.information_retrieval.faiss.FAISS_Store.load_documents")
@patch("rasa.core.information_retrieval.faiss.persist_faiss_vector_store")
def test_faiss_store_initialization_without_faq_parsing(
    mock_persist_faiss_vector_store: Mock,
    mock_load_documents: Mock,
    mock_faiss_from_texts: Mock,
):
    # Given
    mock_embedding = Mock()
    mock_load_documents.return_value = [
        Document(
            page_content="Finley is a helpful assistant of the FinX platform.",
            metadata={},
        )
    ]

    # When
    FAISS_Store(
        embeddings=mock_embedding,
        index_path="/fake/index/path",
        docs_folder="/fake/docs/folder",
        create_index=True,
        # Disable parsing the document as FAQ pair
        parse_as_faq_pairs=False,
    )

    # Then
    mock_faiss_from_texts.assert_called_once()
    args, kwargs = mock_faiss_from_texts.call_args

    texts = args[0]
    metadatas = kwargs.get("metadatas")

    assert texts == ["Finley is a helpful assistant of the FinX platform."]
    assert metadatas == [{}]


@patch("rasa.core.information_retrieval.faiss.FAISS.from_texts")
@patch("rasa.core.information_retrieval.faiss.FAISS_Store.load_documents")
@patch("rasa.core.information_retrieval.faiss.persist_faiss_vector_store")
def test_faiss_store_initialization_with_faq_parsing(
    mock_persist_faiss_vector_store: Mock,
    mock_load_documents: Mock,
    mock_faiss_from_texts: Mock,
):
    # Given
    mock_embedding = Mock()
    mock_load_documents.return_value = [
        Document(
            page_content=(
                "Q: What is Finley?\nA: Finley is assistant.\n\n"
                "Q: What is FinX?\nA: FinX is a platform."
            ),
            metadata={},
        )
    ]

    # When
    FAISS_Store(
        embeddings=mock_embedding,
        index_path="/fake/index/path",
        docs_folder="/fake/docs/folder",
        create_index=True,
        # Enable parsing the document as FAQ pair
        parse_as_faq_pairs=True,
    )

    # Then
    mock_faiss_from_texts.assert_called_once()
    args, kwargs = mock_faiss_from_texts.call_args

    texts = args[0]
    metadatas = kwargs.get("metadatas")

    assert texts == ["What is Finley?", "What is FinX?"]
    assert metadatas == [
        {
            "title": "what_is_finley",
            "type": "faq",
            "answer": "Finley is assistant.",
        },
        {
            "title": "what_is_finx",
            "type": "faq",
            "answer": "FinX is a platform.",
        },
    ]
