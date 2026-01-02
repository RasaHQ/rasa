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


@patch("rasa.core.information_retrieval.faiss.FAISS.from_texts")
@patch("rasa.core.information_retrieval.faiss.FAISS_Store.load_documents")
@patch("rasa.core.information_retrieval.faiss.persist_faiss_vector_store")
def test_create_document_index_no_batching_when_under_batch_size(
    mock_persist_faiss_vector_store: Mock,
    mock_load_documents: Mock,
    mock_faiss_from_texts: Mock,
):
    """Test that documents under batch size (50) don't trigger add_texts."""
    # Given - 10 documents (under batch size of 50)
    mock_embedding = Mock()
    documents = [
        Document(page_content=f"Document {i}", metadata={"id": i}) for i in range(10)
    ]
    mock_load_documents.return_value = documents

    mock_index = Mock()
    mock_faiss_from_texts.return_value = mock_index

    # When
    FAISS_Store(
        embeddings=mock_embedding,
        index_path="/fake/index/path",
        docs_folder="/fake/docs/folder",
        create_index=True,
        parse_as_faq_pairs=False,
    )

    # Then - from_texts called once, add_texts never called
    mock_faiss_from_texts.assert_called_once()
    mock_index.add_texts.assert_not_called()

    # Verify correct texts were passed
    args, kwargs = mock_faiss_from_texts.call_args
    texts = args[0]
    metadatas = kwargs.get("metadatas")
    assert len(texts) == 10
    assert texts == [f"Document {i}" for i in range(10)]
    assert metadatas == [{"id": i} for i in range(10)]


@patch("rasa.core.information_retrieval.faiss.FAISS.from_texts")
@patch("rasa.core.information_retrieval.faiss.FAISS_Store.load_documents")
@patch("rasa.core.information_retrieval.faiss.persist_faiss_vector_store")
def test_create_document_index_no_batching_at_exact_batch_size(
    mock_persist_faiss_vector_store: Mock,
    mock_load_documents: Mock,
    mock_faiss_from_texts: Mock,
):
    """Test that exactly 50 documents don't trigger add_texts."""
    # Given - exactly 50 documents (batch size)
    mock_embedding = Mock()
    documents = [
        Document(page_content=f"Document {i}", metadata={"id": i}) for i in range(50)
    ]
    mock_load_documents.return_value = documents

    mock_index = Mock()
    mock_faiss_from_texts.return_value = mock_index

    # When
    FAISS_Store(
        embeddings=mock_embedding,
        index_path="/fake/index/path",
        docs_folder="/fake/docs/folder",
        create_index=True,
        parse_as_faq_pairs=False,
    )

    # Then - from_texts called once with all 50 docs, add_texts never called
    mock_faiss_from_texts.assert_called_once()
    mock_index.add_texts.assert_not_called()

    args, kwargs = mock_faiss_from_texts.call_args
    texts = args[0]
    assert len(texts) == 50


@patch("rasa.core.information_retrieval.faiss.FAISS.from_texts")
@patch("rasa.core.information_retrieval.faiss.FAISS_Store.load_documents")
@patch("rasa.core.information_retrieval.faiss.persist_faiss_vector_store")
def test_create_document_index_batching_with_two_batches(
    mock_persist_faiss_vector_store: Mock,
    mock_load_documents: Mock,
    mock_faiss_from_texts: Mock,
):
    """Test that 75 documents trigger one add_texts call for the second batch."""
    # Given - 75 documents (1.5 batches)
    mock_embedding = Mock()
    documents = [
        Document(page_content=f"Document {i}", metadata={"id": i}) for i in range(75)
    ]
    mock_load_documents.return_value = documents

    mock_index = Mock()
    mock_faiss_from_texts.return_value = mock_index

    # When
    FAISS_Store(
        embeddings=mock_embedding,
        index_path="/fake/index/path",
        docs_folder="/fake/docs/folder",
        create_index=True,
        parse_as_faq_pairs=False,
    )

    # Then - from_texts called once with first 50, add_texts called once with
    # remaining 25
    mock_faiss_from_texts.assert_called_once()

    # Check first batch (from_texts)
    args, kwargs = mock_faiss_from_texts.call_args
    first_batch_texts = args[0]
    first_batch_metadatas = kwargs.get("metadatas")
    assert len(first_batch_texts) == 50
    assert first_batch_texts == [f"Document {i}" for i in range(50)]
    assert first_batch_metadatas == [{"id": i} for i in range(50)]

    # Check second batch (add_texts)
    mock_index.add_texts.assert_called_once()
    args, kwargs = mock_index.add_texts.call_args
    second_batch_texts = args[0]
    second_batch_metadatas = kwargs.get("metadatas")
    assert len(second_batch_texts) == 25
    assert second_batch_texts == [f"Document {i}" for i in range(50, 75)]
    assert second_batch_metadatas == [{"id": i} for i in range(50, 75)]


@patch("rasa.core.information_retrieval.faiss.FAISS.from_texts")
@patch("rasa.core.information_retrieval.faiss.FAISS_Store.load_documents")
@patch("rasa.core.information_retrieval.faiss.persist_faiss_vector_store")
def test_create_document_index_batching_with_multiple_batches(
    mock_persist_faiss_vector_store: Mock,
    mock_load_documents: Mock,
    mock_faiss_from_texts: Mock,
):
    """Test that 125 documents result in 3 batches (50 + 50 + 25)."""
    # Given - 125 documents (2.5 batches)
    mock_embedding = Mock()
    documents = [
        Document(page_content=f"Document {i}", metadata={"id": i}) for i in range(125)
    ]
    mock_load_documents.return_value = documents

    mock_index = Mock()
    mock_faiss_from_texts.return_value = mock_index

    # When
    FAISS_Store(
        embeddings=mock_embedding,
        index_path="/fake/index/path",
        docs_folder="/fake/docs/folder",
        create_index=True,
        parse_as_faq_pairs=False,
    )

    # Then - from_texts once, add_texts twice
    mock_faiss_from_texts.assert_called_once()
    assert mock_index.add_texts.call_count == 2

    # Check first batch (from_texts) - documents 0-49
    args, kwargs = mock_faiss_from_texts.call_args
    first_batch_texts = args[0]
    assert len(first_batch_texts) == 50
    assert first_batch_texts == [f"Document {i}" for i in range(50)]

    # Check second batch (first add_texts) - documents 50-99
    second_call_args, second_call_kwargs = mock_index.add_texts.call_args_list[0]
    second_batch_texts = second_call_args[0]
    second_batch_metadatas = second_call_kwargs.get("metadatas")
    assert len(second_batch_texts) == 50
    assert second_batch_texts == [f"Document {i}" for i in range(50, 100)]
    assert second_batch_metadatas == [{"id": i} for i in range(50, 100)]

    # Check third batch (second add_texts) - documents 100-124
    third_call_args, third_call_kwargs = mock_index.add_texts.call_args_list[1]
    third_batch_texts = third_call_args[0]
    third_batch_metadatas = third_call_kwargs.get("metadatas")
    assert len(third_batch_texts) == 25
    assert third_batch_texts == [f"Document {i}" for i in range(100, 125)]
    assert third_batch_metadatas == [{"id": i} for i in range(100, 125)]


@patch("rasa.core.information_retrieval.faiss.FAISS.from_texts")
@patch("rasa.core.information_retrieval.faiss.FAISS_Store.load_documents")
@patch("rasa.core.information_retrieval.faiss.persist_faiss_vector_store")
def test_create_document_index_batching_exact_multiple_of_batch_size(
    mock_persist_faiss_vector_store: Mock,
    mock_load_documents: Mock,
    mock_faiss_from_texts: Mock,
):
    """Test that 100 documents result in exactly 2 full batches."""
    # Given - 100 documents (exactly 2 batches)
    mock_embedding = Mock()
    documents = [
        Document(page_content=f"Document {i}", metadata={"id": i}) for i in range(100)
    ]
    mock_load_documents.return_value = documents

    mock_index = Mock()
    mock_faiss_from_texts.return_value = mock_index

    # When
    FAISS_Store(
        embeddings=mock_embedding,
        index_path="/fake/index/path",
        docs_folder="/fake/docs/folder",
        create_index=True,
        parse_as_faq_pairs=False,
    )

    # Then - from_texts once with 50, add_texts once with 50
    mock_faiss_from_texts.assert_called_once()
    mock_index.add_texts.assert_called_once()

    # Check first batch
    args, kwargs = mock_faiss_from_texts.call_args
    assert len(args[0]) == 50

    # Check second batch
    add_args, add_kwargs = mock_index.add_texts.call_args
    assert len(add_args[0]) == 50
    assert add_args[0] == [f"Document {i}" for i in range(50, 100)]


@patch("rasa.core.information_retrieval.faiss.FAISS.from_texts")
@patch("rasa.core.information_retrieval.faiss.FAISS_Store.load_documents")
@patch("rasa.core.information_retrieval.faiss.persist_faiss_vector_store")
def test_create_document_index_warns_on_large_document_size(
    mock_persist_faiss_vector_store: Mock,
    mock_load_documents: Mock,
    mock_faiss_from_texts: Mock,
    capsys: pytest.CaptureFixture,
):
    """Test that a warning is logged when document size exceeds 2GB."""
    # Given - Create documents that exceed 2GB total
    mock_embedding = Mock()
    # Create a single large document (we'll mock the size calculation)
    large_content = "x" * 1000  # 1KB of content
    documents = [Document(page_content=large_content, metadata={})]
    mock_load_documents.return_value = documents

    mock_index = Mock()
    mock_faiss_from_texts.return_value = mock_index

    # Patch the size calculation to simulate > 2GB
    with patch(
        "rasa.core.information_retrieval.faiss.sum",
        return_value=3 * 1024 * 1024 * 1024,  # 3GB
    ):
        # When
        FAISS_Store(
            embeddings=mock_embedding,
            index_path="/fake/index/path",
            docs_folder="/fake/docs/folder",
            create_index=True,
            parse_as_faq_pairs=False,
        )

    # Then - Warning should be logged (structlog outputs to stdout)
    captured = capsys.readouterr()
    assert "Document size exceeds 2GB" in captured.out
    assert "total_size_gb=3.0" in captured.out


@patch("rasa.core.information_retrieval.faiss.FAISS.from_texts")
@patch("rasa.core.information_retrieval.faiss.FAISS_Store.load_documents")
@patch("rasa.core.information_retrieval.faiss.persist_faiss_vector_store")
def test_create_document_index_no_warning_under_size_limit(
    mock_persist_faiss_vector_store: Mock,
    mock_load_documents: Mock,
    mock_faiss_from_texts: Mock,
    capsys: pytest.CaptureFixture,
):
    """Test that no warning is logged when document size is under 2GB."""
    # Given - Small documents under 2GB
    mock_embedding = Mock()
    documents = [
        Document(page_content="Small document content", metadata={}) for _ in range(10)
    ]
    mock_load_documents.return_value = documents

    mock_index = Mock()
    mock_faiss_from_texts.return_value = mock_index

    # When
    FAISS_Store(
        embeddings=mock_embedding,
        index_path="/fake/index/path",
        docs_folder="/fake/docs/folder",
        create_index=True,
        parse_as_faq_pairs=False,
    )

    # Then - No warning about large document size (structlog outputs to stdout)
    captured = capsys.readouterr()
    assert "Document size exceeds 2GB" not in captured.out
