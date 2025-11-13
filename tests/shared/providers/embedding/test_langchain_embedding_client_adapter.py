from typing import Any, Dict, Iterator, List

import pytest

from rasa.shared.providers.embedding._langchain_embedding_client_adapter import (
    _LangchainEmbeddingClientAdapter,
    embedding_metadata_context,
    get_embedding_metadata,
    set_embedding_metadata,
)
from rasa.shared.providers.embedding.embedding_response import EmbeddingResponse


class MockEmbeddingClient:
    """Mock embedding client for testing."""

    def __init__(self) -> None:
        self.embed_calls: List[Dict[str, Any]] = []
        self.aembed_calls: List[Dict[str, Any]] = []

    def embed(self, documents: list[str], **kwargs: Any) -> EmbeddingResponse:
        self.embed_calls.append({"documents": documents, "kwargs": kwargs})
        return EmbeddingResponse(
            data=[[0.1, 0.2, 0.3] for _ in documents],
            model="test-model",
        )

    async def aembed(self, documents: list[str], **kwargs: Any) -> EmbeddingResponse:
        self.aembed_calls.append({"documents": documents, "kwargs": kwargs})
        return EmbeddingResponse(
            data=[[0.1, 0.2, 0.3] for _ in documents],
            model="test-model",
        )


class TestEmbeddingMetadataContext:
    """Tests for embedding metadata context functions."""

    @pytest.fixture(autouse=True)
    def clear_metadata(self) -> Iterator[None]:
        """Clear embedding metadata before each test to ensure clean state."""
        set_embedding_metadata(None)
        yield
        # Also clear after test to prevent leakage
        set_embedding_metadata(None)

    def test_get_embedding_metadata_returns_none_by_default(self) -> None:
        """Test that get_embedding_metadata returns None when no metadata is set."""
        assert get_embedding_metadata() is None

    def test_set_and_get_embedding_metadata(self) -> None:
        """Test setting and getting embedding metadata."""
        metadata = {"session_id": "user123", "component": "test"}
        set_embedding_metadata(metadata)
        assert get_embedding_metadata() == metadata

    def test_set_embedding_metadata_to_none_clears_metadata(self) -> None:
        """Test that setting metadata to None clears it."""
        metadata = {"session_id": "user123"}
        set_embedding_metadata(metadata)
        assert get_embedding_metadata() == metadata

        set_embedding_metadata(None)
        assert get_embedding_metadata() is None

    def test_embedding_metadata_context_sets_and_clears_metadata(self) -> None:
        """Test that the context manager sets and clears metadata correctly."""
        metadata = {"session_id": "user123", "component": "test"}

        # Before context
        assert get_embedding_metadata() is None

        # Inside context
        with embedding_metadata_context(metadata):
            assert get_embedding_metadata() == metadata

        # After context
        assert get_embedding_metadata() is None

    def test_embedding_metadata_context_cleans_up_on_exception(self) -> None:
        """Test that the context manager cleans up metadata even when an exception
        occurs."""
        metadata = {"session_id": "user123"}

        with pytest.raises(ValueError):
            with embedding_metadata_context(metadata):
                assert get_embedding_metadata() == metadata
                raise ValueError("Test exception")

        # Metadata should be cleared even after exception
        assert get_embedding_metadata() is None

    def test_embedding_metadata_context_nested(self) -> None:
        """Test nested embedding metadata contexts."""
        outer_metadata = {"session_id": "user123", "level": "outer"}
        inner_metadata = {"session_id": "user456", "level": "inner"}

        with embedding_metadata_context(outer_metadata):
            assert get_embedding_metadata() == outer_metadata

            with embedding_metadata_context(inner_metadata):
                assert get_embedding_metadata() == inner_metadata

            # Should restore to outer metadata
            assert get_embedding_metadata() == outer_metadata

        # Should be cleared after all contexts
        assert get_embedding_metadata() is None

    def test_embedding_metadata_context_overwrites_previous(self) -> None:
        """Test that setting metadata directly and then using context works
        correctly."""
        initial_metadata = {"initial": "value"}
        set_embedding_metadata(initial_metadata)

        context_metadata = {"context": "value"}
        with embedding_metadata_context(context_metadata):
            assert get_embedding_metadata() == context_metadata

        # Should restore to initial metadata after context
        assert get_embedding_metadata() == initial_metadata


class TestLangchainEmbeddingClientAdapterMetadata:
    """Tests for _LangchainEmbeddingClientAdapter with metadata."""

    @pytest.fixture(autouse=True)
    def clear_metadata(self) -> Iterator[None]:
        """Clear embedding metadata before each test to ensure clean state."""
        set_embedding_metadata(None)
        yield
        # Also clear after test to prevent leakage
        set_embedding_metadata(None)

    @pytest.fixture
    def mock_client(self) -> MockEmbeddingClient:
        """Create a mock embedding client."""
        return MockEmbeddingClient()

    @pytest.fixture
    def adapter(
        self, mock_client: MockEmbeddingClient
    ) -> _LangchainEmbeddingClientAdapter:
        """Create an adapter with a mock client."""
        # MockEmbeddingClient doesn't implement the full EmbeddingClient protocol,
        # but it has the methods we need for testing
        return _LangchainEmbeddingClientAdapter(mock_client)  # type: ignore[arg-type]

    def test_embed_documents_without_metadata(
        self,
        adapter: _LangchainEmbeddingClientAdapter,
        mock_client: MockEmbeddingClient,
    ) -> None:
        """Test embed_documents without metadata in context."""
        texts = ["text1", "text2"]
        result = adapter.embed_documents(texts)

        assert len(result) == 2
        assert len(mock_client.embed_calls) == 1
        assert mock_client.embed_calls[0]["documents"] == texts
        assert "metadata" not in mock_client.embed_calls[0]["kwargs"]

    def test_embed_documents_with_metadata(
        self,
        adapter: _LangchainEmbeddingClientAdapter,
        mock_client: MockEmbeddingClient,
    ) -> None:
        """Test embed_documents with metadata in context."""
        texts = ["text1", "text2"]
        metadata = {"session_id": "user123", "component": "test"}

        with embedding_metadata_context(metadata):
            result = adapter.embed_documents(texts)

        assert len(result) == 2
        assert len(mock_client.embed_calls) == 1
        assert mock_client.embed_calls[0]["documents"] == texts
        assert mock_client.embed_calls[0]["kwargs"]["metadata"] == metadata

    def test_embed_query_without_metadata(
        self,
        adapter: _LangchainEmbeddingClientAdapter,
        mock_client: MockEmbeddingClient,
    ) -> None:
        """Test embed_query without metadata in context."""
        text = "query text"
        result = adapter.embed_query(text)

        assert len(result) == 3
        assert len(mock_client.embed_calls) == 1
        assert mock_client.embed_calls[0]["documents"] == [text]
        assert "metadata" not in mock_client.embed_calls[0]["kwargs"]

    def test_embed_query_with_metadata(
        self,
        adapter: _LangchainEmbeddingClientAdapter,
        mock_client: MockEmbeddingClient,
    ) -> None:
        """Test embed_query with metadata in context."""
        text = "query text"
        metadata = {"session_id": "user123", "component": "test"}

        with embedding_metadata_context(metadata):
            result = adapter.embed_query(text)

        assert len(result) == 3
        assert len(mock_client.embed_calls) == 1
        assert mock_client.embed_calls[0]["documents"] == [text]
        assert mock_client.embed_calls[0]["kwargs"]["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_aembed_documents_without_metadata(
        self,
        adapter: _LangchainEmbeddingClientAdapter,
        mock_client: MockEmbeddingClient,
    ) -> None:
        """Test aembed_documents without metadata in context."""
        texts = ["text1", "text2"]
        result = await adapter.aembed_documents(texts)

        assert len(result) == 2
        assert len(mock_client.aembed_calls) == 1
        assert mock_client.aembed_calls[0]["documents"] == texts
        assert "metadata" not in mock_client.aembed_calls[0]["kwargs"]

    @pytest.mark.asyncio
    async def test_aembed_documents_with_metadata(
        self,
        adapter: _LangchainEmbeddingClientAdapter,
        mock_client: MockEmbeddingClient,
    ) -> None:
        """Test aembed_documents with metadata in context."""
        texts = ["text1", "text2"]
        metadata = {"session_id": "user123", "component": "test"}

        with embedding_metadata_context(metadata):
            result = await adapter.aembed_documents(texts)

        assert len(result) == 2
        assert len(mock_client.aembed_calls) == 1
        assert mock_client.aembed_calls[0]["documents"] == texts
        assert mock_client.aembed_calls[0]["kwargs"]["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_aembed_query_without_metadata(
        self,
        adapter: _LangchainEmbeddingClientAdapter,
        mock_client: MockEmbeddingClient,
    ) -> None:
        """Test aembed_query without metadata in context."""
        text = "query text"
        result = await adapter.aembed_query(text)

        assert len(result) == 3
        assert len(mock_client.aembed_calls) == 1
        assert mock_client.aembed_calls[0]["documents"] == [text]
        assert "metadata" not in mock_client.aembed_calls[0]["kwargs"]

    @pytest.mark.asyncio
    async def test_aembed_query_with_metadata(
        self,
        adapter: _LangchainEmbeddingClientAdapter,
        mock_client: MockEmbeddingClient,
    ) -> None:
        """Test aembed_query with metadata in context."""
        text = "query text"
        metadata = {"session_id": "user123", "component": "test"}

        with embedding_metadata_context(metadata):
            result = await adapter.aembed_query(text)

        assert len(result) == 3
        assert len(mock_client.aembed_calls) == 1
        assert mock_client.aembed_calls[0]["documents"] == [text]
        assert mock_client.aembed_calls[0]["kwargs"]["metadata"] == metadata

    def test_metadata_isolation_between_calls(
        self,
        adapter: _LangchainEmbeddingClientAdapter,
        mock_client: MockEmbeddingClient,
    ) -> None:
        """Test that metadata is isolated between different embedding calls."""
        texts = ["text1", "text2"]

        # First call with metadata
        metadata1 = {"session_id": "user123"}
        with embedding_metadata_context(metadata1):
            adapter.embed_documents(texts)

        # Second call without metadata
        adapter.embed_documents(texts)

        # Third call with different metadata
        metadata2 = {"session_id": "user456"}
        with embedding_metadata_context(metadata2):
            adapter.embed_documents(texts)

        assert len(mock_client.embed_calls) == 3
        assert mock_client.embed_calls[0]["kwargs"]["metadata"] == metadata1
        assert "metadata" not in mock_client.embed_calls[1]["kwargs"]
        assert mock_client.embed_calls[2]["kwargs"]["metadata"] == metadata2

    @pytest.mark.asyncio
    async def test_metadata_passed_to_both_sync_and_async(
        self,
        adapter: _LangchainEmbeddingClientAdapter,
        mock_client: MockEmbeddingClient,
    ) -> None:
        """Test that metadata is correctly passed to both sync and async methods."""
        metadata = {"session_id": "user123", "component": "test"}

        with embedding_metadata_context(metadata):
            # Sync call
            adapter.embed_documents(["text1"])

            # Async call
            await adapter.aembed_documents(["text2"])

        assert len(mock_client.embed_calls) == 1
        assert mock_client.embed_calls[0]["kwargs"]["metadata"] == metadata

        assert len(mock_client.aembed_calls) == 1
        assert mock_client.aembed_calls[0]["kwargs"]["metadata"] == metadata
