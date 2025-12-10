"""Tests for MCP document search tool."""

from unittest.mock import AsyncMock, patch

import pytest

from rasa.builder.copilot.mcp_server.models import (
    DocumentSearchResponse,
)
from rasa.builder.copilot.mcp_server.tools.document_search import (
    search_rasa_documentation,
)
from rasa.builder.document_retrieval.models import Document


class TestSearchRasaDocumentation:
    """Test search_rasa_documentation function."""

    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self) -> None:
        """Test that search returns properly formatted DocumentSearchResponse."""
        mock_documents = [
            Document(
                title="Rasa Flows Guide",
                url="https://rasa.com/docs/flows",
                content="Flows define conversation patterns...",
            ),
            Document(
                title="Slots Documentation",
                url="https://rasa.com/docs/slots",
                content="Slots store information...",
            ),
        ]

        with patch(
            "rasa.builder.copilot.mcp_server.tools.document_search.InKeepDocumentRetrieval"
        ) as mock_retrieval_class:
            mock_retrieval = AsyncMock()
            mock_retrieval.retrieve_documents = AsyncMock(return_value=mock_documents)
            mock_retrieval_class.return_value = mock_retrieval

            result = await search_rasa_documentation("How do I create flows?")

            assert isinstance(result, DocumentSearchResponse)
            assert len(result.documents) == 2
            assert result.error is None

            # Check first document
            assert result.documents[0].index == 1
            assert result.documents[0].title == "Rasa Flows Guide"
            assert result.documents[0].url == "https://rasa.com/docs/flows"
            assert "Flows define" in result.documents[0].content

            # Check second document
            assert result.documents[1].index == 2
            assert result.documents[1].title == "Slots Documentation"

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_no_results(self) -> None:
        """Test that search returns empty list when no documents found."""
        with patch(
            "rasa.builder.copilot.mcp_server.tools.document_search.InKeepDocumentRetrieval"
        ) as mock_retrieval_class:
            mock_retrieval = AsyncMock()
            mock_retrieval.retrieve_documents = AsyncMock(return_value=[])
            mock_retrieval_class.return_value = mock_retrieval

            result = await search_rasa_documentation("very obscure query xyz123")

            assert isinstance(result, DocumentSearchResponse)
            assert len(result.documents) == 0
            assert result.error is None

    @pytest.mark.asyncio
    async def test_search_handles_exception(self) -> None:
        """Test that search handles exceptions gracefully."""
        with patch(
            "rasa.builder.copilot.mcp_server.tools.document_search.InKeepDocumentRetrieval"
        ) as mock_retrieval_class:
            mock_retrieval = AsyncMock()
            mock_retrieval.retrieve_documents = AsyncMock(
                side_effect=Exception("API connection failed")
            )
            mock_retrieval_class.return_value = mock_retrieval

            result = await search_rasa_documentation("test query")

            assert isinstance(result, DocumentSearchResponse)
            assert len(result.documents) == 0
            assert result.error is not None
            assert "Failed to search documentation" in result.error
            assert "API connection failed" in result.error

    @pytest.mark.asyncio
    async def test_search_result_indexing_starts_at_one(self) -> None:
        """Test that document indices start at 1, not 0."""
        mock_documents = [
            Document(title="Doc 1", url="url1", content="content1"),
            Document(title="Doc 2", url="url2", content="content2"),
            Document(title="Doc 3", url="url3", content="content3"),
        ]

        with patch(
            "rasa.builder.copilot.mcp_server.tools.document_search.InKeepDocumentRetrieval"
        ) as mock_retrieval_class:
            mock_retrieval = AsyncMock()
            mock_retrieval.retrieve_documents = AsyncMock(return_value=mock_documents)
            mock_retrieval_class.return_value = mock_retrieval

            result = await search_rasa_documentation("query")

            # Indices should be 1, 2, 3 (not 0, 1, 2)
            assert result.documents[0].index == 1
            assert result.documents[1].index == 2
            assert result.documents[2].index == 3
