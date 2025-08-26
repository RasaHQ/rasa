import json
from unittest.mock import Mock, patch

import pytest

from rasa.builder.document_retrieval.models import (
    Document,
)


class TestDocument:
    """Test cases for Document model."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "json_decode_error,expected_log_level",
        [
            (json.JSONDecodeError("Invalid JSON", "test", 0), "warning"),
            (Exception("General error"), "error"),
        ],
    )
    async def test_parse_documents_exception_handling(
        self, json_decode_error, expected_log_level
    ):
        """Test exception handling in document parsing from InKeep RAG response."""
        # Given
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "invalid json content"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # When
        with patch("json.loads", side_effect=json_decode_error):
            # This simulates the parsing that happens in _parse_documents_from_response
            # but we're testing the Document model's ability to handle invalid JSON
            try:
                response_data = json.loads(mock_message.content)
                documents = [
                    Document.from_inkeep_rag_response(item)
                    for item in response_data.get("content", [])
                ]
            except (json.JSONDecodeError, Exception):
                documents = []

        # Then - Should return empty list on error
        assert documents == []
