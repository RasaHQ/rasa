"""Unit tests for CopilotExecutor."""

from typing import AsyncIterator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.builder.copilot.models import (
    CopilotContext,
    CopilotGenerationContext,
    GeneratedContent,
    ReferenceEntry,
    ReferenceSection,
    ResponseCategory,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.evaluator.shared.copilot_executor import (
    run_copilot_with_response_handler,
)


class TestRunCopilotWithResponseHandler:
    """Tests for run_copilot_with_response_handler function."""

    @pytest.mark.asyncio
    @patch("rasa.builder.evaluator.shared.copilot_executor.get_copilot_class")
    @pytest.mark.parametrize(
        "response_chunks,"
        "response_category,"
        "has_relevant_documents,"
        "has_reference_section,"
        "expected_complete_response,"
        "expected_category,"
        "should_extract_references",
        [
            # Test case 1: Single chunk, no references
            (
                ["Hello"],
                ResponseCategory.COPILOT,
                False,
                False,
                "Hello",
                ResponseCategory.COPILOT,
                False,
            ),
            # Test case 2: Multiple chunks, no references
            (
                ["Hello", " ", "world", "!"],
                ResponseCategory.COPILOT,
                False,
                False,
                "Hello world!",
                ResponseCategory.COPILOT,
                False,
            ),
            # Test case 3: Empty chunks
            (
                [],
                None,
                False,
                False,
                None,
                None,
                False,
            ),
            # Test case 4: With relevant documents and reference section
            (
                ["Response with references"],
                ResponseCategory.COPILOT,
                True,
                True,
                "Response with references",
                ResponseCategory.COPILOT,
                True,
            ),
            # Test case 5: With relevant documents but no reference section
            (
                ["Response without references"],
                ResponseCategory.COPILOT,
                True,
                False,
                "Response without references",
                ResponseCategory.COPILOT,
                True,
            ),
            # Test case 6: No relevant documents (should not extract)
            (
                ["Response"],
                ResponseCategory.COPILOT,
                False,
                False,
                "Response",
                ResponseCategory.COPILOT,
                False,
            ),
        ],
    )
    async def test_run_copilot_with_response_handler(
        self,
        mock_copilot_class: MagicMock,
        response_chunks: List[str],
        response_category: Optional[ResponseCategory],
        has_relevant_documents: bool,
        has_reference_section: bool,
        expected_complete_response: Optional[str],
        expected_category: Optional[ResponseCategory],
        should_extract_references: bool,
    ) -> None:
        """Test execution with various scenarios."""
        # Given
        context = CopilotContext(
            tracker_context=None,
            assistant_logs="",
            assistant_files={},
            copilot_chat_history=[],
        )

        # Create documents if needed
        documents = (
            [
                Document(
                    content="Test content 1",
                    url="https://test.com/doc1",
                    title="Document 1",
                ),
                Document(
                    content="Test content 2",
                    url="https://test.com/doc2",
                    title="Document 2",
                ),
            ]
            if has_relevant_documents
            else []
        )

        # Create mock generation context
        generation_context = CopilotGenerationContext(
            relevant_documents=documents,
            system_message={},
            chat_history=[],
            tracker_event_attachments=[],
            last_user_message=None,
        )

        # Create mock reference section if needed
        reference_section = (
            ReferenceSection(
                references=[
                    ReferenceEntry(
                        index=1, title="Document 1", url="https://test.com/doc1"
                    ),
                    ReferenceEntry(
                        index=2, title="Document 2", url="https://test.com/doc2"
                    ),
                ]
            )
            if has_reference_section
            else None
        )

        # Create mock response handler that yields chunks
        mock_response_handler = MagicMock()

        async def mock_stream() -> AsyncIterator[GeneratedContent]:
            for chunk in response_chunks:
                yield GeneratedContent(
                    content=chunk,
                    response_category=response_category or ResponseCategory.COPILOT,
                )

        mock_response_handler.stream = mock_stream
        mock_response_handler.extract_references = MagicMock(
            return_value=reference_section
        )

        # Mock copilot instance
        mock_copilot_instance = MagicMock()
        mock_copilot_instance.generate_response = AsyncMock(
            return_value=(mock_response_handler, generation_context)
        )
        mock_copilot_class.return_value.return_value = mock_copilot_instance

        # When
        result = await run_copilot_with_response_handler(context)

        # Then
        assert result is not None
        assert result.complete_response == expected_complete_response
        assert result.response_category == expected_category
        assert result.generation_context == generation_context

        if should_extract_references:
            mock_response_handler.extract_references.assert_called_once_with(documents)
            assert result.reference_section == reference_section
        else:
            mock_response_handler.extract_references.assert_not_called()
            assert result.reference_section is None

    @pytest.mark.asyncio
    @patch("rasa.builder.evaluator.shared.copilot_executor.get_copilot_class")
    @pytest.mark.parametrize(
        "exception,expected_error",
        [
            # Test case 1: ValueError from copilot
            (ValueError("Copilot error"), ValueError),
            # Test case 2: RuntimeError from copilot
            (RuntimeError("Runtime error"), RuntimeError),
            # Test case 3: Exception from copilot
            (Exception("Generic error"), Exception),
        ],
    )
    async def test_run_copilot_with_response_handler_exception_handling(
        self,
        mock_copilot_class: MagicMock,
        exception: Exception,
        expected_error: type[Exception],
    ) -> None:
        """Test that exceptions from copilot are propagated."""
        # Given
        context = CopilotContext(
            tracker_context=None,
            assistant_logs="",
            assistant_files={},
            copilot_chat_history=[],
        )

        # Mock copilot instance to raise an exception
        mock_copilot_instance = MagicMock()
        mock_copilot_instance.generate_response = AsyncMock(side_effect=exception)
        mock_copilot_class.return_value.return_value = mock_copilot_instance

        # When/Then
        with pytest.raises(expected_error):
            await run_copilot_with_response_handler(context)
