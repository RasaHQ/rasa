import asyncio
from typing import Any
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletion

from rasa.builder.telemetry.langfuse_integration.document_retrieval_langfuse_telemetry import (  # noqa: E501
    DocumentRetrievalLangfuseTelemetry,
)


class TestDocumentRetrievalLangfuseTelemetry:
    """Test class for document retrieval telemetry functionality."""

    @patch(
        "rasa.builder.telemetry.langfuse_integration.document_retrieval_langfuse_telemetry.langfuse.get_client"
    )
    def test_trace_document_retrieval_generation_decorator(
        self, mock_get_client: Mock
    ) -> None:
        """Test the trace_document_retrieval_generation decorator."""
        # Given
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_generation = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_generation)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_client.start_as_current_generation.return_value = mock_context_manager

        # Create a mock ChatCompletion response
        mock_chat_completion = Mock(spec=ChatCompletion)
        mock_chat_completion.usage = Mock()
        mock_chat_completion.usage.prompt_tokens = 100
        mock_chat_completion.usage.completion_tokens = 50
        mock_chat_completion.usage.total_tokens = 150
        # Set up cached tokens properly
        mock_chat_completion.usage.prompt_tokens_details = Mock()
        mock_chat_completion.usage.prompt_tokens_details.cached_tokens = 0
        mock_chat_completion.model = "gpt-4"

        # Create a mock InKeepDocumentRetrieval instance
        mock_retrieval = Mock()
        mock_retrieval.__class__.__name__ = "InKeepDocumentRetrieval"

        # Create the decorator and apply it to the mock retrieval function
        decorator = (
            DocumentRetrievalLangfuseTelemetry.trace_document_retrieval_generation
        )

        async def mock_retrieval_function(
            self: Any, query: str, temperature: float, timeout: float
        ) -> ChatCompletion:
            return mock_chat_completion

        decorated_function = decorator(mock_retrieval_function)

        # When
        async def run_test() -> ChatCompletion:
            return await decorated_function(mock_retrieval, "test query", 0.7, 30.0)

        result: ChatCompletion = asyncio.run(run_test())

        # Then
        assert result == mock_chat_completion
        # Verify Langfuse client calls
        mock_client.start_as_current_generation.assert_called_once()
        assert (
            mock_client.start_as_current_generation.call_args.kwargs["input"]["query"]
            == "test query"
        )
        assert (
            mock_client.start_as_current_generation.call_args.kwargs["input"][
                "temperature"
            ]
            == 0.7
        )
        assert (
            mock_client.start_as_current_generation.call_args.kwargs["input"]["timeout"]
            == 30.0
        )

        assert mock_generation.update.call_count == 2
        # Check that the first call has the expected parameters
        first_call = mock_generation.update.call_args_list[0]
        assert first_call.kwargs["output"] == mock_chat_completion
        assert first_call.kwargs["model_parameters"]["temperature"] == "0.7"
        assert first_call.kwargs["model_parameters"]["timeout"] == "30.0"
