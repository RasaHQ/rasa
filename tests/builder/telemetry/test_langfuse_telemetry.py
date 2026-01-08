import asyncio
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletion

from rasa.builder.copilot.models import (
    CopilotChatMessage,
    CopilotTurnRequest,
    TextContent,
    UsageStatistics,
    UserChatMessage,
)
from rasa.builder.shared.tracker_context import CurrentState, TrackerContext
from rasa.builder.telemetry.copilot_langfuse_telemetry import CopilotLangfuseTelemetry


class TestCopilotLangfuseTelemetry:
    """Test class for CopilotLangfuseTelemetry public methods."""

    @patch("rasa.builder.telemetry.copilot_langfuse_telemetry.langfuse.get_client")
    def test_trace_copilot_tracker_context(self, mock_get_client: Mock) -> None:
        # Given
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        current_state = CurrentState(
            latest_message="Hello",
            active_flow=None,
            flow_stack=None,
            slots={},
            latest_action=None,
            followup_action=None,
        )

        tracker_context = TrackerContext(
            current_state=current_state,
            conversation_turns=[],
        )
        max_conversation_turns = 10
        session_id = "test-session-123"

        # When
        CopilotLangfuseTelemetry.trace_copilot_tracker_context(
            tracker_context, max_conversation_turns, session_id
        )

        # Then - verify Langfuse client is called with correct parameters
        mock_client.update_current_span.assert_called_once()
        call_args = mock_client.update_current_span.call_args

        assert "tracker_context" in call_args[1]["output"]
        assert call_args[1]["metadata"]["max_conversation_turns"] == 10
        assert call_args[1]["metadata"]["session_id"] == "test-session-123"

    @patch("rasa.builder.telemetry.copilot_langfuse_telemetry.langfuse.get_client")
    def test_trace_copilot_tracker_context_with_none_tracker_context(
        self, mock_get_client: Mock
    ) -> None:
        """Test that trace_copilot_tracker_context handles None without error."""
        # Given
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        tracker_context = None
        max_conversation_turns = 10
        session_id = "test-session-123"

        # When
        CopilotLangfuseTelemetry.trace_copilot_tracker_context(
            tracker_context, max_conversation_turns, session_id
        )

        # Then - verify Langfuse client is called with correct parameters
        mock_client.update_current_span.assert_called_once()
        call_args = mock_client.update_current_span.call_args

        # Verify that tracker_context is None in the output
        assert "tracker_context" in call_args[1]["output"]
        assert call_args[1]["output"]["tracker_context"] is None
        assert call_args[1]["metadata"]["max_conversation_turns"] == 10
        assert call_args[1]["metadata"]["session_id"] == "test-session-123"

    @patch("rasa.builder.telemetry.copilot_langfuse_telemetry.langfuse.get_client")
    def test_trace_copilot_relevant_assistant_files(
        self, mock_get_client: Mock
    ) -> None:
        # Given
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # BotFiles is a Dict[str, Optional[str]]
        relevant_files = {"file1.py": "content1", "file2.py": "content2"}

        # When
        CopilotLangfuseTelemetry.trace_copilot_relevant_assistant_files(relevant_files)

        # Then - verify Langfuse client is called with correct parameters
        mock_client.update_current_span.assert_called_once()
        call_args = mock_client.update_current_span.call_args
        assert "relevant_assistant_files" in call_args[1]["output"]
        assert call_args[1]["output"]["relevant_assistant_files"] == relevant_files

    @patch("rasa.builder.telemetry.copilot_langfuse_telemetry.langfuse.get_client")
    def test_setup_copilot_endpoint_call_trace_attributes(
        self, mock_get_client: Mock
    ) -> None:
        # Given
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Create test data
        hello_rasa_project_id = "proj-123"
        chat_id = "chat-456"
        user_id = "user-789"

        # Create a proper CopilotTurnRequest with a UserChatMessage
        user_message = UserChatMessage(
            role="user",
            content=[
                TextContent(
                    type="text", text="Can you show me how to create a new intent?"
                )
            ],
        )
        request = CopilotTurnRequest(
            session_id="test-session",
            message=user_message,
            chat_id=chat_id,
            project_id=hello_rasa_project_id,
        )

        # Mock handler
        handler = Mock()
        mock_response = Mock()
        mock_response.content = "Response text"
        handler.generated_responses = [mock_response]
        handler.extract_references.return_value = Mock(references=[])
        relevant_documents = [Mock()]

        # Mock copilot_context
        copilot_context = Mock()
        copilot_context.copilot_chat_history = [
            UserChatMessage(
                role="user",
                content=[TextContent(type="text", text="Hello, how can I help you?")],
            ),
            CopilotChatMessage(
                role="copilot",
                content=[TextContent(type="text", text="I can help you with that!")],
            ),
            UserChatMessage(
                role="user",
                content=[
                    TextContent(
                        type="text", text="Can you show me how to create a new intent?"
                    )
                ],
            ),
        ]

        # When
        CopilotLangfuseTelemetry.setup_copilot_endpoint_call_trace_attributes(
            hello_rasa_project_id,
            chat_id,
            user_id,
            request,
            handler,
            relevant_documents,
            copilot_context,
        )

        # Then - verify Langfuse client is called with correct parameters
        mock_client.update_current_trace.assert_called_once()
        call_args = mock_client.update_current_trace.call_args

        # Check the call arguments
        assert call_args[1]["user_id"] == user_id
        assert (
            call_args[1]["session_id"]
            == f"PID-{hello_rasa_project_id}-UID-{user_id}-CID-{chat_id}"
        )
        assert "message" in call_args[1]["input"]
        assert "tracker_event_attachments" in call_args[1]["input"]
        assert "answer" in call_args[1]["output"]
        assert "response_category" in call_args[1]["output"]
        assert "references" in call_args[1]["output"]
        # Check metadata structure
        metadata = call_args[1]["metadata"]
        assert "ids" in metadata
        assert "copilot_additional_context" in metadata
        # Check ids section
        assert metadata["ids"]["user_id"] == user_id
        assert metadata["ids"]["project_id"] == hello_rasa_project_id
        assert metadata["ids"]["chat_history_id"] == chat_id
        # Check copilot_additional_context section
        copilot_context_section = metadata["copilot_additional_context"]
        assert "relevant_documents" in copilot_context_section
        assert "relevant_assistant_files" in copilot_context_section
        assert "assistant_tracker_context" in copilot_context_section
        assert "assistant_logs" in copilot_context_section
        assert "copilot_chat_history" in copilot_context_section
        assert len(copilot_context_section["copilot_chat_history"]) == 3

    @patch("rasa.builder.telemetry.copilot_langfuse_telemetry.langfuse.get_client")
    def test_trace_copilot_streaming_generation_decorator(
        self, mock_get_client: Mock
    ) -> None:
        """Test the trace_copilot_streaming_generation decorator."""
        # Given
        copilot_input_messages = [{"role": "user", "content": "Hello"}]

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_generation = Mock()
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_generation)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_client.start_as_current_generation.return_value = mock_context_manager

        # Create a mock Copilot instance
        mock_copilot = Mock()
        mock_copilot.__class__.__name__ = "Copilot"
        mock_copilot.llm_config = {"temperature": 0.7, "model": "gpt-4"}
        mock_copilot.usage_statistics = UsageStatistics(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4",
        )

        # Create the decorator and apply it to the mock streamingfunction
        decorator = CopilotLangfuseTelemetry.trace_legacy_copilot_streaming_generation

        async def mock_streaming_function(
            self: Any, messages: List[Dict[str, Any]]
        ) -> AsyncGenerator[str, None]:
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"

        decorated_function = decorator(mock_streaming_function)

        # When
        async def run_test() -> List[str]:
            result: List[str] = []
            async for chunk in decorated_function(mock_copilot, copilot_input_messages):
                result.append(chunk)
            return result

        result: List[str] = asyncio.run(run_test())

        # Then
        assert result == ["chunk1", "chunk2", "chunk3"]

        mock_client.start_as_current_generation.assert_called_once()
        assert (
            mock_client.start_as_current_generation.call_args[1]["input"]["messages"]
            == copilot_input_messages
        )

        # The update on the span is called twice:
        # - Once for the model parameters and output
        # - Once for the usage statistics
        assert mock_generation.update.call_count == 2

        assert (
            mock_generation.update.call_args_list[0].kwargs["model_parameters"]
            == mock_copilot.llm_config
        )
        assert (
            mock_generation.update.call_args_list[0].kwargs["output"]
            == "chunk1chunk2chunk3"
        )
        assert "usage_details" in mock_generation.update.call_args_list[1].kwargs
        assert "cost_details" in mock_generation.update.call_args_list[1].kwargs
        assert "model" in mock_generation.update.call_args_list[1].kwargs

    @patch("rasa.builder.telemetry.copilot_langfuse_telemetry.langfuse.get_client")
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
        decorator = CopilotLangfuseTelemetry.trace_document_retrieval_generation

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
