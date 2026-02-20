from unittest.mock import Mock, patch

from rasa.builder.copilot.models import (
    CopilotChatMessage,
    CopilotTurnRequest,
    ExceptionContent,
    ResponseCategory,
    TextContent,
    UserChatMessage,
)
from rasa.builder.shared.tracker_context import CurrentState, TrackerContext
from rasa.builder.telemetry.langfuse.copilot_endpoint_langfuse_telemetry import (
    CopilotEndpointLangfuseTelemetry,
)


class TestCopilotEndpointLangfuseTelemetry:
    """Test class for CopilotEndpointLangfuseTelemetry public methods."""

    @patch("rasa.builder.telemetry.langfuse.langfuse_compat.langfuse.get_client")
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
        CopilotEndpointLangfuseTelemetry.trace_copilot_tracker_context(
            tracker_context, max_conversation_turns, session_id
        )

        # Then - verify Langfuse client is called with correct parameters
        mock_client.update_current_span.assert_called_once()
        call_args = mock_client.update_current_span.call_args

        assert "tracker_context" in call_args[1]["output"]
        assert call_args[1]["metadata"]["max_conversation_turns"] == 10
        assert call_args[1]["metadata"]["session_id"] == "test-session-123"

    @patch("rasa.builder.telemetry.langfuse.langfuse_compat.langfuse.get_client")
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
        CopilotEndpointLangfuseTelemetry.trace_copilot_tracker_context(
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

    @patch("rasa.builder.telemetry.langfuse.langfuse_compat.langfuse.get_client")
    def test_trace_copilot_relevant_assistant_files(
        self, mock_get_client: Mock
    ) -> None:
        # Given
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # BotFiles is a Dict[str, Optional[str]]
        relevant_files = {"file1.py": "content1", "file2.py": "content2"}

        # When
        CopilotEndpointLangfuseTelemetry.trace_copilot_relevant_assistant_files(
            relevant_files
        )

        # Then - verify Langfuse client is called with correct parameters
        mock_client.update_current_span.assert_called_once()
        call_args = mock_client.update_current_span.call_args
        assert "relevant_assistant_files" in call_args[1]["output"]
        assert call_args[1]["output"]["relevant_assistant_files"] == relevant_files

    @patch("rasa.builder.telemetry.langfuse.langfuse_compat.langfuse.get_client")
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
        handler.extract_exception_response.return_value = None
        handler.extract_response_category.return_value = ResponseCategory.COPILOT
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
        CopilotEndpointLangfuseTelemetry.setup_copilot_endpoint_call_trace_attributes(
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

    @patch("rasa.builder.telemetry.langfuse.langfuse_compat.langfuse.get_client")
    def test_setup_copilot_endpoint_call_trace_attributes_with_exception_response(
        self, mock_get_client: Mock
    ) -> None:
        # Given
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        hello_rasa_project_id = "proj-123"
        chat_id = "chat-456"
        user_id = "user-789"

        user_message = UserChatMessage(
            role="user",
            content=[TextContent(type="text", text="Hello")],
        )
        request = CopilotTurnRequest(
            session_id="test-session",
            message=user_message,
            chat_id=chat_id,
            project_id=hello_rasa_project_id,
        )

        original_exception = Exception("boom")
        exception_response = ExceptionContent(
            content="Sorry, something went wrong.",
            original_exception=original_exception,
        )

        handler = Mock()
        handler.generated_responses = [exception_response]
        handler.extract_exception_response.return_value = exception_response
        handler.extract_response_category.return_value = ResponseCategory.EXCEPTION
        handler.extract_references.return_value = Mock(references=[Mock()])

        relevant_documents = [Mock()]
        copilot_context = Mock()
        copilot_context.assistant_files = {}
        copilot_context.assistant_logs = []
        copilot_context.tracker_context = None
        copilot_context.copilot_chat_history = [user_message]

        # When
        CopilotEndpointLangfuseTelemetry.setup_copilot_endpoint_call_trace_attributes(
            hello_rasa_project_id,
            chat_id,
            user_id,
            request,
            handler,
            relevant_documents,
            copilot_context,
        )

        # Then
        mock_client.update_current_trace.assert_called_once()
        call_args = mock_client.update_current_trace.call_args
        output = call_args[1]["output"]

        assert output["answer"] == exception_response.content
        assert output["response_category"] == ResponseCategory.EXCEPTION.value
        assert output["references"] == []
        # Uses repr() for serialization
        assert output["original_exception"] == repr(original_exception)
        # Diagnostic metadata dict (safe_metadata omits exception_stack_trace)
        metadata = output["exception_metadata"]
        assert metadata is not None
        assert metadata["exception_type"] == "Exception"
        assert metadata["exception_message"] == "boom"
        # safe_metadata intentionally excludes exception_stack_trace for telemetry
        assert "exception_stack_trace" not in metadata

    @patch(
        "rasa.builder.telemetry.langfuse.copilot_endpoint_langfuse_telemetry"
        ".with_langfuse"
    )
    def test_update_trace_on_error_calls_update_current_trace(
        self, mock_with_langfuse: Mock
    ) -> None:
        mock_client = Mock()
        mock_langfuse = Mock()
        mock_langfuse.get_client.return_value = mock_client
        mock_with_langfuse.return_value.__enter__ = Mock(return_value=mock_langfuse)
        mock_with_langfuse.return_value.__exit__ = Mock(return_value=False)

        user_message = UserChatMessage(
            role="user",
            content=[TextContent(type="text", text="Help me")],
        )
        request = CopilotTurnRequest(
            session_id="s1",
            message=user_message,
            chat_id="c1",
            project_id="p1",
        )
        exc = ValueError("something failed")

        CopilotEndpointLangfuseTelemetry.update_trace_on_error(request, exc)

        mock_client.update_current_trace.assert_called_once()
        call_kw = mock_client.update_current_trace.call_args[1]
        assert call_kw["input"]["message"] == "Help me"
        assert call_kw["output"]["error"] == repr(exc)
        assert call_kw["output"]["error_type"] == "ValueError"
        assert "error_stack_trace" in call_kw["output"]
        assert call_kw["tags"] == ["error"]

    @patch(
        "rasa.builder.telemetry.langfuse.copilot_endpoint_langfuse_telemetry"
        ".with_langfuse"
    )
    def test_update_trace_on_error_no_op_when_langfuse_unavailable(
        self, mock_with_langfuse: Mock
    ) -> None:
        mock_with_langfuse.return_value.__enter__ = Mock(return_value=None)
        mock_with_langfuse.return_value.__exit__ = Mock(return_value=False)

        user_message = UserChatMessage(
            role="user",
            content=[TextContent(type="text", text="Hi")],
        )
        request = CopilotTurnRequest(
            session_id="s1",
            message=user_message,
            chat_id="c1",
            project_id="p1",
        )
        exc = RuntimeError("oops")

        CopilotEndpointLangfuseTelemetry.update_trace_on_error(request, exc)

        mock_with_langfuse.assert_called_once()
        # No get_client / update_current_trace when lf is None
        mock_lf = mock_with_langfuse.return_value.__enter__.return_value
        assert mock_lf is None

    @patch(
        "rasa.builder.telemetry.langfuse.copilot_endpoint_langfuse_telemetry"
        ".with_langfuse"
    )
    def test_update_trace_on_error_does_not_reraise_when_update_fails(
        self, mock_with_langfuse: Mock
    ) -> None:
        """update_trace_on_error does not re-raise when update_current_trace fails."""
        mock_client = Mock()
        mock_client.update_current_trace.side_effect = Exception("Langfuse API error")
        mock_langfuse = Mock()
        mock_langfuse.get_client.return_value = mock_client
        mock_with_langfuse.return_value.__enter__ = Mock(return_value=mock_langfuse)
        mock_with_langfuse.return_value.__exit__ = Mock(return_value=False)

        user_message = UserChatMessage(
            role="user",
            content=[TextContent(type="text", text="Hi")],
        )
        request = CopilotTurnRequest(
            session_id="s1",
            message=user_message,
            chat_id="c1",
            project_id="p1",
        )
        exc = RuntimeError("original error")

        CopilotEndpointLangfuseTelemetry.update_trace_on_error(request, exc)

        mock_client.update_current_trace.assert_called_once()
        # No exception propagates; inner failure is logged at debug only
