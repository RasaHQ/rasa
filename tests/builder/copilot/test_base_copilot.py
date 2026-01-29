"""Tests for BaseCopilot helper methods."""

import json

# Import Copilot for testing (uses LegacyCopilot or AgentCopilot based on config)
from rasa.builder.copilot import Copilot
from rasa.builder.copilot.base_copilot import BaseCopilot
from rasa.builder.copilot.constants import ROLE_COPILOT, ROLE_USER
from rasa.builder.copilot.models import (
    CopilotChatMessage,
    EventContent,
    FileContent,
    InternalCopilotRequestChatMessage,
    LogContent,
    ResponseCategory,
    TextContent,
    UserChatMessage,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.shared.tracker_context import (
    AssistantConversationTurn,
    AssistantMessage,
    CurrentState,
    TrackerContext,
    TrackerEvent,
    UserMessage,
)


class TestCreateChatHistoryMessages:
    """Test _create_chat_history_messages method."""

    def test_filters_guardrails_violations(self) -> None:
        """Test that guardrails policy violations are filtered out."""
        copilot = Copilot()
        chat_history = [
            UserChatMessage(
                role=ROLE_USER,
                content=[TextContent(type="text", text="Hello")],
            ),
            CopilotChatMessage(
                role=ROLE_COPILOT,
                content=[TextContent(type="text", text="Blocked response")],
                response_category=ResponseCategory.GUARDRAILS_POLICY_VIOLATION,
            ),
            UserChatMessage(
                role=ROLE_USER,
                content=[TextContent(type="text", text="Second message")],
            ),
        ]

        result = copilot._create_chat_history_messages(chat_history)

        # Should have 2 messages (guardrails violation filtered)
        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Second message"

    def test_converts_to_openai_format(self) -> None:
        """Test that messages are converted to OpenAI format."""
        copilot = Copilot()
        chat_history = [
            UserChatMessage(
                role=ROLE_USER,
                content=[TextContent(type="text", text="User message")],
            ),
            CopilotChatMessage(
                role=ROLE_COPILOT,
                content=[TextContent(type="text", text="Assistant response")],
            ),
        ]

        result = copilot._create_chat_history_messages(chat_history)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "User message"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Assistant response"

    def test_empty_chat_history(self) -> None:
        """Test with empty chat history."""
        copilot = Copilot()
        result = copilot._create_chat_history_messages([])
        assert result == []


class TestFormatConversationHistory:
    """Test _format_conversation_history static method."""

    def test_formats_multiple_turns(self) -> None:
        """Test formatting multiple conversation turns."""
        tracker_context = TrackerContext(
            conversation_turns=[
                AssistantConversationTurn(
                    user_message=UserMessage(
                        text="Hello",
                        predicted_commands=["greet"],
                    ),
                    assistant_messages=[
                        AssistantMessage(text="Hi there!"),
                    ],
                    context_events=[],
                ),
                AssistantConversationTurn(
                    user_message=UserMessage(
                        text="How are you?",
                        predicted_commands=["chitchat"],
                    ),
                    assistant_messages=[
                        AssistantMessage(text="I'm good, thanks!"),
                    ],
                    context_events=[],
                ),
            ],
            current_state=CurrentState(),
        )

        result = BaseCopilot._format_conversation_history(tracker_context)
        parsed = json.loads(result)

        assert len(parsed["conversation_history"]) == 2
        assert parsed["conversation_history"][0]["turn_id"] == 1
        assert parsed["conversation_history"][0]["USER"]["text"] == "Hello"
        assert parsed["conversation_history"][1]["turn_id"] == 2

    def test_formats_context_events(self) -> None:
        """Test that context events are included."""
        tracker_context = TrackerContext(
            conversation_turns=[
                AssistantConversationTurn(
                    user_message=UserMessage(
                        text="Transfer $100", predicted_commands=[]
                    ),
                    assistant_messages=[AssistantMessage(text="Processing...")],
                    context_events=[
                        TrackerEvent(
                            event="slot_set",
                            data={"slot_name": "amount", "slot_value": 100},
                        ),
                        TrackerEvent(
                            event="action_executed",
                            data={"action_name": "action_transfer"},
                        ),
                    ],
                ),
            ],
            current_state=CurrentState(),
        )

        result = BaseCopilot._format_conversation_history(tracker_context)
        parsed = json.loads(result)

        turn = parsed["conversation_history"][0]
        assert "other_tracker_events" in turn
        assert len(turn["other_tracker_events"]) == 2
        assert turn["other_tracker_events"][0]["event"] == "slot_set"

    def test_handles_none_tracker_context(self) -> None:
        """Test that None tracker context returns empty history."""
        result = BaseCopilot._format_conversation_history(None)
        parsed = json.loads(result)
        assert parsed["conversation_history"] == []

    def test_handles_turn_without_user_message(self) -> None:
        """Test turn without user message."""
        tracker_context = TrackerContext(
            conversation_turns=[
                AssistantConversationTurn(
                    user_message=None,
                    assistant_messages=[AssistantMessage(text="Welcome!")],
                    context_events=[],
                ),
            ],
            current_state=CurrentState(),
        )

        result = BaseCopilot._format_conversation_history(tracker_context)
        parsed = json.loads(result)

        turn = parsed["conversation_history"][0]
        assert "USER" not in turn
        assert "ASSISTANT" in turn


class TestFormatCurrentState:
    """Test _format_current_state static method."""

    def test_formats_populated_state(self) -> None:
        """Test formatting populated current state."""
        tracker_context = TrackerContext(
            conversation_turns=[],
            current_state=CurrentState(
                latest_message="Hello",
                active_flow="greeting_flow",
                flow_stack=[{"flow_name": "main"}],
                slots={"user_name": "John"},
                latest_action="action_greet",
                followup_action="utter_welcome",
            ),
        )

        result = BaseCopilot._format_current_state(tracker_context)
        parsed = json.loads(result)

        assert parsed["latest_message"] == "Hello"
        assert parsed["active_flow"] == "greeting_flow"
        assert parsed["slots"] == {"user_name": "John"}

    def test_handles_none_tracker_context(self) -> None:
        """Test that None returns empty dict."""
        result = BaseCopilot._format_current_state(None)
        assert result == "{}"

    def test_handles_empty_current_state(self) -> None:
        """Test that empty current_state returns default values."""
        tracker_context = TrackerContext(
            conversation_turns=[],
            current_state=CurrentState(),  # Empty state with defaults
        )
        result = BaseCopilot._format_current_state(tracker_context)
        # Should contain default None values for optional fields
        import json

        parsed = json.loads(result)
        assert "latest_message" in parsed
        assert "active_flow" in parsed


class TestFormatDocuments:
    """Test _format_documents static method."""

    def test_formats_single_document(self) -> None:
        """Test formatting a single document."""
        documents = [
            Document(
                title="Test Doc",
                url="https://test.com",
                content="Test content",
            )
        ]

        result = BaseCopilot._format_documents(documents)
        parsed = json.loads(result)

        assert "sources" in parsed
        assert len(parsed["sources"]) == 1
        assert parsed["sources"][0]["idx"] == 1
        assert parsed["sources"][0]["title"] == "Test Doc"
        assert parsed["sources"][0]["url"] == "https://test.com"
        assert parsed["sources"][0]["content"] == "Test content"

    def test_returns_none_for_empty_list(self) -> None:
        """Test that empty list returns None."""
        result = BaseCopilot._format_documents([])
        assert result is None

    def test_indices_start_at_one(self) -> None:
        """Test that document indices start at 1."""
        documents = [
            Document(title="Doc 1", url="url1", content="content1"),
            Document(title="Doc 2", url="url2", content="content2"),
        ]

        result = BaseCopilot._format_documents(documents)
        parsed = json.loads(result)

        assert parsed["sources"][0]["idx"] == 1
        assert parsed["sources"][1]["idx"] == 2


class TestFormatTrackerEventAttachments:
    """Test _format_tracker_event_attachments static method."""

    def test_formats_multiple_events(self) -> None:
        """Test formatting multiple event attachments."""
        events = [
            EventContent(
                type="event",
                event="user_uttered",
                event_data={"text": "Hello", "intent": {"name": "greet"}},
            ),
            EventContent(
                type="event",
                event="slot_set",
                event_data={"slot_name": "name", "slot_value": "John"},
            ),
        ]

        result = BaseCopilot._format_tracker_event_attachments(events)
        parsed = json.loads(result)

        assert len(parsed) == 2
        assert parsed[0]["event"] == "user_uttered"
        assert parsed[1]["event"] == "slot_set"

    def test_returns_none_for_empty_list(self) -> None:
        """Test that empty list returns None."""
        result = BaseCopilot._format_tracker_event_attachments([])
        assert result is None


class TestExtractTrackerEventAttachments:
    """Test _extract_tracker_event_attachments static method."""

    def test_extracts_events_from_user_message(self) -> None:
        """Test extracting EventContent from UserChatMessage."""
        message = UserChatMessage(
            role=ROLE_USER,
            content=[
                TextContent(type="text", text="What happened?"),
                EventContent(
                    type="event",
                    event="action_executed",
                    event_data={"action_name": "test"},
                ),
                EventContent(
                    type="event",
                    event="slot_set",
                    event_data={"slot": "value"},
                ),
            ],
        )

        result = BaseCopilot._extract_tracker_event_attachments(message)

        assert len(result) == 2
        assert all(isinstance(e, EventContent) for e in result)
        assert result[0].event == "action_executed"
        assert result[1].event == "slot_set"

    def test_returns_empty_for_copilot_message(self) -> None:
        """Test that CopilotChatMessage returns empty list."""
        message = CopilotChatMessage(
            role=ROLE_COPILOT,
            content=[TextContent(type="text", text="Response")],
        )

        result = BaseCopilot._extract_tracker_event_attachments(message)
        assert result == []

    def test_returns_empty_for_message_without_events(self) -> None:
        """Test message without EventContent returns empty list."""
        message = UserChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="Just text")],
        )

        result = BaseCopilot._extract_tracker_event_attachments(message)
        assert result == []


class TestFormatNormalMessageForQueryChatHistory:
    """Test _format_normal_message_for_query_chat_history static method."""

    def test_formats_user_message(self) -> None:
        """Test formatting user message."""
        message = UserChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="Hello world")],
        )

        result = BaseCopilot._format_normal_message_for_query_chat_history(message)
        assert result == "Hello world"

    def test_formats_copilot_message(self) -> None:
        """Test formatting copilot message."""
        message = CopilotChatMessage(
            role=ROLE_COPILOT,
            content=[TextContent(type="text", text="Hi there!")],
        )

        result = BaseCopilot._format_normal_message_for_query_chat_history(message)
        assert result == "Hi there!"


class TestFormatInternalMessageForQueryChatHistory:
    """Test _format_internal_message_for_query_chat_history static method."""

    def test_formats_message_with_text_and_logs(self) -> None:
        """Test formatting internal message with both text and logs."""
        message = InternalCopilotRequestChatMessage(
            role="internal_copilot_request",
            content=[
                TextContent(type="text", text="Training failed"),
                LogContent(type="log", content="Error in domain.yml"),
            ],
            response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
        )

        result = BaseCopilot._format_internal_message_for_query_chat_history(message)
        assert "Training failed" in result
        assert "Logs: Error in domain.yml" in result

    def test_formats_message_with_only_logs(self) -> None:
        """Test formatting internal message with only logs."""
        message = InternalCopilotRequestChatMessage(
            role="internal_copilot_request",
            content=[
                LogContent(type="log", content="Some log content"),
            ],
            response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
        )

        result = BaseCopilot._format_internal_message_for_query_chat_history(message)
        assert result == "Logs: Some log content"

    def test_formats_message_with_only_text(self) -> None:
        """Test formatting internal message with only text."""
        message = InternalCopilotRequestChatMessage(
            role="internal_copilot_request",
            content=[
                TextContent(type="text", text="Just text"),
            ],
            response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
        )

        result = BaseCopilot._format_internal_message_for_query_chat_history(message)
        assert result == "Just text"

    def test_formats_empty_message(self) -> None:
        """Test formatting internal message with no content."""
        message = InternalCopilotRequestChatMessage(
            role="internal_copilot_request",
            content=[
                FileContent(type="file", file_path="test.yml", file_content="content"),
            ],
            response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
        )

        result = BaseCopilot._format_internal_message_for_query_chat_history(message)
        assert result == ""
