import json
from typing import Any, Dict, List, Optional

import pytest

from rasa.builder.copilot import Copilot
from rasa.builder.copilot.constants import (
    ROLE_COPILOT,
    ROLE_COPILOT_INTERNAL,
    ROLE_SYSTEM,
)
from rasa.builder.copilot.legacy_copilot import LegacyCopilot
from rasa.builder.copilot.models import (
    ChatMessage,
    CopilotChatMessage,
    CopilotContext,
    CopilotSystemMessage,
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
from rasa.core.agent import Agent
from rasa.shared.constants import ROLE_USER
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, SessionStarted, UserUttered
from rasa.shared.core.flows.yaml_flows_io import YamlFlowsWriter
from rasa.shared.core.trackers import DialogueStateTracker


def sample_tracker(domain: Domain) -> DialogueStateTracker:
    """Create a sample tracker with conversation history."""
    events = [
        SessionStarted(),
        UserUttered("test"),
        ActionExecuted("utter_test"),
    ]

    return DialogueStateTracker.from_events(
        sender_id="test_sender", evts=events, slots=domain.slots
    )


def sample_chat_history() -> list[UserChatMessage]:
    """Create sample chat history."""
    return [
        UserChatMessage(
            role="user",
            content=[
                TextContent(
                    type="text",
                    text="What is a rephraser?",
                )
            ],
        )
    ]


def sample_bot_logs() -> str:
    """Create sample bot logs."""
    return """"""


async def bot_files_for_agent(agent: Agent) -> dict[str, str]:
    """Get bot files for an agent."""
    return {
        "domain.yml": agent.domain.as_yaml(),
        "flows.yml": YamlFlowsWriter.dumps(
            (await agent.processor.get_flows()).underlying_flows,
            should_clean_json=True,
        ),
        "config.yml": "",
    }


class TestCopilotCore:
    @pytest.mark.parametrize(
        "chat_history,expected_openai_format",
        [
            # Only one message
            (
                [
                    UserChatMessage(
                        role="user",
                        content=[TextContent(type="text", text="Hello")],
                        response_category=None,
                    ),
                ],
                [
                    # System prompt
                    {
                        "role": "system",
                        "content": "[SYSTEM_PROMPT_PLACEHOLDER]",
                    },
                    # Context + user message
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "[LAST_USER_MESSAGE_CONTEXT_PROMPT_PLACEHOLDER]",  # noqa: E501
                            },
                            {"type": "text", "text": "Hello"},
                        ],
                    },
                ],
            ),
            # Multiple messages
            (
                [
                    UserChatMessage(
                        role="user",
                        content=[TextContent(type="text", text="Hello")],
                        response_category=None,
                    ),
                    CopilotChatMessage(
                        role="copilot",
                        content=[TextContent(type="text", text="Hi there!")],
                        response_category=None,
                    ),
                    UserChatMessage(
                        role="user",
                        content=[TextContent(type="text", text="How can I do...?")],
                        response_category=None,
                    ),
                ],
                [
                    # System prompt
                    {
                        "role": "system",
                        "content": "[SYSTEM_PROMPT_PLACEHOLDER]",
                    },
                    # Chat history
                    {
                        "role": "user",
                        "content": "Hello",
                    },
                    {
                        "role": "assistant",
                        "content": "Hi there!",
                    },
                    # Last message with context
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "[LAST_USER_MESSAGE_CONTEXT_PROMPT_PLACEHOLDER]",  # noqa: E501
                            },
                            {"type": "text", "text": "How can I do...?"},
                        ],
                    },  # Last message with context
                ],
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_build_messages_openai_format(
        self,
        chat_history: List[ChatMessage],
        expected_openai_format: List[Dict[str, Any]],
    ):
        """Test that _build_messages produces the exact OpenAI format structure."""
        # Given
        context = CopilotContext(
            tracker_context=None,
            assistant_logs="Some assistant logs",
            assistant_files={"domain.yml": "version: '3.1'"},
            copilot_chat_history=chat_history,
        )
        copilot = Copilot()

        # When
        result = await copilot._build_messages(context=context, relevant_documents=[])

        # Then
        assert len(result) == len(expected_openai_format)

        for i, (actual, expected) in enumerate(zip(result, expected_openai_format)):
            # Check role
            assert actual["role"] == expected["role"]

            # System prompt - just check it's a string and not empty
            if expected["content"] == "[SYSTEM_PROMPT_PLACEHOLDER]":
                assert isinstance(actual["content"], str)
                assert len(actual["content"]) > 0

            # Simple string content (chat history)
            elif isinstance(expected["content"], str):
                assert actual["content"] == expected["content"]

            # Last user message with context
            elif isinstance(expected["content"], list):
                assert isinstance(actual["content"], list)
                assert len(actual["content"]) == len(expected["content"])

                # First content block should always be context (rendered prompt)
                assert actual["content"][0]["type"] == "text"
                assert "Some assistant logs" in actual["content"][0]["text"]
                assert "domain.yml" in actual["content"][0]["text"]

                # Second content block should always be the actual user message
                assert actual["content"][1]["text"] == chat_history[-1].content[0].text

    @pytest.mark.asyncio
    async def test_build_messages_openai_format_with_attachments(self):
        """Test _build_messages with EventContent attachments in user message."""
        # Given
        user_message_with_events = UserChatMessage(
            role="user",
            content=[
                TextContent(type="text", text="What's this error?"),
                EventContent(
                    type="event",
                    event="user_uttered",
                    event_data={
                        "text": "I want to transfer money",
                        "intent": {"name": "transfer_money", "confidence": 0.95},
                    },
                ),
                EventContent(
                    type="event",
                    event="action_executed",
                    event_data={
                        "action_name": "action_check_balance",
                    },
                ),
                EventContent(
                    type="event",
                    event="bot",
                    event_data={
                        "response": "Your balance is 100 USD. Do you want to proceed?",
                    },
                ),
            ],
            response_category=None,
        )

        chat_history = [user_message_with_events]

        context = CopilotContext(
            tracker_context=TrackerContext(
                conversation_turns=[],
                current_state=CurrentState(),
            ),
            assistant_logs="Some logs",
            assistant_files={"domain.yml": "version: '3.1'"},
            copilot_chat_history=chat_history,
        )

        copilot = Copilot()

        # When
        result = await copilot._build_messages(context=context, relevant_documents=[])

        # Then
        assert len(result) == 2  # System message + user message with context

        # Check that the last user message has context + text content
        last_message: Dict[str, Any] = result[-1]
        context_content: Dict[str, Any] = last_message["content"][0]
        user_text_content: Dict[str, Any] = last_message["content"][1]

        assert last_message["role"] == "user"
        assert isinstance(last_message["content"], list)
        assert len(last_message["content"]) == 2  # Context + user text

        assert context_content["type"] == "text"
        assert "user_uttered" in context_content["text"]
        assert "action_executed" in context_content["text"]
        assert "action_check_balance" in context_content["text"]

        assert user_text_content["type"] == "text"
        assert user_text_content["text"] == "What's this error?"

    @pytest.mark.parametrize(
        "message, raises_error",
        [
            (
                CopilotChatMessage(
                    role=ROLE_COPILOT,
                    content=[
                        TextContent(type="text", text="This is a copilot response")
                    ],
                ),
                True,
            ),
            (
                CopilotSystemMessage(role=ROLE_SYSTEM),
                True,
            ),
            (
                InternalCopilotRequestChatMessage(
                    role=ROLE_COPILOT_INTERNAL,
                    content=[LogContent(type="log", content="...")],
                    response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                ),
                False,
            ),
            (
                UserChatMessage(
                    role=ROLE_USER, content=[TextContent(type="text", text="...")]
                ),
                False,
            ),
        ],
    )
    def test_process_latest_message_raises_error_for_unsupported_message_types(
        self, message: ChatMessage, raises_error: bool
    ):
        """Test that _process_latest_message raises ValueError for unsupported types."""
        # Given
        copilot = Copilot()
        context = CopilotContext(
            copilot_chat_history=[message],
            assistant_logs="",
            assistant_files={},
            tracker_context=None,
        )

        # When / Then
        if raises_error:
            with pytest.raises(ValueError):
                copilot._process_latest_message(context, [])
        else:
            copilot._process_latest_message(context, [])

    def test_process_latest_message_internal_copilot_request_uses_training_error_prompt(
        self,
    ):
        # Given
        copilot = Copilot()
        context = CopilotContext(
            tracker_context=None,
            assistant_logs="",  # Part of the internal message
            assistant_files={},  # Part of the internal message
            copilot_chat_history=[
                InternalCopilotRequestChatMessage(
                    role=ROLE_COPILOT_INTERNAL,
                    content=[
                        LogContent(
                            type="log", content="Training failed: Invalid YAML syntax"
                        ),
                        FileContent(
                            type="file",
                            file_path="domain.yml",
                            file_content="version: '3.1'\nintents:\n  - greet",
                        ),
                    ],
                    response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                )
            ],
        )

        mock_documents = [
            Document(
                content="Rasa training documentation content",
                url="https://docs.rasa.com/training",
                title="Rasa Training Guide",
            )
        ]

        # When
        result = copilot._process_latest_message(context, mock_documents)

        # Then
        assert result["role"] == "user"
        assert "content" in result
        assert isinstance(result["content"], str)

        # Check that the content contains the training error prompt elements
        prompt_content = result["content"]
        assert (
            "Training failed: Invalid YAML syntax" in prompt_content
        )  # from log content
        assert "domain.yml" in prompt_content  # from file content
        assert "Rasa training documentation content" in prompt_content  # documentation
        assert "Relevant Documentation" in prompt_content  # documentation section


class TestCopilotFormattingContents:
    def test_format_conversation_history(self):
        # Given
        conversation_turns = [
            AssistantConversationTurn(
                user_message=UserMessage(
                    text="I want to transfer money",
                    predicted_commands=["start flow", "set slot"],
                ),
                assistant_messages=[
                    AssistantMessage(text="How much would you like to transfer?")
                ],
                context_events=[
                    TrackerEvent(
                        event="action_executed",
                        data={"action_name": "action_ask_amount"},
                    ),
                    TrackerEvent(
                        event="slot_set",
                        data={
                            "slot_name": "amount_of_money",
                            "slot_value": 100,
                        },
                    ),
                ],
            ),
        ]

        current_state = CurrentState(
            latest_message="100 USD",
            active_flow="transfer_money",
            flow_stack=[],
            slots={"amount": 100, "recipient": "Anna"},
            latest_action="action_ask_amount",
            followup_action=None,
        )

        tracker_context = TrackerContext(
            conversation_turns=conversation_turns,
            current_state=current_state,
        )

        expected_result: Dict[str, Any] = {
            "conversation_history": [
                {
                    "turn_id": 1,
                    "USER": {
                        "text": "I want to transfer money",
                        "predicted_commands": ["start flow", "set slot"],
                    },
                    "BOT": [{"text": "How much would you like to transfer?"}],
                    "other_tracker_events": [
                        {
                            "event": "action_executed",
                            "data": {"action_name": "action_ask_amount"},
                        },
                        {
                            "event": "slot_set",
                            "data": {
                                "slot_name": "amount_of_money",
                                "slot_value": 100,
                            },
                        },
                    ],
                },
            ]
        }

        # When
        result = Copilot._format_conversation_history(tracker_context)
        parsed_result = json.loads(result)

        # Then
        assert parsed_result == expected_result

    def test_format_conversation_history_empty(self):
        """Test the _format_conversation_history method with empty/None input."""
        # Test with None
        result_none = Copilot._format_conversation_history(None)
        assert result_none == '{\n  "conversation_history": []\n}'

        # Test with empty TrackerContext
        empty_context = TrackerContext(
            conversation_turns=[],
            current_state=CurrentState(),
        )

        result_empty = Copilot._format_conversation_history(empty_context)
        assert result_empty == '{\n  "conversation_history": []\n}'

    @pytest.mark.parametrize(
        "chat_history,expected_query",
        [
            # Empty chat history
            ([], ""),
            # Single user message
            (
                [
                    UserChatMessage(
                        role=ROLE_USER,
                        content=[TextContent(type="text", text="Hello")],
                    )
                ],
                "User: Hello",
            ),
            # Single internal copilot request message
            (
                [
                    InternalCopilotRequestChatMessage(
                        role=ROLE_COPILOT_INTERNAL,
                        content=[
                            LogContent(type="log", content="Training error occurred"),
                            FileContent(
                                type="file",
                                file_path="domain.yml",
                                file_content="version: '3.1'",
                            ),
                        ],
                        response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                    )
                ],
                "User: Logs: Training error occurred",
            ),
            # Multiple messages with guardrails violation (should be filtered out)
            (
                [
                    UserChatMessage(
                        role=ROLE_USER,
                        content=[TextContent(type="text", text="Hello")],
                    ),
                    CopilotChatMessage(
                        role=ROLE_COPILOT,
                        content=[TextContent(type="text", text="Hi there!")],
                        response_category=ResponseCategory.GUARDRAILS_POLICY_VIOLATION,
                    ),
                    UserChatMessage(
                        role=ROLE_USER,
                        content=[TextContent(type="text", text="How can I help?")],
                    ),
                ],
                "User: Hello\nUser: How can I help?",
            ),
            # Mixed message types
            (
                [
                    UserChatMessage(
                        role=ROLE_USER,
                        content=[TextContent(type="text", text="What is Rasa?")],
                    ),
                    CopilotChatMessage(
                        role=ROLE_COPILOT,
                        content=[
                            TextContent(
                                type="text", text="Rasa is an open source framework"
                            )
                        ],
                    ),
                    InternalCopilotRequestChatMessage(
                        role=ROLE_COPILOT_INTERNAL,
                        content=[
                            LogContent(type="log", content="Error in training"),
                            FileContent(
                                type="file",
                                file_path="config.yml",
                                file_content="language: en",
                            ),
                        ],
                        response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                    ),
                ],
                "User: What is Rasa?\nAssistant: Rasa is an open source framework\n"
                "User: Logs: Error in training",
            ),
        ],
    )
    def test_create_documentation_search_query(
        self, chat_history: List[ChatMessage], expected_query: str
    ):
        """Test _create_documentation_search_query method with chat history.

        Note: This method is specific to LegacyCopilot implementation.
        """
        # Given
        context = CopilotContext(
            tracker_context=None,
            assistant_logs="",
            assistant_files={},
            copilot_chat_history=chat_history,
        )

        # When - Use LegacyCopilot directly as this method is specific to legacy
        result = LegacyCopilot._create_documentation_search_query(context)

        # Then
        assert result == expected_query

    @pytest.mark.parametrize(
        "events,expected_count,expected_events",
        [
            # Empty events list
            ([], 0, []),
            # Single event
            (
                [
                    EventContent(
                        type="event",
                        event="user_uttered",
                        event_data={
                            "text": "Hello",
                            "intent": {"name": "greet", "confidence": 0.95},
                        },
                    )
                ],
                1,
                ["user_uttered"],
            ),
            # Multiple events
            (
                [
                    EventContent(
                        type="event",
                        event="user_uttered",
                        event_data={"text": "Hello", "intent": {"name": "greet"}},
                    ),
                    EventContent(
                        type="event",
                        event="action_executed",
                        event_data={"action_name": "action_greet"},
                    ),
                    EventContent(
                        type="event",
                        event="slot_set",
                        event_data={"slot_name": "user_name", "slot_value": "John"},
                    ),
                ],
                3,
                ["user_uttered", "action_executed", "slot_set"],
            ),
        ],
    )
    def test_format_tracker_event_attachments(
        self,
        events: List[EventContent],
        expected_count: int,
        expected_events: List[str],
    ):
        """Test _format_tracker_event_attachments method with EventContent."""
        result = Copilot._format_tracker_event_attachments(events)

        if expected_count == 0:
            assert result is None
        else:
            parsed_result = json.loads(result)
            assert len(parsed_result) == expected_count
            for i, expected_event in enumerate(expected_events):
                assert parsed_result[i]["event"] == expected_event

    @pytest.mark.parametrize(
        "documents,expected_result,expected_sources",
        [
            # Empty documents list
            ([], None, []),
            # Single document
            (
                [
                    Document(
                        content="Rasa is a conversational AI framework",
                        url="https://docs.rasa.com/intro",
                        title="Rasa Introduction",
                    )
                ],
                "not_none",
                [
                    {
                        "idx": 1,
                        "title": "Rasa Introduction",
                        "url": "https://docs.rasa.com/intro",
                        "content": "Rasa is a conversational AI framework",
                    }
                ],
            ),
            # Multiple documents
            (
                [
                    Document(
                        content="Rasa is a conversational AI framework",
                        url="https://docs.rasa.com/intro",
                        title="Rasa Introduction",
                    ),
                    Document(
                        content="Flows define conversation patterns",
                        url="https://docs.rasa.com/flows",
                        title="Rasa Flows",
                    ),
                    Document(
                        content="Slots store conversation memory",
                        url="https://docs.rasa.com/slots",
                        title="Rasa Slots",
                    ),
                ],
                "not_none",
                [
                    {"idx": 1, "title": "Rasa Introduction"},
                    {"idx": 2, "title": "Rasa Flows"},
                    {"idx": 3, "title": "Rasa Slots"},
                ],
            ),
        ],
    )
    def test_format_documents(
        self,
        documents: List[Document],
        expected_result: Optional[str],
        expected_sources: List[Dict[str, Any]],
    ):
        """Test _format_documents method with empty and populated document lists."""
        result = Copilot._format_documents(documents)

        if expected_result is None:
            assert result is None
        else:
            parsed_result = json.loads(result)
            assert "sources" in parsed_result
            assert len(parsed_result["sources"]) == len(expected_sources)

            for i, expected_source in enumerate(expected_sources):
                assert parsed_result["sources"][i]["idx"] == expected_source["idx"]
                if "title" in expected_source:
                    assert (
                        parsed_result["sources"][i]["title"] == expected_source["title"]
                    )
                if "url" in expected_source:
                    assert parsed_result["sources"][i]["url"] == expected_source["url"]
                if "content" in expected_source:
                    assert (
                        parsed_result["sources"][i]["content"]
                        == expected_source["content"]
                    )

    @pytest.mark.parametrize(
        "tracker_context,expected_result,expected_fields",
        [
            # None tracker_context
            (None, "{}", {}),
            # Empty current_state
            (
                TrackerContext(conversation_turns=[], current_state=CurrentState()),
                "not_empty",
                {},
            ),
            # Populated current_state
            (
                TrackerContext(
                    conversation_turns=[],
                    current_state=CurrentState(
                        latest_message="Hello",
                        active_flow="greeting_flow",
                        flow_stack=[{"flow_name": "main_flow"}],
                        slots={"user_name": "John", "session_id": "123"},
                        latest_action="action_greet",
                        followup_action="utter_welcome",
                    ),
                ),
                "not_empty",
                {
                    "latest_message": "Hello",
                    "active_flow": "greeting_flow",
                    "flow_stack": [{"flow_name": "main_flow"}],
                    "slots": {"user_name": "John", "session_id": "123"},
                    "latest_action": "action_greet",
                    "followup_action": "utter_welcome",
                },
            ),
        ],
    )
    def test_format_current_state(
        self,
        tracker_context: Optional[TrackerContext],
        expected_result: Optional[str],
        expected_fields: Dict[str, Any],
    ):
        """Test _format_current_state method with TrackerContext."""
        result = Copilot._format_current_state(tracker_context)

        if expected_result == "{}":
            assert result == expected_result
        else:
            parsed_result = json.loads(result)
            for field, expected_value in expected_fields.items():
                assert parsed_result[field] == expected_value


class TestCopilotPromptRendering:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "context,"
        "documents,"
        "tracker_events,"
        "expected_content_parts,"
        "excluded_content_parts",
        [
            # Minimal context - test without any context
            (
                CopilotContext(
                    tracker_context=None,
                    assistant_logs="",
                    assistant_files={},
                    copilot_chat_history=[],
                ),
                [],  # No documents
                [],  # No tracker events
                [
                    "Assistant files are not available",
                ],
                [
                    "## Assistant Logs",
                    "## Attachments",
                ],
            ),
            # Populated context
            (
                CopilotContext(
                    tracker_context=TrackerContext(
                        conversation_turns=[],
                        current_state=CurrentState(
                            latest_message="Hello",
                            active_flow="greeting_flow",
                            slots={"user_name": "John"},
                        ),
                    ),
                    assistant_logs="Some training logs",
                    assistant_files={"domain.yml": "version: '3.1'"},
                    copilot_chat_history=[],
                ),
                [
                    Document(
                        content="Rasa documentation content",
                        url="https://docs.rasa.com/test",
                        title="Test Documentation",
                    )
                ],
                [
                    EventContent(
                        type="event",
                        event="user_uttered",
                        event_data={"text": "Hello", "intent": {"name": "greet"}},
                    )
                ],
                [
                    "Some training logs",
                    "domain.yml",
                    "Rasa documentation content",
                    "user_uttered",
                    "Hello",
                ],
                [],
            ),
        ],
    )
    async def test_render_last_user_message_context_prompt(
        self,
        context: CopilotContext,
        documents: List[Document],
        tracker_events: List[EventContent],
        expected_content_parts: List[str],
        excluded_content_parts: List[str],
    ):
        """Test _render_last_user_message_context_prompt with context."""
        copilot = Copilot()
        result = copilot._render_last_user_message_context_prompt(
            context, documents, tracker_events
        )
        for expected_text in expected_content_parts:
            assert expected_text in result
        for excluded_text in excluded_content_parts:
            assert excluded_text not in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "internal_message,documents,expected_content_parts",
        [
            # Minimal internal message
            (
                InternalCopilotRequestChatMessage(
                    role="internal_copilot_request",
                    content=[],
                    response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                ),
                [],
                [
                    "No modified assistant project files",
                    "No assistant logs available",
                ],
            ),
            # Populated internal message
            (
                InternalCopilotRequestChatMessage(
                    role="internal_copilot_request",
                    content=[
                        LogContent(
                            type="log", content="YAML syntax error in domain.yml"
                        ),
                        FileContent(
                            type="file",
                            file_path="domain.yml",
                            file_content="version: '3.1'\nintents:\n  - greet",
                        ),
                        FileContent(
                            type="file",
                            file_path="flows.yml",
                            file_content=(
                                "flows:\n  greeting_flow:\n    steps:\n      - action: utter_greet"  # noqa: E501
                            ),
                        ),
                    ],
                    response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                ),
                [
                    Document(
                        content="Rasa domain configuration guide",
                        url="https://docs.rasa.com/domain",
                        title="Domain Configuration",
                    )
                ],
                [
                    "YAML syntax error in domain.yml",
                    "domain.yml",
                    "flows.yml",
                    "Rasa domain configuration guide",
                ],
            ),
        ],
    )
    async def test_render_training_error_handler_prompt(
        self,
        internal_message: InternalCopilotRequestChatMessage,
        documents: List[Document],
        expected_content_parts: List[str],
    ):
        """Test _render_training_error_handler_prompt with various scenarios."""
        copilot = Copilot()
        result = copilot._render_training_error_handler_prompt(
            internal_message, documents
        )

        for expected_text in expected_content_parts:
            assert expected_text in result
