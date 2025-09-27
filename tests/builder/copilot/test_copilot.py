import json
from typing import Any, Dict, List

import pytest

from rasa.builder.copilot.constants import (
    ROLE_COPILOT,
    ROLE_COPILOT_INTERNAL,
    ROLE_SYSTEM,
)
from rasa.builder.copilot.copilot import Copilot
from rasa.builder.copilot.models import (
    ChatMessage,
    CopilotChatMessage,
    CopilotContext,
    CopilotSystemMessage,
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


def test_format_conversation_history():
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
                    event="action_executed", data={"action_name": "action_ask_amount"}
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


def test_format_conversation_history_empty():
    """Test the _format_conversation_history method with empty/None input."""
    from rasa.builder.copilot.copilot import Copilot

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
                            "text": "[LAST_USER_MESSAGE_CONTEXT_PROMPT_PLACEHOLDER]",
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
                            "text": "[LAST_USER_MESSAGE_CONTEXT_PROMPT_PLACEHOLDER]",
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


@pytest.mark.parametrize(
    "message, raises_error",
    [
        (
            CopilotChatMessage(
                role=ROLE_COPILOT,
                content=[TextContent(type="text", text="This is a copilot response")],
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
    message: ChatMessage, raises_error: bool
):
    """Test that _process_latest_message raises ValueError for unsupported types."""
    # Given
    copilot = Copilot()
    context = CopilotContext(
        copilot_chat_history=[],
        assistant_logs="",
        assistant_files={},
        tracker_context=None,
    )

    # When / Then
    if raises_error:
        with pytest.raises(ValueError):
            copilot._process_latest_message(message, context, [])
    else:
        copilot._process_latest_message(message, context, [])


def test_process_latest_message_internal_copilot_request_uses_training_error_prompt():
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
    result = copilot._process_latest_message(
        context.copilot_chat_history[0], context, mock_documents
    )

    # Then
    assert result["role"] == "user"
    assert "content" in result
    assert isinstance(result["content"], str)

    # Check that the content contains the training error prompt elements
    prompt_content = result["content"]
    assert "Training failed: Invalid YAML syntax" in prompt_content  # from log content
    assert "domain.yml" in prompt_content  # from file content
    assert "Rasa training documentation content" in prompt_content  # documentation
    assert "Relevant Documentation" in prompt_content  # documentation section


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
    chat_history: List[ChatMessage], expected_query: str
):
    """Test _create_documentation_search_query method with various chat histories."""
    # Given
    context = CopilotContext(
        tracker_context=None,
        assistant_logs="",
        assistant_files={},
        copilot_chat_history=chat_history,
    )

    # When
    result = Copilot._create_documentation_search_query(context)

    # Then
    assert result == expected_query
