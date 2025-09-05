import json
from typing import Any, Dict, List

import pytest

from rasa.builder.copilot.copilot import Copilot
from rasa.builder.copilot.models import (
    CopilotChatMessage,
    CopilotContext,
    TextContent,
)
from rasa.builder.shared.tracker_context import (
    AssistantConversationTurn,
    AssistantMessage,
    CurrentState,
    TrackerContext,
    TrackerEvent,
    UserMessage,
)
from rasa.core.agent import Agent
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


def sample_chat_history() -> list[CopilotChatMessage]:
    """Create sample chat history."""
    return [
        CopilotChatMessage(
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
                CopilotChatMessage(
                    role="user",
                    content=[TextContent(type="text", text="Hello")],
                    response_category=None,
                ),
            ],
            [
                {
                    "role": "system",
                    "content": "[SYSTEM_PROMPT_PLACEHOLDER]",
                },  # System prompt (content varies)
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "[LAST_USER_MESSAGE_CONTEXT_PROMPT_PLACEHOLDER]",
                        },
                        {"type": "text", "text": "Hello"},
                    ],
                },  # Context + user message
            ],
        ),
        # Two messages
        (
            [
                CopilotChatMessage(
                    role="user",
                    content=[TextContent(type="text", text="Hello")],
                    response_category=None,
                ),
                CopilotChatMessage(
                    role="copilot",
                    content=[TextContent(type="text", text="Hi there!")],
                    response_category=None,
                ),
            ],
            [
                {
                    "role": "system",
                    "content": "[SYSTEM_PROMPT_PLACEHOLDER]",
                },  # System prompt
                {
                    "role": "user",
                    "content": "Hello",
                },  # Chat history (only first message)
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "[LAST_USER_MESSAGE_CONTEXT_PROMPT_PLACEHOLDER]",
                        },
                        {"type": "text", "text": "Hi there!"},
                    ],
                },  # Last message with context
            ],
        ),
        # Multiple messages
        (
            [
                CopilotChatMessage(
                    role="user",
                    content=[TextContent(type="text", text="Hello")],
                    response_category=None,
                ),
                CopilotChatMessage(
                    role="copilot",
                    content=[TextContent(type="text", text="Hi there!")],
                    response_category=None,
                ),
                CopilotChatMessage(
                    role="user",
                    content=[
                        TextContent(
                            type="text", text="How do I create a custom action?"
                        )
                    ],
                    response_category=None,
                ),
            ],
            [
                {
                    "role": "system",
                    "content": "[SYSTEM_PROMPT_PLACEHOLDER]",
                },  # System prompt
                {"role": "user", "content": "Hello"},  # Chat history (first message)
                {
                    "role": "assistant",
                    "content": "Hi there!",
                },  # Chat history (second message)
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "[LAST_USER_MESSAGE_CONTEXT_PROMPT_PLACEHOLDER]",
                        },
                        {"type": "text", "text": "How do I create a custom action?"},
                    ],
                },  # Last message with context
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_build_messages_openai_format(
    chat_history: List[CopilotChatMessage],
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
