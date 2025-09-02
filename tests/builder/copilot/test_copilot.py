import json
import os
from typing import Any, Dict, List

import pytest
from pytest import MonkeyPatch

from rasa.builder.copilot.copilot import Copilot
from rasa.builder.copilot.models import (
    CopilotChatMessage,
    CopilotContext,
    LogContent,
    ResponseCategory,
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


@pytest.mark.asyncio
async def test_llm_service_copilot_response(
    agent_with_flows: Agent, monkeypatch: MonkeyPatch
):
    os.environ["OPENAI_API_KEY"] = "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"
    os.environ["INKEEP_API_KEY"] = "CHANGEME"

    tracker = sample_tracker(agent_with_flows.domain)
    tracker_context = TrackerContext.from_tracker(tracker)

    context = CopilotContext(
        tracker_context=tracker_context,
        assistant_logs=sample_bot_logs(),
        assistant_files=await bot_files_for_agent(agent_with_flows),
        copilot_chat_history=sample_chat_history(),
    )

    expected_system_prompt = {"role": "system", "content": "TEST_SYSTEM_PROMPT"}

    async def _fake_create_system_message(self, context, relevant_documents):
        return expected_system_prompt

    monkeypatch.setattr(
        Copilot, "_create_system_message", _fake_create_system_message, raising=True
    )

    from rasa.builder.llm_service import llm_service

    stream, documents, system_prompt = await llm_service.copilot.generate_response(
        context
    )

    # Collect all tokens from the stream
    result = ""
    async for token in stream:
        result += token

    # Assert that we got some response
    assert result is not None
    assert len(result) > 0

    # Assert that documents were retrieved (even if empty)
    assert documents is not None

    # Assert that the system prompt returned is the one we patched in
    assert system_prompt == expected_system_prompt["content"]


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
    "chat_history,expected_formatted_messages",
    [
        # Regular chat - no internal messages
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
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": '[{"type": "text", "text": "Hi there!"}]',
                },
            ],
        ),
        # Chat with copilot_internal role and training_error_log_analysis
        (
            [
                CopilotChatMessage(
                    role="copilot_internal",
                    content=[
                        TextContent(
                            type="text",
                            text="The assistant training failed.",
                        ),
                        LogContent(
                            type="log",
                            content="ERROR: Invalid intent 'greet' in training data",
                            context="training session",
                        ),
                    ],
                    response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                ),
            ],
            [
                {
                    "role": "user",
                    "content": (
                        "[{"
                        '"type": "text", "text": "The assistant training failed."}, '
                        '{"type": "log", "content": "ERROR: Invalid intent \'greet\' '
                        'in training data", "context": "training session", '
                        '"metadata": {}}]'
                    ),
                }
            ],
        ),
    ],
)
def test_create_chat_history_messages(
    chat_history: List[CopilotChatMessage],
    expected_formatted_messages: List[Dict[str, Any]],
):
    context = CopilotContext(
        tracker_context=None,
        assistant_logs="",
        assistant_files={},
        copilot_chat_history=chat_history,
    )

    copilot = Copilot()
    formatted_messages = copilot._create_chat_history_messages(context)

    # Assert the number of messages
    assert len(formatted_messages) == len(expected_formatted_messages)

    # Assert each message matches the expected format
    for i, expected_message in enumerate(expected_formatted_messages):
        assert formatted_messages[i]["role"] == expected_message["role"]
        assert formatted_messages[i]["content"] == expected_message["content"]
