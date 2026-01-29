from typing import Any, Dict, List

import pytest

from rasa.builder.shared.tracker_context import (
    AssistantConversationTurn,
    AssistantMessage,
    CurrentState,
    TrackerContext,
    TrackerEvent,
    UserMessage,
)
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    Event,
    FlowCompleted,
    FlowStarted,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.mark.parametrize(
    "events,expected_conversation_turns,expected_current_state",
    [
        # Test single conversation turn that starts with a user message
        (
            [
                UserUttered("hello"),
                BotUttered("Hi there!"),
            ],
            [
                AssistantConversationTurn(
                    user_message=UserMessage(text="hello"),
                    assistant_messages=[AssistantMessage(text="Hi there!")],
                ),
            ],
            CurrentState(
                latest_message="hello",
                active_flow=None,
                flow_stack=[],
                slots=None,
                latest_action=None,
            ),
        ),
        # Test basic conversation that starts with a user message
        (
            [
                UserUttered("hello"),
                BotUttered("Hi there!"),
                UserUttered("how are you?"),
                BotUttered("Good, thanks!"),
            ],
            [
                AssistantConversationTurn(
                    user_message=UserMessage(text="hello"),
                    assistant_messages=[AssistantMessage(text="Hi there!")],
                ),
                AssistantConversationTurn(
                    user_message=UserMessage(text="how are you?"),
                    assistant_messages=[AssistantMessage(text="Good, thanks!")],
                ),
            ],
            CurrentState(
                latest_message="how are you?",
                active_flow=None,
                flow_stack=[],
                slots=None,
                latest_action=None,
                followup_action=None,
            ),
        ),
        # Test single conversation turn that starts with an assistant message
        (
            [
                BotUttered("Hi there!"),
            ],
            [
                AssistantConversationTurn(
                    assistant_messages=[AssistantMessage(text="Hi there!")],
                ),
            ],
            CurrentState(
                latest_message=None,
                active_flow=None,
                flow_stack=[],
                slots=None,
                followup_action="action_listen",
            ),
        ),
        # Test basic conversation that starts with a bot message and ends with the user
        # message.
        (
            [
                BotUttered("Hello, I'm your personal assistant"),
                UserUttered("Hello, I'm John"),
                BotUttered("Nice to meet you, John!"),
                UserUttered("I want to transfer money."),
                BotUttered("Who do you want to transfer money to?"),
                UserUttered("Anna."),
            ],
            [
                AssistantConversationTurn(
                    user_message=None,
                    assistant_messages=[
                        AssistantMessage(text="Hello, I'm your personal assistant")
                    ],
                ),
                AssistantConversationTurn(
                    user_message=UserMessage(text="Hello, I'm John"),
                    assistant_messages=[
                        AssistantMessage(text="Nice to meet you, John!")
                    ],
                ),
                AssistantConversationTurn(
                    user_message=UserMessage(text="I want to transfer money."),
                    assistant_messages=[
                        AssistantMessage(text="Who do you want to transfer money to?")
                    ],
                ),
                AssistantConversationTurn(
                    user_message=UserMessage(text="Anna."), assistant_messages=[]
                ),
            ],
            CurrentState(
                latest_message="Anna.",
                active_flow=None,
                flow_stack=[],
                slots=None,
                latest_action=None,
                followup_action=None,
            ),
        ),
        # Test complex conversation with predicted commands and context events
        # (ActionExecuted, SlotSet, FlowStarted, FlowCompleted)
        (
            [
                BotUttered("Welcome!"),
                ActionExecuted("action_listen", confidence=1.0),
                UserUttered("Hello."),
                FlowStarted("greeting_flow"),
                ActionExecuted("utter_greet", confidence=1.0),
                BotUttered("Hi! How can I help?"),
                ActionExecuted("action_listen", confidence=1.0),
                FlowCompleted("greeting_flow", "greeting_flow_step"),
                UserUttered("I want to transfer money to my friend."),
                FlowStarted("transfer_money"),
                FlowStarted("pattern_collect_information"),
                ActionExecuted("utter_ask_recipient", confidence=1.0),
                BotUttered("Who do you want to transfer money to?"),
                ActionExecuted("action_listen", confidence=1.0),
                UserUttered("I want to transfer money to Anna."),
                SlotSet("recipient", "Anna"),
                FlowCompleted(
                    "pattern_collect_information",
                    step_id="pattern_collect_information_step",
                ),
                FlowStarted("pattern_collect_information"),
                ActionExecuted("utter_ask_amount", confidence=1.0),
                BotUttered("How much would you like to transfer?"),
                ActionExecuted("action_listen", confidence=1.0),
                UserUttered("I want to transfer 100 USD"),
                SlotSet("amount", 100),
                FlowCompleted(
                    "pattern_collect_information",
                    step_id="pattern_collect_information_step",
                ),
                ActionExecuted("utter_confirm_transfer", confidence=1.0),
                ActionExecuted("action_check_transfer_funds", confidence=1.0),
                FlowStarted("pattern_collect_information"),
                BotUttered("I'll transfer $100 for you. Is this correct?"),
                ActionExecuted("action_listen", confidence=1.0),
                UserUttered("Yes, that's correct"),
            ],
            [
                AssistantConversationTurn(
                    user_message=None,
                    assistant_messages=[AssistantMessage(text="Welcome!")],
                    context_events=[
                        TrackerEvent(
                            event="action",
                            data={"action_name": "action_listen", "confidence": 1.0},
                        )
                    ],
                ),
                AssistantConversationTurn(
                    user_message=UserMessage(text="Hello."),
                    assistant_messages=[AssistantMessage(text="Hi! How can I help?")],
                    context_events=[
                        TrackerEvent(
                            event="flow_started", data={"flow_id": "greeting_flow"}
                        ),
                        TrackerEvent(
                            event="action",
                            data={"action_name": "utter_greet", "confidence": 1.0},
                        ),
                        TrackerEvent(
                            event="action",
                            data={"action_name": "action_listen", "confidence": 1.0},
                        ),
                        TrackerEvent(
                            event="flow_completed", data={"flow_id": "greeting_flow"}
                        ),
                    ],
                ),
                AssistantConversationTurn(
                    user_message=UserMessage(
                        text="I want to transfer money to my friend."
                    ),
                    assistant_messages=[
                        AssistantMessage(text="Who do you want to transfer money to?")
                    ],
                    context_events=[
                        TrackerEvent(
                            event="flow_started", data={"flow_id": "transfer_money"}
                        ),
                        TrackerEvent(
                            event="flow_started",
                            data={"flow_id": "pattern_collect_information"},
                        ),
                        TrackerEvent(
                            event="action",
                            data={
                                "action_name": "utter_ask_recipient",
                                "confidence": 1.0,
                            },
                        ),
                        TrackerEvent(
                            event="action",
                            data={"action_name": "action_listen", "confidence": 1.0},
                        ),
                    ],
                ),
                AssistantConversationTurn(
                    user_message=UserMessage(text="I want to transfer money to Anna."),
                    assistant_messages=[
                        AssistantMessage(text="How much would you like to transfer?")
                    ],
                    context_events=[
                        TrackerEvent(
                            event="slot",
                            data={"slot_name": "recipient", "slot_value": "Anna"},
                        ),
                        TrackerEvent(
                            event="flow_completed",
                            data={"flow_id": "pattern_collect_information"},
                        ),
                        TrackerEvent(
                            event="flow_started",
                            data={"flow_id": "pattern_collect_information"},
                        ),
                        TrackerEvent(
                            event="action",
                            data={"action_name": "utter_ask_amount", "confidence": 1.0},
                        ),
                        TrackerEvent(
                            event="action",
                            data={"action_name": "action_listen", "confidence": 1.0},
                        ),
                    ],
                ),
                AssistantConversationTurn(
                    user_message=UserMessage(text="I want to transfer 100 USD"),
                    assistant_messages=[
                        AssistantMessage(
                            text="I'll transfer $100 for you. Is this correct?"
                        )
                    ],
                    context_events=[
                        TrackerEvent(
                            event="slot",
                            data={"slot_name": "amount", "slot_value": 100},
                        ),
                        TrackerEvent(
                            event="flow_completed",
                            data={"flow_id": "pattern_collect_information"},
                        ),
                        TrackerEvent(
                            event="action",
                            data={
                                "action_name": "utter_confirm_transfer",
                                "confidence": 1.0,
                            },
                        ),
                        TrackerEvent(
                            event="action",
                            data={
                                "action_name": "action_check_transfer_funds",
                                "confidence": 1.0,
                            },
                        ),
                        TrackerEvent(
                            event="flow_started",
                            data={"flow_id": "pattern_collect_information"},
                        ),
                        TrackerEvent(
                            event="action",
                            data={"action_name": "action_listen", "confidence": 1.0},
                        ),
                    ],
                ),
                AssistantConversationTurn(
                    user_message=UserMessage(text="Yes, that's correct"),
                    assistant_messages=[],
                    context_events=[],
                ),
            ],
            CurrentState(
                latest_message="Yes, that's correct",
                active_flow=None,
                flow_stack=[],
                slots={"recipient": "Anna", "amount": "100"},
                latest_action="action_listen",
                followup_action=None,
            ),
        ),
        # Test empty tracker
        (
            [],
            [],
            CurrentState(
                latest_message=None,
                active_flow=None,
                flow_stack=None,
                slots=None,
                latest_action=None,
                followup_action=None,
            ),
        ),
    ],
)
def test_tracker_context_from_tracker(
    events: List[Event],
    expected_conversation_turns: List[AssistantConversationTurn],
    expected_current_state: CurrentState,
) -> None:
    """Test that TrackerContext is built correctly from a tracker."""
    # Given
    tracker = DialogueStateTracker.from_events("test_session", events)

    # When
    tracker_context = TrackerContext.from_tracker(tracker)

    # Then - If no events, tracker_context should be None
    if not events:
        assert tracker_context is None
        return

    # Then - Tracker not empty
    assert tracker_context is not None

    # Test conversation turns
    assert len(tracker_context.conversation_turns) == len(expected_conversation_turns)

    for i, (actual_turn, expected_turn) in enumerate(
        zip(tracker_context.conversation_turns, expected_conversation_turns)
    ):
        # Test user message
        if expected_turn.user_message:
            assert actual_turn.user_message is not None
            assert actual_turn.user_message.text == expected_turn.user_message.text
            assert (
                actual_turn.user_message.predicted_commands
                == expected_turn.user_message.predicted_commands
            )
        else:
            assert actual_turn.user_message is None

        # Test assistant messages
        assert len(actual_turn.assistant_messages) == len(
            expected_turn.assistant_messages
        )
        for j, expected_message in enumerate(expected_turn.assistant_messages):
            assert actual_turn.assistant_messages[j].text == expected_message.text

        # Test context events
        assert len(actual_turn.context_events) == len(expected_turn.context_events)
        for j, expected_event in enumerate(expected_turn.context_events):
            assert actual_turn.context_events[j].event == expected_event.event
            assert actual_turn.context_events[j].data == expected_event.data

    # Test current state
    current_state = tracker_context.current_state
    assert current_state.latest_message == expected_current_state.latest_message
    assert current_state.active_flow == expected_current_state.active_flow
    assert current_state.flow_stack == expected_current_state.flow_stack
    assert current_state.latest_action == expected_current_state.latest_action
    assert current_state.followup_action == expected_current_state.followup_action

    # Test slots (handle None case)
    if expected_current_state.slots is None:
        assert current_state.slots is None
    else:
        assert current_state.slots == expected_current_state.slots


@pytest.mark.parametrize(
    "events,expected_openai_format,max_turns",
    [
        # Test basic conversation that starts with a user message
        (
            [
                UserUttered("message 1"),
                BotUttered("message 2"),
                UserUttered("message 3"),
                BotUttered("message 4"),
            ],
            [
                {"role": "user", "content": "message 1"},
                {"role": "assistant", "content": "message 2"},
                {"role": "user", "content": "message 3"},
                {"role": "assistant", "content": "message 4"},
            ],
            10,
        ),
        # Test conversation that starts with a bot message
        (
            [
                BotUttered("message 1"),
                UserUttered("message 2"),
                BotUttered("message 3"),
                UserUttered("message 4"),
            ],
            [
                {"role": "assistant", "content": "message 1"},
                {"role": "user", "content": "message 2"},
                {"role": "assistant", "content": "message 3"},
                {"role": "user", "content": "message 4"},
            ],
            10,
        ),
        # Test conversation with multiple assistant messages
        (
            [
                BotUttered("message 1"),
                BotUttered("message 2"),
                BotUttered("message 3"),
                UserUttered("message 4"),
                BotUttered("message 5"),
                UserUttered("message 6"),
                BotUttered("message 7"),
                BotUttered("message 8"),
                BotUttered("message 9"),
            ],
            [
                {"role": "assistant", "content": "message 1"},
                {"role": "assistant", "content": "message 2"},
                {"role": "assistant", "content": "message 3"},
                {"role": "user", "content": "message 4"},
                {"role": "assistant", "content": "message 5"},
                {"role": "user", "content": "message 6"},
                {"role": "assistant", "content": "message 7"},
                {"role": "assistant", "content": "message 8"},
                {"role": "assistant", "content": "message 9"},
            ],
            10,
        ),
        # Test conversation with multiple assistant messages (only the last two turns)
        (
            [
                # Turn 1
                BotUttered("message 1"),
                BotUttered("message 2"),
                # Turn 2
                UserUttered("message 4"),
                BotUttered("message 5"),
                # Turn 3
                UserUttered("message 6"),
                BotUttered("message 7"),
                BotUttered("message 8"),
                # Turn 4
                UserUttered("message 9"),
                BotUttered("message 10"),
            ],
            [
                {"role": "user", "content": "message 6"},
                {"role": "assistant", "content": "message 7"},
                {"role": "assistant", "content": "message 8"},
                {"role": "user", "content": "message 9"},
                {"role": "assistant", "content": "message 10"},
            ],
            2,
        ),
    ],
)
def test_tracker_context_to_openai_format(
    events: List[Event],
    expected_openai_format: List[Dict[str, str]],
    max_turns: int,
) -> None:
    """Test that TrackerContext converts to OpenAI format correctly."""
    # Given

    tracker = DialogueStateTracker.from_events("test_session", events)
    tracker_context = TrackerContext.from_tracker(tracker, max_turns=max_turns)

    # When
    openai_format = tracker_context.to_openai_format()

    # Then
    assert openai_format == expected_openai_format


def test_tracker_context_formatted_conversation_turns() -> None:
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

    tracker_context = TrackerContext(
        conversation_turns=conversation_turns,
        current_state=CurrentState(),
    )

    expected_turns: List[Dict[str, Any]] = [
        {
            "turn_id": 1,
            "USER": {
                "text": "I want to transfer money",
                "predicted_commands": ["start flow", "set slot"],
            },
            "ASSISTANT": [{"text": "How much would you like to transfer?"}],
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
        }
    ]

    assert tracker_context.formatted_conversation_turns == expected_turns


def test_tracker_context_formatted_conversation_turns_empty() -> None:
    tracker_context = TrackerContext(
        conversation_turns=[],
        current_state=CurrentState(),
    )
    assert tracker_context.formatted_conversation_turns == []
