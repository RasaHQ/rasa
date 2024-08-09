from typing import Any, Dict, List

import pytest

from rasa.dialogue_understanding.patterns.clarify import FLOW_PATTERN_CLARIFICATION
from rasa.e2e_test.assertions import (
    ActionExecutedAssertion,
    AssertedButton,
    AssertedSlot,
    Assertion,
    AssertionFailure,
    BotUtteredAssertion,
    FlowCancelledAssertion,
    FlowCompletedAssertion,
    FlowStartedAssertion,
    GenerativeResponseIsGroundedAssertion,
    GenerativeResponseIsRelevantAssertion,
    InvalidAssertionType,
    PatternClarificationContainsAssertion,
    SlotWasNotSetAssertion,
    SlotWasSetAssertion,
)
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    Event,
    FlowCancelled,
    FlowCompleted,
    FlowStarted,
    SlotSet,
)
from rasa.shared.exceptions import RasaException


@pytest.mark.parametrize(
    "data, expected_assertion",
    [
        (
            {"flow_started": "transfer_money"},
            FlowStartedAssertion(flow_id="transfer_money"),
        ),
        (
            {
                "flow_completed": {
                    "flow_id": "transfer_money",
                    "flow_step_id": "utter_confirm_transfer",
                }
            },
            FlowCompletedAssertion(
                flow_id="transfer_money", flow_step_id="utter_confirm_transfer"
            ),
        ),
        (
            {
                "flow_cancelled": {
                    "flow_id": "transfer_money",
                    "flow_step_id": "utter_ask_confirmation",
                }
            },
            FlowCancelledAssertion(
                flow_id="transfer_money", flow_step_id="utter_ask_confirmation"
            ),
        ),
        (
            {
                "pattern_clarification_contains": [
                    "list_contacts",
                    "add_contacts",
                    "remove_contacts",
                ]
            },
            PatternClarificationContainsAssertion(
                flow_ids={"list_contacts", "add_contacts", "remove_contacts"}
            ),
        ),
        (
            {"action_executed": "action_session_start"},
            ActionExecutedAssertion(action_name="action_session_start"),
        ),
        (
            {"slot_was_set": [{"name": "name", "value": "John"}]},
            SlotWasSetAssertion(slots=[AssertedSlot(name="name", value="John")]),
        ),
        (
            {"slot_was_not_set": [{"name": "name"}]},
            SlotWasNotSetAssertion(
                slots=[AssertedSlot(name="name", value="value key is undefined")]
            ),
        ),
        (
            {
                "bot_uttered": {
                    "utter_name": "utter_options",
                    "buttons": [
                        {"title": "Transfer Money", "payload": "/transfer_money"}
                    ],
                    "text_matches": "You can transfer money or check your balance.",
                }
            },
            BotUtteredAssertion(
                utter_name="utter_options",
                buttons=[
                    AssertedButton(title="Transfer Money", payload="/transfer_money")
                ],
                text_matches="You can transfer money or check your balance.",
            ),
        ),
        (
            {
                "generative_response_is_relevant": {
                    "threshold": 0.9,
                    "utter_name": "utter_options",
                    "ground_truth": "You can transfer money or check your balance.",
                }
            },
            GenerativeResponseIsRelevantAssertion(
                threshold=0.9,
                utter_name="utter_options",
                ground_truth="You can transfer money or check your balance.",
            ),
        ),
        (
            {
                "generative_response_is_grounded": {
                    "threshold": 0.88,
                    "utter_name": "utter_fee",
                    "ground_truth": "The fee for transferring money is $5.",
                }
            },
            GenerativeResponseIsGroundedAssertion(
                threshold=0.88,
                utter_name="utter_fee",
                ground_truth="The fee for transferring money is $5.",
            ),
        ),
    ],
)
def test_create_typed_assertion_valid_subclasses(
    data: Dict[str, Any], expected_assertion: Assertion
):
    assert Assertion.create_typed_assertion(data) == expected_assertion


def test_create_typed_assertion_with_unknown_type():
    with pytest.raises(
        InvalidAssertionType, match="Invalid assertion type 'llm_commands'."
    ):
        Assertion.create_typed_assertion({"llm_commands": "unknown"})


def test_empty_bot_uttered_raises_exception():
    with pytest.raises(RasaException, match="A 'bot_uttered' assertion is empty, "):
        Assertion.create_typed_assertion({"bot_uttered": {}})


@pytest.mark.parametrize(
    "assertion, turn_events",
    [
        (
            FlowStartedAssertion(flow_id="transfer_money"),
            [FlowStarted(flow_id="transfer_money")],
        ),
        (
            FlowCompletedAssertion(
                flow_id="transfer_money", flow_step_id="utter_confirm_transfer"
            ),
            [FlowCompleted(flow_id="transfer_money", step_id="utter_confirm_transfer")],
        ),
        (
            FlowCancelledAssertion(
                flow_id="transfer_money", flow_step_id="utter_ask_confirmation"
            ),
            [FlowCancelled(flow_id="transfer_money", step_id="utter_ask_confirmation")],
        ),
        (
            PatternClarificationContainsAssertion(
                flow_ids={"list_contacts", "add_contacts", "remove_contacts"}
            ),
            [
                FlowStarted(
                    flow_id=FLOW_PATTERN_CLARIFICATION,
                    metadata={
                        "names": ["list_contacts", "add_contacts", "remove_contacts"]
                    },
                )
            ],
        ),
        (
            ActionExecutedAssertion(action_name="action_session_start"),
            [ActionExecuted(action_name="action_session_start")],
        ),
        (
            SlotWasSetAssertion(slots=[AssertedSlot(name="name", value="John")]),
            [SlotSet(key="name", value="John")],
        ),
        (
            SlotWasSetAssertion(
                slots=[AssertedSlot(name="name", value="value key is undefined")]
            ),
            [SlotSet(key="name", value="John")],
        ),
        (
            BotUtteredAssertion(
                text_matches="You can transfer money or check your balance."
            ),
            [BotUttered(text="You can transfer money or check your balance.")],
        ),
        (
            BotUtteredAssertion(
                utter_name="utter_options",
            ),
            [BotUttered(metadata={"utter_action": "utter_options"})],
        ),
        (
            BotUtteredAssertion(
                buttons=[
                    AssertedButton(title="Transfer Money", payload="/transfer_money")
                ]
            ),
            [
                BotUttered(
                    data={
                        "buttons": [
                            {"title": "Transfer Money", "payload": "/transfer_money"}
                        ]
                    }
                )
            ],
        ),
    ],
)
def test_assertion_run_returns_no_assertion_failure(
    assertion: Assertion, turn_events: List[Event]
) -> None:
    assertion_failure, matching_event = assertion.run(turn_events, [])
    assert assertion_failure is None
    assert matching_event == turn_events[0]


def test_slot_was_not_set_assertion_returns_no_assertion_failure() -> None:
    assertion = SlotWasNotSetAssertion(
        slots=[AssertedSlot(name="name", value="value key is undefined")]
    )
    turn_events = []
    assertion_failure, matching_event = assertion.run(turn_events, [])
    assert assertion_failure is None
    assert matching_event is None


@pytest.mark.parametrize(
    "assertion, expected_assertion_failure",
    [
        (
            FlowStartedAssertion(flow_id="transfer_money"),
            AssertionFailure(
                assertion=FlowStartedAssertion(flow_id="transfer_money", line=None),
                error_message="Flow with id 'transfer_money' did not start.",
                actual_events_transcript=[],
                error_line=None,
            ),
        ),
        (
            FlowCompletedAssertion(
                flow_id="transfer_money", flow_step_id="utter_confirm_transfer"
            ),
            AssertionFailure(
                assertion=FlowCompletedAssertion(
                    flow_id="transfer_money",
                    flow_step_id="utter_confirm_transfer",
                    line=None,
                ),
                error_message="Flow with id 'transfer_money' did not complete.",
                actual_events_transcript=[],
                error_line=None,
            ),
        ),
        (
            FlowCancelledAssertion(
                flow_id="transfer_money", flow_step_id="utter_ask_confirmation"
            ),
            AssertionFailure(
                assertion=FlowCancelledAssertion(
                    flow_id="transfer_money",
                    flow_step_id="utter_ask_confirmation",
                    line=None,
                ),
                error_message="Flow with id 'transfer_money' was not cancelled.",
                actual_events_transcript=[],
                error_line=None,
            ),
        ),
        (
            PatternClarificationContainsAssertion(
                flow_ids={"list_contacts", "add_contacts", "remove_contacts"}
            ),
            AssertionFailure(
                assertion=PatternClarificationContainsAssertion(
                    flow_ids={"list_contacts", "add_contacts", "remove_contacts"},
                    line=None,
                ),
                error_message="'pattern_clarification' pattern did not " "trigger.",
                actual_events_transcript=[],
                error_line=None,
            ),
        ),
        (
            ActionExecutedAssertion(action_name="action_session_start"),
            AssertionFailure(
                assertion=ActionExecutedAssertion(
                    action_name="action_session_start", line=None
                ),
                error_message="Action 'action_session_start' did not execute.",
                actual_events_transcript=[],
                error_line=None,
            ),
        ),
        (
            SlotWasSetAssertion(slots=[AssertedSlot(name="name", value="John")]),
            AssertionFailure(
                assertion=SlotWasSetAssertion(
                    slots=[AssertedSlot(name="name", value="John", line=None)]
                ),
                error_message="Slot 'name' was not set.",
                actual_events_transcript=[],
                error_line=None,
            ),
        ),
        (
            BotUtteredAssertion(
                text_matches="You can transfer money or check your balance."
            ),
            AssertionFailure(
                assertion=BotUtteredAssertion(
                    utter_name=None,
                    text_matches="You can transfer " "money or check " "your balance.",
                    buttons=None,
                    line=None,
                ),
                error_message="Bot did not utter any response which matches "
                "the provided text pattern 'You can transfer "
                "money or check your balance.'.",
                actual_events_transcript=[],
                error_line=None,
            ),
        ),
        (
            BotUtteredAssertion(
                utter_name="utter_options",
            ),
            AssertionFailure(
                assertion=BotUtteredAssertion(
                    utter_name="utter_options",
                    text_matches=None,
                    buttons=None,
                    line=None,
                ),
                error_message="Bot did not utter 'utter_options' response.",
                actual_events_transcript=[],
                error_line=None,
            ),
        ),
        (
            BotUtteredAssertion(
                buttons=[
                    AssertedButton(title="Transfer Money", payload="/transfer_money")
                ]
            ),
            AssertionFailure(
                assertion=BotUtteredAssertion(
                    utter_name=None,
                    text_matches=None,
                    buttons=[
                        AssertedButton(
                            title="Transfer " "Money", payload="/transfer_money"
                        )
                    ],
                    line=None,
                ),
                error_message="Bot did not utter any response with the "
                "expected buttons.",
                actual_events_transcript=[],
                error_line=None,
            ),
        ),
    ],
)
def test_assertion_run_returns_assertion_failure(
    assertion: Assertion, expected_assertion_failure: AssertionFailure
) -> None:
    assertion_failure, matching_event = assertion.run([], [])
    assert assertion_failure == expected_assertion_failure
    assert matching_event is None


@pytest.mark.parametrize(
    "assertion, expected_assertion_failure",
    [
        (
            SlotWasNotSetAssertion(
                slots=[AssertedSlot(name="address", value="value key is undefined")]
            ),
            AssertionFailure(
                assertion=SlotWasNotSetAssertion(
                    slots=[
                        AssertedSlot(
                            name="address", value="value key is undefined", line=None
                        )
                    ]
                ),
                error_message="Slot 'address' was set to '13 Pine Road' "
                "but it should not have been set.",
                actual_events_transcript=["SlotSet(key: address, value: 13 Pine Road)"],
                error_line=None,
            ),
        ),
        (
            SlotWasNotSetAssertion(
                slots=[AssertedSlot(name="address", value="13 Pine Road")]
            ),
            AssertionFailure(
                assertion=SlotWasNotSetAssertion(
                    slots=[
                        AssertedSlot(name="address", value="13 Pine Road", line=None)
                    ]
                ),
                error_message="Slot 'address' was set to '13 Pine Road' "
                "but it should not have been set.",
                actual_events_transcript=["SlotSet(key: address, value: 13 Pine Road)"],
            ),
        ),
    ],
)
def test_slot_was_not_set_assertions_returns_assertion_failure(
    assertion: SlotWasNotSetAssertion, expected_assertion_failure: AssertionFailure
) -> None:
    assertion_failure, matching_event = assertion.run(
        [SlotSet(key="address", value="13 Pine Road")], []
    )
    assert assertion_failure == expected_assertion_failure
    assert matching_event is None


@pytest.mark.parametrize(
    "assertion, expected_assertion_dict",
    [
        (
            FlowStartedAssertion(flow_id="transfer_money"),
            {"flow_id": "transfer_money", "type": "flow_started", "line": None},
        ),
        (
            FlowCompletedAssertion(
                flow_id="transfer_money", flow_step_id="utter_confirm_transfer"
            ),
            {
                "flow_id": "transfer_money",
                "flow_step_id": "utter_confirm_transfer",
                "type": "flow_completed",
                "line": None,
            },
        ),
        (
            FlowCancelledAssertion(
                flow_id="transfer_money", flow_step_id="utter_ask_confirmation"
            ),
            {
                "flow_id": "transfer_money",
                "flow_step_id": "utter_ask_confirmation",
                "type": "flow_cancelled",
                "line": None,
            },
        ),
        (
            PatternClarificationContainsAssertion(
                flow_ids={"list_contacts", "add_contacts", "remove_contacts"}
            ),
            {
                "flow_ids": {"list_contacts", "add_contacts", "remove_contacts"},
                "type": "pattern_clarification_contains",
                "line": None,
            },
        ),
        (
            ActionExecutedAssertion(action_name="action_session_start"),
            {
                "action_name": "action_session_start",
                "type": "action_executed",
                "line": None,
            },
        ),
        (
            SlotWasSetAssertion(slots=[AssertedSlot(name="name", value="John")]),
            {
                "slots": [{"name": "name", "value": "John", "line": None}],
                "type": "slot_was_set",
            },
        ),
        (
            SlotWasNotSetAssertion(
                slots=[AssertedSlot(name="name", value="value key is undefined")]
            ),
            {
                "slots": [
                    {"name": "name", "value": "value key is undefined", "line": None}
                ],
                "type": "slot_was_not_set",
            },
        ),
        (
            BotUtteredAssertion(
                utter_name="utter_options",
                buttons=[
                    AssertedButton(title="Transfer Money", payload="/transfer_money")
                ],
                text_matches="You can transfer money or check your balance.",
            ),
            {
                "utter_name": "utter_options",
                "text_matches": "You can transfer money or check your balance.",
                "buttons": [{"title": "Transfer Money", "payload": "/transfer_money"}],
                "type": "bot_uttered",
                "line": None,
            },
        ),
        (
            GenerativeResponseIsRelevantAssertion(
                threshold=0.9,
                utter_name="utter_options",
                ground_truth="You can transfer money or check your balance.",
            ),
            {
                "threshold": 0.9,
                "utter_name": "utter_options",
                "ground_truth": "You can transfer money or check your balance.",
                "type": "generative_response_is_relevant",
                "line": None,
            },
        ),
        (
            GenerativeResponseIsGroundedAssertion(
                threshold=0.88,
                utter_name="utter_fee",
                ground_truth="The fee for transferring money is $5.",
            ),
            {
                "threshold": 0.88,
                "utter_name": "utter_fee",
                "ground_truth": "The fee for transferring money is $5.",
                "type": "generative_response_is_grounded",
                "line": None,
            },
        ),
    ],
)
def test_assertion_failure_as_dict(
    assertion: Assertion, expected_assertion_dict: Dict[str, Any]
) -> None:
    assertion_failure = AssertionFailure(
        assertion=assertion,
        error_message="Test error message",
        actual_events_transcript=["test_event"],
    )

    assert assertion_failure.as_dict() == {
        "assertion": expected_assertion_dict,
        "error_message": "Test error message",
        "actual_events_transcript": ["test_event"],
    }
