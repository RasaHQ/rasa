import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest import MonkeyPatch

from rasa.core.policies.enterprise_search_policy import SEARCH_RESULTS_METADATA_KEY
from rasa.dialogue_understanding.patterns.clarify import FLOW_PATTERN_CLARIFICATION
from rasa.e2e_test.assertions import (
    ActionExecutedAssertion,
    AssertedButton,
    AssertedSlot,
    Assertion,
    AssertionFailure,
    AssertionType,
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
from rasa.e2e_test.e2e_config import LLMJudgeConfig
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    Event,
    FlowCancelled,
    FlowCompleted,
    FlowStarted,
    SessionStarted,
    SlotSet,
    UserUttered,
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
                flow_names={"list_contacts", "add_contacts", "remove_contacts"}
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
    ],
)
def test_create_typed_assertion_valid_subclasses(
    data: Dict[str, Any], expected_assertion: Assertion
):
    assert Assertion.create_typed_assertion(data) == expected_assertion


@pytest.mark.parametrize(
    "data, assertion_type, metric_name",
    [
        (
            {
                "generative_response_is_relevant": {
                    "threshold": 0.9,
                    "utter_name": "utter_options",
                }
            },
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value,
            "answer_relevance",
        ),
        (
            {
                "generative_response_is_grounded": {
                    "threshold": 0.88,
                    "utter_name": "utter_fee",
                    "ground_truth": "The fee for transferring money is $5.",
                }
            },
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value,
            "answer_correctness",
        ),
    ],
)
def test_create_typed_assertion_valid_generative_assertions(
    monkeypatch: MonkeyPatch,
    data: Dict[str, Any],
    assertion_type: str,
    metric_name: str,
):
    mlflow_mock = MagicMock()
    sys.modules["mlflow"] = mlflow_mock
    mock_metric = MagicMock()
    monkeypatch.setattr(mlflow_mock, f"metrics.genai.{metric_name}", mock_metric)

    def get_expected_assertion(assertion_type: str) -> Assertion:
        import mlflow

        if assertion_type == AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value:
            return GenerativeResponseIsGroundedAssertion(
                threshold=0.88,
                utter_name="utter_fee",
                ground_truth="The fee for transferring money is $5.",
                metric_name="answer_correctness",
                metric_adjective="grounded",
                mlflow_metric=mlflow.metrics.genai.answer_correctness,
            )
        elif assertion_type == AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value:
            return GenerativeResponseIsRelevantAssertion(
                threshold=0.9,
                utter_name="utter_options",
                metric_name="answer_relevance",
                metric_adjective="relevant",
                mlflow_metric=mlflow.metrics.genai.answer_relevance,
            )

    assert Assertion.create_typed_assertion(data) == get_expected_assertion(
        assertion_type
    )


def test_create_typed_assertion_with_unknown_type():
    with pytest.raises(
        InvalidAssertionType, match="Invalid assertion type 'llm_commands'."
    ):
        Assertion.create_typed_assertion({"llm_commands": "unknown"})


def test_empty_bot_uttered_raises_exception():
    with pytest.raises(RasaException, match="A 'bot_uttered' assertion is empty, "):
        Assertion.create_typed_assertion({"bot_uttered": {}})


def test_pattern_clarification_contains_assertion_test():
    assertion = PatternClarificationContainsAssertion(
        flow_names={"add a card", "add a contact"}, line=12
    )
    try:
        assertion.__hash__()
    except TypeError:
        pytest.fail("Unexpected TypeError")


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
                flow_names={"list_contacts", "add_contacts", "remove_contacts"}
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
                flow_names={"list_contacts", "add_contacts", "remove_contacts"}
            ),
            AssertionFailure(
                assertion=PatternClarificationContainsAssertion(
                    flow_names={"list_contacts", "add_contacts", "remove_contacts"},
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
                flow_names={"list_contacts", "add_contacts", "remove_contacts"}
            ),
            {
                "flow_names": {"list_contacts", "add_contacts", "remove_contacts"},
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
            ),
            {
                "threshold": 0.9,
                "utter_name": "utter_options",
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


def set_up_tests_for_generative_response_assertions(
    monkeypatch: MonkeyPatch, table: Dict[str, Any], metric_name: str
) -> None:
    # we need to mock mlflow because it's an optional dependency of rasa-pro,
    # and it won't get installed in the CI
    mlflow_mock = MagicMock()
    sys.modules["mlflow"] = mlflow_mock
    mlflow_evaluate_mock = MagicMock()
    monkeypatch.setattr(mlflow_mock, "evaluate", mlflow_evaluate_mock)

    mock_result = MagicMock()
    monkeypatch.setattr(mock_result, "tables", table)
    mlflow_evaluate_mock.return_value = mock_result

    mock_metric = MagicMock()
    monkeypatch.setattr(mlflow_mock, f"metrics.genai.{metric_name}", mock_metric)


def get_assertion(assertion_type: str):
    if assertion_type == AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value:
        return GenerativeResponseIsGroundedAssertion.from_dict(
            {assertion_type: {"threshold": 0.85, "utter_name": "utter_free_transfers"}}
        )
    elif assertion_type == AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value:
        return GenerativeResponseIsRelevantAssertion.from_dict(
            {assertion_type: {"threshold": 0.85, "utter_name": "utter_free_transfers"}}
        )


@pytest.mark.parametrize(
    "assertion_type, data, metric_name",
    [
        (
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value,
            [
                {
                    "answer_relevance/v1/score": 5,
                    "answer_relevance/v1/justification": "test justification",
                }
            ],
            "answer_relevance",
        ),
        (
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value,
            [
                {
                    "answer_correctness/v1/score": 5,
                    "answer_correctness/v1/justification": "test justification",
                }
            ],
            "answer_correctness",
        ),
    ],
)
def test_generative_response_assertions_run_llm_evaluation_success(
    monkeypatch: MonkeyPatch,
    assertion_type: str,
    data: List[Dict[str, Any]],
    metric_name: str,
) -> None:
    table = {"eval_results_table": pd.DataFrame(data=data)}
    set_up_tests_for_generative_response_assertions(monkeypatch, table, metric_name)
    assertion = get_assertion(assertion_type)

    matching_event = BotUttered("Transfers are free for domestic service.")

    failure, event = assertion._run_llm_evaluation(
        matching_event,
        "Are transfers on free with this service?",
        LLMJudgeConfig(),
        "",
        [SessionStarted()],
        [UserUttered("Are transfers on free with this service?"), matching_event],
    )

    assert failure is None
    assert event == matching_event


def test_generative_response_is_relevant_run_llm_evaluation_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    table = {
        "eval_results_table": pd.DataFrame(
            data=[
                {
                    "answer_relevance/v1/score": 1,
                    "answer_relevance/v1/justification": "test justification",
                }
            ],
        )
    }
    set_up_tests_for_generative_response_assertions(
        monkeypatch, table, "answer_relevance"
    )
    assertion = get_assertion(AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value)

    matching_event = BotUttered("Transfers are free for domestic service.")
    prior_events = [SessionStarted()]
    turn_events = [
        UserUttered("Are transfers on free with this service?"),
        matching_event,
    ]
    failure, event = assertion._run_llm_evaluation(
        matching_event,
        "Are transfers on free with this service?",
        LLMJudgeConfig(),
        "",
        prior_events,
        turn_events,
    )

    assert failure is not None
    assert failure.assertion == assertion
    assert failure.error_message == (
        "Generative response 'Transfers are free for domestic service.' "
        "given to the user input 'Are transfers on free with this service?' "
        "was not relevant. "
        "Expected score to be above '0.85' threshold, but was '0.2'. "
        "The explanation for this score is: test justification."
    )
    assert event is None


def test_generative_response_is_grounded_run_llm_evaluation_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    table = {
        "eval_results_table": pd.DataFrame(
            data=[
                {
                    "answer_correctness/v1/score": 1,
                    "answer_correctness/v1/justification": "test justification",
                }
            ],
        )
    }
    set_up_tests_for_generative_response_assertions(
        monkeypatch, table, "answer_correctness"
    )
    assertion = get_assertion(AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value)

    matching_event = BotUttered(
        "Transfers are free for domestic service.",
        metadata={
            SEARCH_RESULTS_METADATA_KEY: "Domestic transfers are free of charge."
        },
    )
    prior_events = [SessionStarted()]
    turn_events = [
        UserUttered("Are transfers on free with this service?"),
        matching_event,
    ]
    failure, event = assertion._run_llm_evaluation(
        matching_event,
        "Are transfers on free with this service?",
        LLMJudgeConfig(),
        "",
        prior_events,
        turn_events,
    )

    assert failure is not None
    assert failure.assertion == assertion
    assert failure.error_message == (
        "Generative response 'Transfers are free for domestic service.' "
        "given to the user input 'Are transfers on free with this service?' "
        "was not grounded. Expected score to be above '0.85' threshold, "
        "but was '0.2'. The explanation for this score is: "
        "test justification."
    )
    assert event is None


@pytest.mark.parametrize(
    "assertion_type, data, metric_name",
    [
        (
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value,
            [
                {
                    "answer_correctness/v1/score": 5,
                    "answer_correctness/v1/justification": "test justification",
                }
            ],
            "answer_correctness",
        ),
        (
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value,
            [
                {
                    "answer_relevance/v1/score": 5,
                    "answer_relevance/v1/justification": "test justification",
                }
            ],
            "answer_relevance",
        ),
    ],
)
def test_generative_response_assertions_run_assertion_with_utter_name_success(
    monkeypatch: MonkeyPatch,
    assertion_type: str,
    data: List[Dict[str, Any]],
    metric_name: str,
) -> None:
    table = {"eval_results_table": pd.DataFrame(data=data)}

    set_up_tests_for_generative_response_assertions(monkeypatch, table, metric_name)
    assertion = get_assertion(assertion_type)

    matching_events = [
        BotUttered(
            "Transfers are free for domestic service.",
            metadata={
                "utter_action": "utter_free_transfers",
            },
        ),
        BotUttered(
            "Is there anything else I can help you with?",
            metadata={"utter_action": "utter_help"},
        ),
    ]

    failure, event = assertion._run_assertion_with_utter_name(
        matching_events,
        "Are transfers on free with this service?",
        LLMJudgeConfig(),
        "",
        [SessionStarted()],
        [UserUttered("Are transfers on free with this service?"), *matching_events],
    )

    assert failure is None
    assert event == matching_events[0]


@pytest.mark.parametrize(
    "assertion_type, data, metric_name",
    [
        (
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value,
            [
                {
                    "answer_correctness/v1/score": 5,
                    "answer_correctness/v1/justification": "test justification",
                }
            ],
            "answer_correctness",
        ),
        (
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value,
            [
                {
                    "answer_relevance/v1/score": 5,
                    "answer_relevance/v1/justification": "test justification",
                }
            ],
            "answer_relevance",
        ),
    ],
)
def test_generative_response_run_assertion_with_utter_name_failure(
    monkeypatch: MonkeyPatch,
    assertion_type: str,
    data: List[Dict[str, Any]],
    metric_name: str,
) -> None:
    table = {"eval_results_table": pd.DataFrame(data=data)}
    set_up_tests_for_generative_response_assertions(monkeypatch, table, metric_name)
    assertion = get_assertion(assertion_type)
    matching_events = [
        BotUttered(
            "International transfers are not free for domestic service.",
            metadata={
                "utter_action": "utter_international_transfers",
            },
        ),
        BotUttered(
            "Is there anything else I can help you with?",
            metadata={"utter_action": "utter_help"},
        ),
    ]

    failure, event = assertion._run_assertion_with_utter_name(
        matching_events,
        "Are international transfers free with this service?",
        LLMJudgeConfig(),
        "",
        [SessionStarted()],
        [
            UserUttered("Are international transfers free with this service?"),
            *matching_events,
        ],
    )

    assert event is None
    assert failure is not None
    assert failure.assertion == assertion
    assert failure.error_message == (
        "Bot did not utter 'utter_free_transfers' response."
    )


@pytest.mark.parametrize(
    "assertion_type, data, metric_name, expected_error_message",
    [
        (
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value,
            [
                {
                    "answer_correctness/v1/score": 1,
                    "answer_correctness/v1/justification": "test justification",
                }
            ],
            "answer_correctness",
            "None of the generative responses issued by either the "
            "Enterprise Search Policy, IntentlessPolicy or the "
            "Contextual Response Rephraser were grounded.",
        ),
        (
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value,
            [
                {
                    "answer_relevance/v1/score": 1,
                    "answer_relevance/v1/justification": "test justification",
                }
            ],
            "answer_relevance",
            "None of the generative responses issued by either the "
            "Enterprise Search Policy, IntentlessPolicy or the "
            "Contextual Response Rephraser were relevant.",
        ),
    ],
)
def test_generative_response_assertions_run_multiple_responses_failure(
    monkeypatch: MonkeyPatch,
    assertion_type: str,
    data: List[Dict[str, Any]],
    metric_name: str,
    expected_error_message: str,
) -> None:
    table = {"eval_results_table": pd.DataFrame(data=data)}
    set_up_tests_for_generative_response_assertions(monkeypatch, table, metric_name)
    assertion = get_assertion(assertion_type)
    matching_events = [
        BotUttered(
            "I'm afraid I don't have any knowledge of this.",
            metadata={
                "utter_action": "utter_no_knowledge",
            },
        ),
        BotUttered(
            "Is there anything else I can help you with?",
            metadata={"utter_action": "utter_help"},
        ),
    ]

    failure, event = assertion._run_assertion_for_multiple_generative_responses(
        matching_events,
        "Are international transfers free with this service?",
        LLMJudgeConfig(),
        "",
        [SessionStarted()],
        [
            UserUttered("Are international transfers free with this service?"),
            *matching_events,
        ],
    )

    assert event is None
    assert failure is not None
    assert failure.assertion == assertion
    assert failure.error_message == expected_error_message


@pytest.mark.parametrize(
    "assertion_type, data, metric_name,",
    [
        (
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value,
            [
                {
                    "answer_correctness/v1/score": 5,
                    "answer_correctness/v1/justification": "test justification",
                }
            ],
            "answer_correctness",
        ),
        (
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value,
            [
                {
                    "answer_relevance/v1/score": 5,
                    "answer_relevance/v1/justification": "test justification",
                }
            ],
            "answer_relevance",
        ),
    ],
)
def test_generative_response_assertions_run_multiple_responses_success(
    monkeypatch: MonkeyPatch,
    assertion_type: str,
    data: List[Dict[str, Any]],
    metric_name: str,
) -> None:
    table = {"eval_results_table": pd.DataFrame(data=data)}
    set_up_tests_for_generative_response_assertions(monkeypatch, table, metric_name)
    assertion = get_assertion(assertion_type)
    matching_events = [
        BotUttered(
            "International transfers are not free for the domestic service.",
            metadata={
                "utter_action": "utter_international_transfers",
            },
        ),
        BotUttered(
            "Is there anything else I can help you with?",
            metadata={"utter_action": "utter_help"},
        ),
    ]

    failure, event = assertion._run_assertion_for_multiple_generative_responses(
        matching_events,
        "Are international transfers free with this service?",
        LLMJudgeConfig(),
        "",
        [SessionStarted()],
        [
            UserUttered("Are international transfers free with this service?"),
            *matching_events,
        ],
    )

    assert event is not None
    assert failure is None


@pytest.mark.parametrize(
    "assertion_type, data, metric_name,",
    [
        (
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value,
            [
                {
                    "answer_correctness/v1/score": 5,
                    "answer_correctness/v1/justification": "test justification",
                }
            ],
            "answer_correctness",
        ),
        (
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value,
            [
                {
                    "answer_relevance/v1/score": 5,
                    "answer_relevance/v1/justification": "test justification",
                }
            ],
            "answer_relevance",
        ),
    ],
)
def test_generative_response_run_no_matching_events(
    monkeypatch: MonkeyPatch,
    assertion_type: str,
    data: List[Dict[str, Any]],
    metric_name: str,
) -> None:
    table = {"eval_results_table": pd.DataFrame(data=data)}
    set_up_tests_for_generative_response_assertions(monkeypatch, table, metric_name)
    assertion = get_assertion(assertion_type)
    matching_events = [
        SlotSet("service_name", "domestic"),
        ActionExecuted("action_listen"),
    ]

    failure, event = assertion.run(
        [
            UserUttered("Are international transfers free with this service?"),
            *matching_events,
        ],
        [SessionStarted()],
        "",
    )

    assert event is None
    assert failure is not None
    assert failure.assertion == assertion
    assert failure.error_message == (
        "No generative response issued by either the Enterprise Search Policy, "
        "IntentlessPolicy or the Contextual Response Rephraser was found, "
        "but one was expected."
    )


@pytest.mark.parametrize(
    "assertion_dict, expected_assertion_type",
    [
        ({"slot_was_set": [{"name": "name", "value": None}]}, SlotWasSetAssertion),
        (
            {"slot_was_not_set": [{"name": "name", "value": None}]},
            SlotWasNotSetAssertion,
        ),
    ],
)
def test_slot_assertions_with_null_value(
    assertion_dict: Dict[str, Any], expected_assertion_type: Assertion
) -> None:
    assertion = Assertion.create_typed_assertion(assertion_dict)
    assert assertion is not None
    assert isinstance(assertion, expected_assertion_type)
    assert hasattr(assertion, "slots")
    assert assertion.slots[0].value is None
