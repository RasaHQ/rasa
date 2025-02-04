import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
from litellm.types.utils import EmbeddingResponse
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
    BotDidNotUtterAssertion,
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
    _get_default_prompt_template,
    calculate_score,
)
from rasa.e2e_test.constants import (
    DEFAULT_ANSWER_RELEVANCE_PROMPT_TEMPLATE_FILE_NAME,
    DEFAULT_GROUNDEDNESS_PROMPT_TEMPLATE_FILE_NAME,
)
from rasa.e2e_test.e2e_config import LLMJudgeConfig
from rasa.e2e_test.utils.generative_assertions import ScoreInputs
from rasa.shared.constants import OPENAI_PROVIDER
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
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.providers.llm.openai_llm_client import OpenAILLMClient


@pytest.fixture
def llm_judge_config() -> LLMJudgeConfig:
    config = LLMJudgeConfig.from_dict(
        {
            "llm": {},
            "embeddings": {},
        }
    )
    return config


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
        (
            {
                "bot_did_not_utter": {
                    "utter_name": "utter_options",
                    "text_matches": "You can transfer money or check your balance.",
                    "buttons": [
                        {"title": "Transfer Money", "payload": "/transfer_money"}
                    ],
                }
            },
            BotDidNotUtterAssertion(
                utter_name="utter_options",
                text_matches="You can transfer money or check your balance.",
                buttons=[
                    AssertedButton(title="Transfer Money", payload="/transfer_money")
                ],
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
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT,
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
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED,
            "answer_correctness",
        ),
    ],
)
def test_create_typed_assertion_valid_generative_assertions(
    monkeypatch: MonkeyPatch,
    data: Dict[str, Any],
    assertion_type: AssertionType,
    metric_name: str,
):
    def get_expected_assertion(assertion_type: AssertionType) -> Assertion:
        if assertion_type == AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED:
            return GenerativeResponseIsGroundedAssertion(
                threshold=0.88,
                utter_name="utter_fee",
                ground_truth="The fee for transferring money is $5.",
                metric_adjective="grounded",
            )
        elif assertion_type == AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT:
            return GenerativeResponseIsRelevantAssertion(
                threshold=0.9,
                utter_name="utter_options",
                metric_adjective="relevant",
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


@pytest.mark.parametrize(
    "assertion, turn_events",
    [
        (
            BotDidNotUtterAssertion(
                text_matches="You can transfer money or check your balance."
            ),
            [BotUttered(text="Something else.")],
        ),
        (
            BotDidNotUtterAssertion(
                utter_name="utter_options",
            ),
            [BotUttered(metadata={"utter_action": "utter_something_else"})],
        ),
        (
            BotDidNotUtterAssertion(
                buttons=[
                    AssertedButton(title="Transfer Money", payload="/transfer_money")
                ]
            ),
            [
                BotUttered(
                    data={
                        "buttons": [
                            {"title": "Check Balance", "payload": "/check_balance"}
                        ]
                    }
                )
            ],
        ),
    ],
)
def test_assertion_run_returns_no_assertion_failure_for_bot_did_not_utter_assertion(
    assertion: Assertion, turn_events: List[Event]
) -> None:
    assertion_failure, matching_event = assertion.run(turn_events, [])
    assert assertion_failure is None
    assert matching_event is None


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
    assertion: Assertion,
    expected_assertion_failure: AssertionFailure,
) -> None:
    assertion_failure, matching_event = assertion.run([], [])
    assert assertion_failure == expected_assertion_failure
    assert matching_event is None


@pytest.mark.parametrize(
    "turn_events, assertion, expected_assertion_failure",
    [
        (
            [
                BotUttered(
                    text="You can transfer money or check your balance.",
                )
            ],
            BotDidNotUtterAssertion(
                text_matches="You can transfer money or check your balance."
            ),
            AssertionFailure(
                assertion=BotDidNotUtterAssertion(
                    text_matches="You can transfer money or check your balance.",
                    line=None,
                ),
                error_message=(
                    "Bot uttered a forbidden message matching the pattern "
                    "'You can transfer money or check your balance.'."
                ),
                actual_events_transcript=[
                    "BotUttered('You can transfer money or check your balance.', "
                    "{}, {}, None)"
                ],
                error_line=None,
            ),
        ),
        (
            [BotUttered(metadata={"utter_action": "utter_options"})],
            BotDidNotUtterAssertion(
                utter_name="utter_options",
            ),
            AssertionFailure(
                assertion=BotDidNotUtterAssertion(
                    utter_name="utter_options",
                    text_matches=None,
                    buttons=None,
                    line=None,
                ),
                error_message="Bot uttered a forbidden utterance 'utter_options'.",
                actual_events_transcript=[
                    'BotUttered(\'None\', {}, {"utter_action": "utter_options"}, None)'
                ],
                error_line=None,
            ),
        ),
        (
            [
                BotUttered(
                    data={
                        "buttons": [
                            {"title": "Transfer Money", "payload": "/transfer_money"}
                        ]
                    }
                )
            ],
            BotDidNotUtterAssertion(
                buttons=[
                    AssertedButton(title="Transfer Money", payload="/transfer_money")
                ]
            ),
            AssertionFailure(
                assertion=BotDidNotUtterAssertion(
                    utter_name=None,
                    text_matches=None,
                    buttons=[
                        AssertedButton(
                            title="Transfer Money", payload="/transfer_money"
                        )
                    ],
                    line=None,
                ),
                error_message=(
                    "Bot uttered a forbidden response with specified buttons."
                ),
                actual_events_transcript=[
                    'BotUttered(\'None\', {"buttons": [{"title": "Transfer Money", '
                    '"payload": "/transfer_money"}]}, {}, None)'
                ],
                error_line=None,
            ),
        ),
    ],
)
def test_assertion_run_returns_assertion_failure_for_bot_did_not_utter_assertion(
    turn_events: List[Event],
    assertion: Assertion,
    expected_assertion_failure: AssertionFailure,
) -> None:
    # Remove timestamps from events to make the test deterministic
    turn_events[0].timestamp = None
    assertion_failure, matching_event = assertion.run(turn_events, [])
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
                metric_adjective="relevant",
            ),
            {
                "threshold": 0.9,
                "utter_name": "utter_options",
                "type": "generative_response_is_relevant",
                "line": None,
                "utter_source": None,
            },
        ),
        (
            GenerativeResponseIsGroundedAssertion(
                threshold=0.88,
                utter_name="utter_fee",
                ground_truth="The fee for transferring money is $5.",
                metric_adjective="grounded",
            ),
            {
                "threshold": 0.88,
                "utter_name": "utter_fee",
                "ground_truth": "The fee for transferring money is $5.",
                "type": "generative_response_is_grounded",
                "utter_source": None,
                "line": None,
            },
        ),
        (
            BotDidNotUtterAssertion(
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
                "type": "bot_did_not_utter",
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
    monkeypatch: MonkeyPatch, llm_response: str
) -> None:
    def mock_invoke_llm(*args, **kwargs):
        return llm_response

    monkeypatch.setattr(
        "rasa.e2e_test.assertions.GenerativeResponseMixin._invoke_llm", mock_invoke_llm
    )


def set_up_tests_for_answer_relevance_assertion(
    monkeypatch: MonkeyPatch, embedding_response: EmbeddingResponse
) -> Mock:
    monkeypatch.setenv("OPENAI_API_KEY", "openai llm embedding validation key")

    mock_embed = Mock(return_value=embedding_response)
    monkeypatch.setattr(
        "rasa.shared.providers.embedding._base_litellm_embedding_client.embedding",
        mock_embed,
    )
    return mock_embed


def get_assertion(assertion_type: AssertionType):
    if assertion_type == AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED:
        return GenerativeResponseIsGroundedAssertion.from_dict(
            {
                assertion_type.value: {
                    "threshold": 0.85,
                    "ground_truth": "Sending money to friends and family with FinX incurs no charges.",  # noqa: E501
                }
            }
        )
    elif assertion_type == AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT:
        return GenerativeResponseIsRelevantAssertion.from_dict(
            {
                assertion_type.value: {
                    "threshold": 0.85,
                    "utter_name": "utter_free_transfers",
                }
            }
        )


def test_generative_response_grounded_assertion_run_llm_evaluation_success(
    monkeypatch: MonkeyPatch, llm_judge_config: LLMJudgeConfig
) -> None:
    llm_response = """```json
            {
                "statements":[
                    {
                        "statement": "Transfers to friends and family through FinX are free of charge.",
                        "score": 1,
                        "justification": "The ground truth confirms that sending money to friends and family with FinX incurs no charges."
                    },
                    {
                        "statement": "FinX allows fee-free transactions.",
                        "score": 1,
                        "justification": "The ground truth explicitly states that FinX offers fee-free transactions."
                    },
                     {
                        "statement": "FinX provides the convenience of instant transfers.",
                        "score": 1,
                        "justification": "The ground truth highlights that FinX emphasises the ease of use in transferring funds."
                    }
                ]
            }
            ```
            """  # noqa: E501
    set_up_tests_for_generative_response_assertions(monkeypatch, llm_response)
    assertion = get_assertion(AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED)

    matching_event = BotUttered("Transfers are free for domestic service.")

    failure, event = assertion._run_llm_evaluation(
        matching_event,
        "Are transfers on free with this service?",
        llm_judge_config,
        "",
        [SessionStarted()],
        [UserUttered("Are transfers on free with this service?"), matching_event],
    )

    assert failure is None
    assert event == matching_event


def test_generative_response_answer_relevance_assertion_run_llm_evaluation_success(
    monkeypatch: MonkeyPatch, llm_judge_config: LLMJudgeConfig
) -> None:
    generated_question = "Are international transfers free with the domestic service?"
    llm_response = f"""```json
            {{
                "question_variations":["{generated_question}"]
            }}
            ```
            """
    set_up_tests_for_generative_response_assertions(monkeypatch, llm_response)
    embedding_response = EmbeddingResponse(
        data=[
            {"embedding": [0.9, 0.9, 0.9], "index": 0, "object": "embedding"},
        ]
    )
    mock_embed = set_up_tests_for_answer_relevance_assertion(
        monkeypatch, embedding_response
    )
    assertion = get_assertion(AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT)

    matching_event = BotUttered("Transfers are free for domestic service.")
    user_question = "Are transfers on free with this service?"

    failure, event = assertion._run_llm_evaluation(
        matching_event,
        user_question,
        llm_judge_config,
        "",
        [SessionStarted()],
        [UserUttered(user_question), matching_event],
    )

    assert failure is None
    assert event == matching_event

    assert mock_embed.call_count == 2
    assert mock_embed.call_args_list == [
        call(
            input=[user_question],
            model="openai/text-embedding-ada-002",
            api_base=None,
            api_type="openai",
            api_version=None,
        ),
        call(
            input=[generated_question],
            model="openai/text-embedding-ada-002",
            api_base=None,
            api_type="openai",
            api_version=None,
        ),
    ]


def test_generative_response_is_relevant_run_llm_evaluation_failure_no_generated_questions(  # noqa: E501
    monkeypatch: MonkeyPatch, llm_judge_config: LLMJudgeConfig
) -> None:
    llm_response = """
            ```json
            {
                "question_variations": []
            }
            ```
    """
    set_up_tests_for_generative_response_assertions(monkeypatch, llm_response)

    embedding_response = EmbeddingResponse(
        data=[
            {"embedding": [], "index": 0, "object": "embedding"},
        ]
    )
    mock_embed = set_up_tests_for_answer_relevance_assertion(
        monkeypatch, embedding_response
    )

    assertion = get_assertion(AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT)

    matching_event = BotUttered("I don't know how to answer this.")
    prior_events = [SessionStarted()]

    user_question = "Are transfers on free with this service?"
    turn_events = [
        UserUttered(user_question),
        matching_event,
    ]
    failure, event = assertion._run_llm_evaluation(
        matching_event,
        user_question,
        llm_judge_config,
        "",
        prior_events,
        turn_events,
    )

    assert failure is not None
    assert failure.assertion == assertion
    assert (
        failure.error_message
        == "No question variations were extracted by the LLM Judge."
    )

    assert event is None
    assert mock_embed.call_count == 0


def test_generative_response_is_relevant_run_llm_evaluation_failure(
    monkeypatch: MonkeyPatch, llm_judge_config: LLMJudgeConfig
) -> None:
    generated_question = "How is the weather today?"
    llm_response = f"""
            ```json
            {{
                "question_variations": ["{generated_question}"]
            }}
            ```
    """
    set_up_tests_for_generative_response_assertions(monkeypatch, llm_response)

    embedding_responses = [
        EmbeddingResponse(
            data=[
                {"embedding": [0.9, 0.9, 0.9], "index": 0, "object": "embedding"},
            ]
        ),
        EmbeddingResponse(
            data=[
                {"embedding": [0.1, 0.1, -0.1], "index": 0, "object": "embedding"},
            ]
        ),
    ]
    monkeypatch.setenv("OPENAI_API_KEY", "openai llm embedding validation key")

    mock_embed = Mock()
    mock_embed.side_effect = embedding_responses
    monkeypatch.setattr(
        "rasa.shared.providers.embedding._base_litellm_embedding_client.embedding",
        mock_embed,
    )

    assertion = get_assertion(AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT)

    matching_event = BotUttered("The weather is pleasant today. Enjoy your day!")
    prior_events = [SessionStarted()]

    user_question = "Are transfers on free with this service?"
    turn_events = [
        UserUttered(user_question),
        matching_event,
    ]
    failure, event = assertion._run_llm_evaluation(
        matching_event,
        user_question,
        llm_judge_config,
        "",
        prior_events,
        turn_events,
    )

    assert failure is not None
    assert failure.assertion == assertion
    assert failure.error_message == (
        "Generative response 'The weather is pleasant today. "
        "Enjoy your day!' given to the user input "
        "'Are transfers on free with this service?' was "
        "not relevant. Expected score to be above '0.85' "
        "threshold, but was '0.33'. The LLM Judge model "
        "has justified its score like so: Question 'How "
        "is the weather today?' has a cosine similarity "
        "score of '0.33' with the user question "
        "'Are transfers on free with this service?'."
    )

    assert event is None
    assert mock_embed.call_count == 2
    assert mock_embed.call_args_list == [
        call(
            input=[user_question],
            model="openai/text-embedding-ada-002",
            api_base=None,
            api_type=OPENAI_PROVIDER,
            api_version=None,
        ),
        call(
            input=[generated_question],
            model="openai/text-embedding-ada-002",
            api_base=None,
            api_type=OPENAI_PROVIDER,
            api_version=None,
        ),
    ]


def test_generative_response_is_grounded_run_llm_evaluation_failure(
    monkeypatch: MonkeyPatch, llm_judge_config: LLMJudgeConfig
) -> None:
    llm_response = """```json
    {
        "statements": [
            {
                "statement": "Local transfers through FinX incur a small fee.",
                "score": 0,
                "justification": "test justification"
            }
        ]
    }
    ```
    """
    set_up_tests_for_generative_response_assertions(monkeypatch, llm_response)
    assertion = get_assertion(AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED)

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
        llm_judge_config,
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
        "but was '0.0'. The LLM Judge model has justified its score "
        "like so: There were 1 incorrect statements out of 1 total "
        "extracted statements. The justifications for these statements "
        "include: test justification."
    )
    assert event is None


def test_generative_response_answer_relevance_assertion_run_assertion_with_utter_name_success(  # noqa: E501
    monkeypatch: MonkeyPatch, llm_judge_config: LLMJudgeConfig
) -> None:
    generated_question = "Are international transfers free with the domestic service?"
    llm_response = f"""
            ```json
            {{
                "question_variations": ["{generated_question}"]
            }}
            ```
            """
    set_up_tests_for_generative_response_assertions(monkeypatch, llm_response)

    embedding_response = EmbeddingResponse(
        data=[
            {"embedding": [0.9, 0.9, 0.9], "index": 0, "object": "embedding"},
        ]
    )
    mock_embed = set_up_tests_for_answer_relevance_assertion(
        monkeypatch, embedding_response
    )

    assertion = get_assertion(AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT)

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

    user_question = "Are transfers on free with this service?"
    failure, event = assertion._run_assertion_with_utter_name(
        matching_events,
        user_question,
        llm_judge_config,
        "",
        [SessionStarted()],
        [UserUttered(user_question), *matching_events],
    )

    assert failure is None
    assert event == matching_events[0]
    assert mock_embed.call_count == 2
    assert mock_embed.call_args_list == [
        call(
            input=[user_question],
            model="openai/text-embedding-ada-002",
            api_base=None,
            api_type="openai",
            api_version=None,
        ),
        call(
            input=[generated_question],
            model="openai/text-embedding-ada-002",
            api_base=None,
            api_type="openai",
            api_version=None,
        ),
    ]


def test_generative_response_run_assertion_with_utter_name_failure(
    monkeypatch: MonkeyPatch, llm_judge_config: LLMJudgeConfig
) -> None:
    llm_response = """
            ```json
            {
                "question_variations": ["Are international transfers free with the domestic service?"]
            }
            ```
            """  # noqa: E501
    set_up_tests_for_generative_response_assertions(monkeypatch, llm_response)
    assertion = get_assertion(AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT)
    matching_events = [
        BotUttered(
            "International transfers are not free for domestic service.",
            metadata={
                "utter_source": "EnterpriseSearchPolicy",
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
        llm_judge_config,
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
    "assertion_type, expected_adjective",
    [
        (
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED,
            "grounded",
        ),
        (
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT,
            "relevant",
        ),
    ],
)
def test_generative_response_assertions_run_multiple_responses_failure(
    monkeypatch: MonkeyPatch,
    assertion_type: AssertionType,
    expected_adjective: str,
    llm_judge_config: LLMJudgeConfig,
) -> None:
    llm_response = """```json
            {}
            ```
            """
    set_up_tests_for_generative_response_assertions(monkeypatch, llm_response)
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
            metadata={
                "utter_action": "utter_help",
            },
        ),
    ]

    failure, event = assertion.run(
        turn_events=[
            UserUttered("Are international transfers free with this service?"),
            *matching_events,
        ],
        prior_events=[SessionStarted()],
        llm_judge_config=llm_judge_config,
        step_text="Are international transfers free with this service?",
    )

    expected_error_message = (
        "No generative response issued by either "
        "the Enterprise Search Policy, IntentlessPolicy "
        "or the Contextual Response Rephraser was found, "
        "but one was expected."
    )

    assert event is None
    assert failure is not None
    assert failure.assertion == assertion
    assert failure.error_message == expected_error_message


def test_generative_response_grounded_assertion_run_multiple_responses_success(
    monkeypatch: MonkeyPatch, llm_judge_config: LLMJudgeConfig
) -> None:
    llm_response = """
            ```json
            {
                "statements":[
                    {
                        "statement": "International transfers are not free for the domestic service.",
                        "score": 1,
                        "justification": "test justification"
                    }
                ]
            }
            ```
            """  # noqa: E501
    set_up_tests_for_generative_response_assertions(monkeypatch, llm_response)
    assertion = GenerativeResponseIsGroundedAssertion.from_dict(
        {
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value: {
                "threshold": 0.85,
                "utter_source": "EnterpriseSearchPolicy",
            }
        }
    )
    matching_events = [
        BotUttered(
            "International transfers are not free for the domestic service.",
            metadata={
                "utter_action": "utter_international_transfers",
                "utter_source": "EnterpriseSearchPolicy",
            },
        ),
        BotUttered(
            "International transfers are free for premium service only.",
            metadata={
                "utter_action": "utter_premium_service",
                "utter_source": "EnterpriseSearchPolicy",
            },
        ),
    ]

    failure, event = assertion.run(
        turn_events=[
            UserUttered("Are international transfers free with this service?"),
            *matching_events,
        ],
        prior_events=[SessionStarted()],
        llm_judge_config=llm_judge_config,
        step_text="Are international transfers free with this service?",
    )

    assert event is not None
    assert failure is None


def test_generative_response_answer_relevance_assertion_run_multiple_responses_success(
    monkeypatch: MonkeyPatch, llm_judge_config: LLMJudgeConfig
) -> None:
    llm_response = """
            ```json
            {
                "question_variations": [
                    "Are international transfers free with the domestic service?",
                    "Are international transfers free with the premium service?"
                ]
            }
            ```
            """
    set_up_tests_for_generative_response_assertions(monkeypatch, llm_response)
    assertion = GenerativeResponseIsRelevantAssertion.from_dict(
        {
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value: {
                "threshold": 0.85,
                "utter_source": "EnterpriseSearchPolicy",
            }
        }
    )

    embedding_response = EmbeddingResponse(
        data=[
            {"embedding": [0.9, 0.9, 0.9], "index": 0, "object": "embedding"},
            {"embedding": [0.9, 0.9, 0.9], "index": 1, "object": "embedding"},
        ]
    )
    mock_embed = set_up_tests_for_answer_relevance_assertion(
        monkeypatch, embedding_response
    )

    matching_events = [
        BotUttered(
            "International transfers are not free for the domestic service.",
            metadata={
                "utter_action": "utter_international_transfers",
                "utter_source": "EnterpriseSearchPolicy",
            },
        ),
        BotUttered(
            "International transfers are free for premium service only.",
            metadata={
                "utter_action": "utter_premium_service",
                "utter_source": "EnterpriseSearchPolicy",
            },
        ),
    ]

    user_question = "Are international transfers free with this service?"
    failure, event = assertion.run(
        turn_events=[
            UserUttered(user_question),
            *matching_events,
        ],
        prior_events=[SessionStarted()],
        llm_judge_config=llm_judge_config,
        step_text=user_question,
    )

    assert event is not None
    assert failure is None

    assert mock_embed.call_count == 2

    assert mock_embed.call_args_list == [
        call(
            input=[user_question],
            model="openai/text-embedding-ada-002",
            api_base=None,
            api_type="openai",
            api_version=None,
        ),
        call(
            input=[
                "Are international transfers free with the domestic service?",
                "Are international transfers free with the premium service?",
            ],
            model="openai/text-embedding-ada-002",
            api_base=None,
            api_type="openai",
            api_version=None,
        ),
    ]


@pytest.mark.parametrize(
    "assertion_type, ",
    [
        AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED,
        AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT,
    ],
)
def test_generative_response_run_no_matching_events(
    monkeypatch: MonkeyPatch,
    assertion_type: AssertionType,
) -> None:
    llm_response = """
    ```json
    {}
    ```
    """
    set_up_tests_for_generative_response_assertions(monkeypatch, llm_response)
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


@pytest.mark.parametrize(
    "assertion_type, turn_events, expected_error_messages",
    [
        (
            AssertionType.BOT_DID_NOT_UTTER.value,
            [
                BotUttered(
                    metadata={"utter_action": "utter_need_help"},
                    text="Do you need help with anything else?",
                    data={
                        "buttons": [
                            {"title": "Yes", "payload": "/yes"},
                            {"title": "no", "payload": "/no"},
                        ]
                    },
                )
            ],
            [
                "Bot uttered a forbidden utterance 'utter_need_help'.",
                (
                    "Bot uttered a forbidden message matching the pattern "
                    "'Do you need help with anything else?'."
                ),
                "Bot uttered a forbidden response with specified buttons.",
            ],
        ),
        (
            AssertionType.BOT_UTTERED.value,
            [
                BotUttered(
                    metadata={"utter_action": "utter_something_else"},
                    text="Something else.",
                    data={"buttons": []},
                )
            ],
            [
                "Bot did not utter 'utter_need_help' response.",
                (
                    "Bot did not utter any response which matches the provided "
                    "text pattern 'Do you need help with anything else?'."
                ),
                "Bot did not utter any response with the expected buttons.",
            ],
        ),
    ],
)
def test_bot_utterance_multiple_errors(
    assertion_type, turn_events, expected_error_messages
) -> None:
    assertion = Assertion.create_typed_assertion(
        {
            assertion_type: {
                "utter_name": "utter_need_help",
                "text_matches": "Do you need help with anything else?",
                "buttons": [
                    {"title": "Yes", "payload": "/yes"},
                    {"title": "no", "payload": "/no"},
                ],
            }
        }
    )
    failure, _ = assertion.run(turn_events, [])
    error_message = " ".join(expected_error_messages)
    assert failure is not None
    assert error_message == failure.error_message


@pytest.mark.parametrize(
    "expected_template_name",
    [
        DEFAULT_GROUNDEDNESS_PROMPT_TEMPLATE_FILE_NAME,
        DEFAULT_ANSWER_RELEVANCE_PROMPT_TEMPLATE_FILE_NAME,
    ],
)
def test_get_default_prompt_template(expected_template_name: str):
    template = _get_default_prompt_template(expected_template_name)

    expected_template = (
        Path(__file__).parent.parent.parent
        / "rasa"
        / "e2e_test"
        / "llm_judge_prompts"
        / expected_template_name
    )

    assert template == expected_template.read_text()


@patch("rasa.e2e_test.assertions.llm_factory")
@pytest.mark.parametrize(
    "assertion_type",
    [
        AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED,
        AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT,
    ],
)
def test_generative_response_mixin_calls_llm_factory_correctly(
    mock_llm_factory: Mock,
    monkeypatch: MonkeyPatch,
    assertion_type: AssertionType,
):
    assertion = get_assertion(assertion_type)

    # Given
    llm_judge_config = LLMJudgeConfig.from_dict(
        {
            "llm": {
                "model": "gpt-4-0613",
                "provider": "openai",
                "timeout": 7,
                "temperature": 0.0,
                "max_tokens": 256,
            }
        }
    )
    prompt = "some prompt"
    mock_llm_client = AsyncMock(spec=OpenAILLMClient)
    mock_completion = Mock(
        return_value=LLMResponse(id="1", created=1, choices=["some response"])
    )
    monkeypatch.setattr(mock_llm_client, "completion", mock_completion)
    mock_llm_factory.return_value = mock_llm_client

    # When
    assertion._invoke_llm(llm_judge_config, prompt)

    # Then
    mock_llm_factory.assert_called_once_with(
        llm_judge_config.llm_config_as_dict, llm_judge_config.get_default_llm_config()
    )
    mock_completion.assert_called_once_with(prompt)


def test_generative_response_mixin_process_response_success_groundedness():
    # Given
    assertion = get_assertion(AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED)
    llm_response = """```json
    {
        "statements":[
            {
                "statement": "Local transfers through FinX incur a small fee.",
                "score": 1,
                "justification": "test justification"
            }
        ]
    }
    ```
    """

    # When
    statements = assertion._process_response(llm_response, "some bot message")

    # Then
    assert len(statements) == 1
    assert (
        statements[0].get("statement")
        == "Local transfers through FinX incur a small fee."
    )
    assert statements[0].get("score") == 1
    assert statements[0].get("justification") == "test justification"


def test_generative_response_mixin_process_response_success_relevance():
    # Given
    assertion = get_assertion(AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT)
    llm_response = """```json
    {
        "question_variations": ["Do local transfers through FinX incur a fee?"]
    }
    ```
    """

    # When
    questions = assertion._process_response(llm_response, "some bot message")

    # Then
    assert len(questions) == 1
    assert questions[0] == "Do local transfers through FinX incur a fee?"


@pytest.mark.parametrize(
    "assertion_type",
    [
        AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED,
        AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT,
    ],
)
def test_generative_response_mixin_process_response_invalid_llm_output(
    assertion_type: AssertionType,
):
    assertion = get_assertion(assertion_type)
    llm_response = """```json
    {
        "statements": [
            {
                "statement": "Local transfers through FinX incur a small fee.",
                "score": 1,
                "justification": "test justification"
            },
        ]
    }
    ```
    """

    with pytest.raises(RasaException, match="Failed to parse the LLM Judge response"):
        assertion._process_response(llm_response, "some bot message")


@pytest.mark.parametrize(
    "assertion_type, llm_response",
    [
        (
            AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED,
            """```json
                {
                    "statements": [
                        {
                            "statement": "Local transfers through FinX
                                         incur a small fee.",
                            "score": 1,
                            "explanation": "incorrect field name"
                        }
                    ]
                }
                ```
                """,
        ),
        (
            AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT,
            """```json
                {
                    "question_variation": [
                        {
                            "variation": "incorrect field name",
                            "non_committed": 1
                        }
                    ]
                }
                ```
                """,
        ),
    ],
)
def test_generative_response_mixin_process_response_invalid_llm_json(
    assertion_type: AssertionType, llm_response: str
):
    assertion = get_assertion(assertion_type)

    with pytest.raises(
        RasaException, match="Failed to validate the LLM Judge json response"
    ):
        assertion._process_response(llm_response, "some bot message")


@pytest.mark.parametrize(
    "assertion_type, llm_output",
    [
        (AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED, "statements"),
        (AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT, "question variations"),
    ],
)
def test_generative_response_mixin_process_response_no_statements(
    assertion_type: AssertionType, llm_output: str
):
    assertion = get_assertion(assertion_type)
    llm_response = """```json
    {
        "statements": []
    }
    ```
    """

    with pytest.raises(RasaException, match=f"No {llm_output} were extracted"):
        assertion._process_response(llm_response, "some bot message")


def test_generative_response_mixin_calculate_score_groundedness() -> None:
    assertion = get_assertion(AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED)
    statements = [
        {
            "statement": "Local transfers through FinX incur a small fee.",
            "score": 1,
            "justification": "test justification 1",
        },
        {
            "statement": "FinX allows fee-free transactions.",
            "score": 1,
            "justification": "test justification 2",
        },
        {
            "statement": "FinX provides the convenience of instant transfers.",
            "score": 0,
            "justification": "test justification 3",
        },
    ]

    score_inputs = ScoreInputs(
        threshold=assertion.threshold,
        matching_event=BotUttered("some bot message"),
        user_question="",
        llm_judge_config=LLMJudgeConfig.from_dict({}),
    )

    score, error_justifications = calculate_score(
        assertion.type(), statements, score_inputs
    )

    assert math.isclose(score, 0.6666666666666666)
    assert error_justifications == (
        "There were 1 incorrect statements out of 3 "
        "total extracted statements. "
        "The justifications for these statements "
        "include: test justification 3"
    )
