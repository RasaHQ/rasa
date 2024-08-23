from typing import Any, Dict, List, Text, Union

import pytest

from rasa.cli.e2e_test import read_e2e_test_schema
from rasa.e2e_test.assertions import (
    ActionExecutedAssertion,
    AssertedSlot,
    AssertionFailure,
    BotUtteredAssertion,
    FlowCancelledAssertion,
    FlowCompletedAssertion,
    FlowStartedAssertion,
    PatternClarificationContainsAssertion,
    SlotWasNotSetAssertion,
    SlotWasSetAssertion,
)
from rasa.e2e_test.e2e_test_case import TestCase, TestStep
from rasa.e2e_test.e2e_test_result import TestResult


@pytest.fixture(scope="module")
def e2e_schema() -> Union[List[Any], Dict[Text, Any]]:
    return read_e2e_test_schema()


@pytest.fixture
def test_cases() -> List[TestCase]:
    return [
        TestCase(
            name="test_case_1",
            steps=[
                TestStep(
                    actor="user",
                    text="contacts",
                    assertions=[
                        FlowStartedAssertion(flow_id="pattern_clarification"),
                        PatternClarificationContainsAssertion(
                            flow_names={"add contact", "remove contact", "list contact"}
                        ),
                    ],
                ),
                TestStep(
                    actor="user",
                    text="add contact",
                    assertions=[
                        FlowCompletedAssertion(flow_id="pattern_clarification"),
                        FlowStartedAssertion(flow_id="add_contact"),
                        BotUtteredAssertion(utter_name="utter_ask_user_name"),
                    ],
                ),
                TestStep(
                    actor="user",
                    text="john",
                    assertions=[
                        SlotWasSetAssertion(
                            slots=[AssertedSlot(name="user_name", value="john")]
                        ),
                        BotUtteredAssertion(utter_name="utter_ask_user_email"),
                    ],
                ),
                TestStep(
                    actor="user",
                    text="You don't need to know this, I don't want to continue.",
                    assertions=[
                        ActionExecutedAssertion(
                            action_name="action_cancel_add_contact"
                        ),
                        FlowCancelledAssertion(flow_id="add_contact"),
                        BotUtteredAssertion(
                            text_matches="Okay, I have cancelled the process."
                        ),
                        SlotWasSetAssertion(
                            slots=[AssertedSlot(name="user_name", value=None)]
                        ),
                    ],
                ),
            ],
        ),
        TestCase(
            name="test_case_2",
            steps=[
                TestStep(
                    actor="user",
                    text="send money",
                    assertions=[
                        FlowStartedAssertion(flow_id="transfer_money"),
                        BotUtteredAssertion(utter_name="utter_ask_recipient"),
                    ],
                ),
                TestStep(
                    actor="user",
                    text="jane",
                    assertions=[
                        SlotWasSetAssertion(
                            slots=[AssertedSlot(name="user_name", value="jane")]
                        ),
                        BotUtteredAssertion(utter_name="utter_ask_amount"),
                    ],
                ),
                TestStep(
                    actor="user",
                    text="100 pounds",
                    assertions=[
                        SlotWasSetAssertion(
                            slots=[AssertedSlot(name="amount", value=100)]
                        ),
                        BotUtteredAssertion(utter_name="utter_ask_confirmation"),
                    ],
                ),
                TestStep(
                    actor="user",
                    text="no",
                    assertions=[
                        SlotWasNotSetAssertion(
                            slots=[
                                AssertedSlot(
                                    name="confirmation", value="value key is undefined"
                                )
                            ]
                        ),
                        ActionExecutedAssertion(action_name="action_transfer_money"),
                        FlowCompletedAssertion(flow_id="transfer_money"),
                    ],
                ),
            ],
        ),
    ]


@pytest.fixture
def passed_assertion_results(test_cases: List[TestCase]) -> List[TestResult]:
    return [
        TestResult(
            test_case=test_cases[0],
            pass_status=True,
            difference=[],
        ),
    ]


@pytest.fixture
def failed_assertion_results(test_cases: List[TestCase]) -> List[TestResult]:
    return [
        TestResult(
            test_case=test_cases[1],
            pass_status=False,
            difference=[],
            assertion_failure=AssertionFailure(
                assertion=ActionExecutedAssertion(action_name="action_transfer_money"),
                error_message="Test error message",
                actual_events_transcript=[],
            ),
        )
    ]
