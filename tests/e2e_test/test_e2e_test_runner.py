import copy
import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Union
from unittest.mock import MagicMock, Mock

import pytest
import requests
from pytest import LogCaptureFixture, MonkeyPatch, CaptureFixture
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from structlog.testing import capture_logs

import rasa.cli.e2e_test
from rasa.core.agent import Agent
from rasa.core.channels import CollectingOutputChannel
from rasa.core.constants import ACTIVE_FLOW_METADATA_KEY, STEP_ID_METADATA_KEY
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.core.utils import AvailableEndpoints
from rasa.e2e_test.e2e_test_case import (
    ActualStepOutput,
    Fixture,
    Metadata,
    TestCase,
    TestStep,
)
from rasa.e2e_test.e2e_test_result import TestResult
from rasa.e2e_test.e2e_test_runner import TEST_TURNS_TYPE, E2ETestRunner
from rasa.llm_fine_tuning.conversations import Conversation
from rasa.shared.core.constants import (
    REQUESTED_SLOT,
    SESSION_START_METADATA_SLOT,
    SLOT_LAST_OBJECT,
    SLOT_LAST_OBJECT_TYPE,
    SLOT_LISTED_ITEMS,
)
from rasa.shared.core.domain import Domain
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
    DialogueStackUpdated,
)
from rasa.shared.core.flows.flow_path import FlowPath, PathNode
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

if sys.version_info[:2] >= (3, 8):
    from unittest.mock import AsyncMock


@pytest.fixture
def test_suite_metadata() -> List[Metadata]:
    return [
        Metadata(name="device_info", metadata={"os": "linux"}),
        Metadata(name="user_info", metadata={"name": "Tom"}),
    ]


@pytest.fixture
def test_case_metadata() -> Metadata:
    return Metadata(name="device_info", metadata={"os": "linux"})


@pytest.fixture
def mock_e2e_test_runner(monkeypatch: MonkeyPatch) -> E2ETestRunner:
    def mock_init(self: Any, *args: Any, **kwargs: Any) -> None:
        domain = Domain.from_dict(
            {
                "entities": ["city"],
                "slots": {
                    "city": {
                        "type": "text",
                        "mappings": [{"type": "from_entity", "entity": "city"}],
                    }
                },
                "actions": [
                    "custom_action_1",
                    "custom_action_2",
                    "custom_action_3",
                ],
            }
        )
        self.agent = Agent(
            domain=domain, tracker_store=InMemoryTrackerStore(domain=domain)
        )
        processor = AsyncMock()
        # using the actual tracker store instead of a mocked one
        processor.fetch_tracker_with_initial_session = (
            self.agent.tracker_store.get_or_create_tracker
        )
        self.agent.processor = processor

    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.__init__", mock_init
    )

    async def mock_handle_message(self: Any, message: Any) -> None:
        tracker = await self.tracker_store.get_or_create_tracker(message.sender_id)
        tracker.update(UserUttered(message.text))
        await self.tracker_store.save(tracker)

    monkeypatch.setattr("rasa.core.agent.Agent.handle_message", mock_handle_message)

    return E2ETestRunner()


@pytest.fixture
def assertions_tracker(default_agent: Agent) -> DialogueStateTracker:
    tracker = DialogueStateTracker.from_events(
        "test_assertions_tracker",
        [
            SessionStarted(),
            UserUttered("send money"),
            FlowStarted("transfer_money"),
            FlowStarted("check_balance"),
            FlowCancelled("check_balance", step_id="action_check_balance"),
            SlotSet("amount", 100),
            BotUttered(
                "How would you like to transfer?",
                data={
                    "buttons": [
                        {"title": "Bank transfer", "payload": "/transfer_bank"},
                        {"title": "Card payment", "payload": "/card_payment"},
                    ]
                },
                metadata={"utter_action": "utter_ask_transfer_method"},
            ),
            ActionExecuted(action_name="action_transfer_money"),
            FlowCompleted("transfer_money", step_id="action_transfer_money"),
        ],
    )

    return tracker


@pytest.fixture
def assertions_tracker_with_duplicate_user_msg(
    default_agent: Agent,
) -> DialogueStateTracker:
    tracker = DialogueStateTracker(
        "test_assertions_tracker_duplicate_user_msg",
        [],
    )
    tracker.model_id = "test_model"
    tracker.assistant_id = "test_assistant"

    tracker.update_with_events(
        [
            SessionStarted(),
            UserUttered("send money"),
            BotUttered("How much would you like to transfer?"),
            UserUttered("100 dollars"),
            SlotSet("amount", 100),
            BotUttered("Who is the recipient"),
            UserUttered("Jane Doe"),
            SlotSet("recipient", "Jane Doe"),
            BotUttered("Please confirm if you'd like to proceed with the transfer?"),
            UserUttered("Yes", metadata={"turn_idx": 1}),
            BotUttered("Transfer completed, anything else I can help you with?"),
            UserUttered("Please make the same transfer to John Doe"),
            SlotSet("recipient", "John Doe"),
            SlotSet("amount", 100),
            BotUttered("Please confirm if you'd like to proceed with the transfer?"),
            UserUttered("Yes", metadata={"turn_idx": 2}),
            FlowCompleted("transfer_money", step_id="action_transfer_money"),
        ],
    )

    return tracker


@pytest.fixture
def assertions_e2e_test_runner(
    default_agent: Agent,
    assertions_tracker: DialogueStateTracker,
    monkeypatch: MonkeyPatch,
) -> E2ETestRunner:
    def mock_init(self, *args, **kwargs) -> None:
        self.agent = default_agent

    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.__init__", mock_init
    )

    test_runner = E2ETestRunner()

    async def mock_get_tracker(self, *args, **kwargs) -> DialogueStateTracker:
        return assertions_tracker

    monkeypatch.setattr(test_runner.agent.processor, "get_tracker", mock_get_tracker)

    monkeypatch.setattr(test_runner.agent.tracker_store, "retrieve", mock_get_tracker)

    return test_runner


def test_generate_test_result_successful() -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
            ],
        ),
        1: TestStep.from_dict({"bot": "Hey! How are you?"}),
        2: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I would like to book a trip."}),
            [
                UserUttered("I would like to book a trip."),
                BotUttered("Where would you like to travel?"),
            ],
        ),
        3: TestStep.from_dict({"bot": "Where would you like to travel?"}),
        4: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I want to go to Lisbon."}),
            [
                UserUttered("I want to go to Lisbon."),
                BotUttered("Your trip to Lisbon has been booked."),
            ],
        ),
        5: TestStep.from_dict({"bot": "Your trip to Lisbon has been booked."}),
    }

    test_case = TestCase(
        steps=[
            TestStep.from_dict({"user": "Hi!"}),
            TestStep.from_dict({"bot": "Hey! How are you?"}),
            TestStep.from_dict({"user": "I would like to book a trip."}),
            TestStep.from_dict({"bot": "Where would you like to travel?"}),
            TestStep.from_dict({"user": "I want to go to Lisbon."}),
            TestStep.from_dict({"bot": "Your trip to Lisbon has been booked."}),
        ],
        name="book_trip_successful",
    )

    result = E2ETestRunner.generate_test_result(test_turns, test_case)

    assert result.pass_status is True
    assert result.difference == []


@pytest.mark.parametrize(
    "events, expected_cursor",
    [
        (
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
                UserUttered("I would like to book a trip."),
                BotUttered("Where would you like to travel?"),
                UserUttered("I want to go to Lisbon."),
                ActionExecuted("action_one"),
                ActionExecuted("action_two"),
                BotUttered("Your trip to Madrid has been booked."),
            ],
            8,
        ),
        (
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
                UserUttered("I would like to book a trip."),
                BotUttered("Where would you like to travel?"),
                UserUttered("I want to go to Lisbon."),
                ActionExecuted("action_listen"),
                BotUttered("Your trip to Madrid has been booked."),
            ],
            7,
        ),
    ],
)
def test_get_actual_step_output(events: List[Event], expected_cursor: int) -> None:
    tracker = DialogueStateTracker.from_events("default", evts=events)
    event_cursor = 5
    test_step = TestStep.from_dict({"user": "I want to go to Lisbon."})
    output, event_cursor = E2ETestRunner.get_actual_step_output(
        tracker, test_step, event_cursor
    )

    for event in output.events:
        assert event.text in [
            "I want to go to Lisbon.",
            "Your trip to Madrid has been booked.",
        ]
    assert event_cursor == expected_cursor


def test_generate_test_result_failed() -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
            ],
        ),
        1: TestStep.from_dict({"bot": "Hey! How are you?"}),
        2: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I would like to book a trip."}),
            [
                UserUttered("I would like to book a trip."),
                BotUttered("Where would you like to travel?"),
            ],
        ),
        3: TestStep.from_dict({"bot": "Where would you like to travel?"}),
        4: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I want to go to Lisbon."}),
            [
                UserUttered("I want to go to Lisbon."),
                BotUttered("Your trip to Madrid has been booked."),
            ],
        ),
        5: TestStep.from_dict({"bot": "Your trip to Lisbon has been booked."}),
    }

    test_case = TestCase(
        steps=[
            TestStep.from_dict({"user": "Hi!"}),
            TestStep.from_dict({"bot": "Hey! How are you?"}),
            TestStep.from_dict({"user": "I would like to book a trip."}),
            TestStep.from_dict({"bot": "Where would you like to travel?"}),
            TestStep.from_dict({"user": "I want to go to Lisbon."}),
            TestStep.from_dict({"bot": "Your trip to Lisbon has been booked."}),
        ],
        name="book_trip_failed",
    )

    result = E2ETestRunner.generate_test_result(test_turns, test_case)
    assert result.pass_status is False

    difference = [
        "  user: Hi!",
        "  bot: Hey! How are you?",
        "  user: I would like to book a trip.",
        "  bot: Where would you like to travel?",
        "  user: I want to go to Lisbon.",
        "- bot: Your trip to Madrid has been booked.",
        "?                   ^^^^ ^\n",
        "+ bot: Your trip to Lisbon has been booked.",
        "?                   ^ ^^^^\n",
    ]
    assert result.difference == difference


@pytest.mark.parametrize(
    "test_turns, expected_diff",
    [
        (
            {
                -1: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"bot": "Test"}),
                    [],
                ),
                0: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"user": "I'd like to check my order status."}),
                    [
                        UserUttered("I'd like to check my order status."),
                        BotUttered("Please confirm your name."),
                    ],
                ),
                1: TestStep.from_dict({"bot": "Please confirm your name."}),
                2: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"user": "John Smith"}),
                    [
                        UserUttered("John Smith"),
                        BotUttered(
                            "Hey! How are you?",
                            metadata={"utter_action": "utter_greet"},
                        ),
                    ],
                ),
                3: TestStep.from_dict({"slot_was_set": "name"}),
                4: TestStep.from_dict({"utter": "utter_ask_order_id"}),
                5: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"user": "GB901234AZ"}),
                    [
                        UserUttered("GB901234AZ"),
                        BotUttered(
                            "Great, carry on!", metadata={"utter_action": "utter_greet"}
                        ),
                        BotUttered(
                            "Great, carry on!", metadata={"utter_action": "utter_greet"}
                        ),
                    ],
                ),
                6: TestStep.from_dict({"slot_was_set": "order_id"}),
                7: TestStep.from_dict({"utter": "utter_ask_postcode"}),
                8: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"user": "E16 4PQ"}),
                    [
                        UserUttered("E16 4PQ"),
                        BotUttered("Great, carry on!"),
                    ],
                ),
                9: TestStep.from_dict({"slot_was_set": "postcode"}),
                10: TestStep.from_dict(
                    {"bot": "Your current order status is: **shipped**"}
                ),
            },
            [
                "  user: I'd like to check my order status.",
                "  bot: Please confirm your name.",
                "  user: John Smith",
                "- * No Slot Set *",
                "- bot: utter_greet",
                "+ slot_was_set: name",
                "+ bot: utter_ask_order_id",
                "  user: GB901234AZ",
                "- * No Slot Set *",
                "- bot: utter_greet",
                "+ slot_was_set: order_id",
                "+ bot: utter_ask_postcode",
                "  user: E16 4PQ",
                "- * No Slot Set *",
                "- bot: Great, carry on!",
                "+ slot_was_set: postcode",
                "+ bot: Your current order status is: **shipped**",
            ],
        ),
        (
            {
                -1: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"bot": "Test"}),
                    [SlotSet("trace", "23123")],
                ),
                0: TestStep.from_dict({"slot_was_set": {"trace": "x-tr-23123"}}),
                1: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"user": "Hi!"}),
                    [
                        UserUttered("Hi!"),
                        BotUttered("Please confirm your name."),
                    ],
                ),
                2: TestStep.from_dict({"bot": "Please confirm your full name."}),
                3: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"user": "John Smith"}),
                    [
                        UserUttered("John Smith"),
                        SlotSet("name", {"First": "John", "Last": "Smith"}),
                        BotUttered("Hi John! What is your order number?"),
                    ],
                ),
                4: TestStep.from_dict({"slot_was_set": {"name": "John Smith"}}),
                5: TestStep.from_dict(
                    {"bot": "Hi John Smith! What is your order number?"}
                ),
                6: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"user": "123456"}),
                    [
                        UserUttered("123456"),
                        SlotSet("order_number", "123456"),
                        BotUttered("Checking order 123456!"),
                    ],
                ),
                7: TestStep.from_dict({"slot_was_set": {"order_number": 123456}}),
                8: TestStep.from_dict({"bot": "Checking order number 123456!"}),
            },
            [
                "- slot_was_set: trace: 23123 (str)",
                "+ slot_was_set: trace: x-tr-23123 (str)",
                "?                      +++++\n",
                "  user: Hi!",
                "- bot: Please confirm your name.",
                "+ bot: Please confirm your full name.",
                "?                         +++++\n",
                "  user: John Smith",
                "- slot_was_set: name: {'First': 'John', 'Last': 'Smith'} (dict)",
                "+ slot_was_set: name: John Smith (str)",
                "- bot: Hi John! What is your order number?",
                "+ bot: Hi John Smith! What is your order number?",
                "?             ++++++\n",
                "  user: 123456",
                "- slot_was_set: order_number: 123456 (str)",
                "?                                     ^ -\n",
                "+ slot_was_set: order_number: 123456 (int)",
                "?                                     ^^\n",
                "- bot: Checking order 123456!",
                "+ bot: Checking order number 123456!",
                "?                    +++++++\n",
            ],
        ),
        (
            {  # Test for multiple slots, subset, order agnostic FAILURE
                -1: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"bot": "Test"}),
                    [BotUttered("Welcome to xyz store.")],
                ),
                0: TestStep.from_dict({"bot": "Welcome to store."}),
                1: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"user": "Hi!"}),
                    [
                        UserUttered("Hi!"),
                        SlotSet("user_number", "123456"),
                        SlotSet("trace", "AB123-456DC-231xCS"),
                        SlotSet("cloud_inst", "x-aws-a231"),
                        SlotSet("random_num", "0x123456789"),
                        BotUttered("Please confirm your name."),
                    ],
                ),
                2: TestStep.from_dict({"slot_was_set": {"user_number": "123456"}}),
                3: TestStep.from_dict({"slot_was_set": {"cloud_inst": "x-aws-a123"}}),
                4: TestStep.from_dict({"bot": "Please confirm your name."}),
                5: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"user": "John Smith"}),
                    [
                        UserUttered("John Smith"),
                        BotUttered("Thank you."),
                    ],
                ),
                6: TestStep.from_dict({"bot": "Thank you."}),
            },
            [
                "- bot: Welcome to xyz store.",
                "?                ----\n",
                "+ bot: Welcome to store.",
                "  user: Hi!",
                "  slot_was_set: user_number: 123456 (str)",
                "- slot_was_set: cloud_inst: x-aws-a231 (str)",
                "?                                    -\n",
                "+ slot_was_set: cloud_inst: x-aws-a123 (str)",
                "?                                  +\n",
            ],
        ),
        (
            {  # Test for multiple slots, subset, correction, FAILURE
                -1: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"bot": "Test"}),
                    [],
                ),
                0: ActualStepOutput.from_test_step(
                    TestStep.from_dict(
                        {
                            "user": (
                                "I want to book a table for 4 "
                                "at Xaigon for 7pm tonight"
                            )
                        }
                    ),
                    [
                        UserUttered(
                            "I want to book a table for 4 at Xaigon for 7pm tonight"
                        ),
                        SlotSet("book_restaurant_name_of_restaurant", "Xaigon"),
                        SlotSet("book_restaurant_number_of_people", "4"),
                        SlotSet("book_restaurant_date", "today"),
                        SlotSet("book_restaurant_time", "7pm"),
                        BotUttered("Hum, the restaurant is not available."),
                        BotUttered("Do you want to alternatively book at 8pm?"),
                    ],
                ),
                1: TestStep.from_dict(
                    {"slot_was_set": {"book_restaurant_name_of_restaurant": "Xaigon"}}
                ),
                2: TestStep.from_dict(
                    {"slot_was_set": {"book_restaurant_number_of_people": "5"}}
                ),
                3: TestStep.from_dict(
                    {"slot_was_set": {"book_restaurant_date": "today"}}
                ),
                4: TestStep.from_dict(
                    {"slot_was_set": {"book_restaurant_time": "7pm"}}
                ),
                5: TestStep.from_dict({"bot": "Hum, the restaurant is not available."}),
                6: TestStep.from_dict(
                    {"bot": "Do you want to alternatively book at 8pm?"}
                ),
                7: ActualStepOutput.from_test_step(
                    TestStep.from_dict({"user": "Yes"}),
                    [
                        UserUttered("Yes"),
                        SlotSet("book_restaurant_time", "8PM"),
                        BotUttered("All good!"),
                    ],
                ),
                8: TestStep.from_dict(
                    {"slot_was_set": {"book_restaurant_time": "8pm"}}
                ),
                9: TestStep.from_dict({"bot": "All good!"}),
            },
            [
                "  user: I want to book a table for 4 at Xaigon for 7pm tonight",
                "  slot_was_set: book_restaurant_name_of_restaurant: Xaigon (str)",
                "- slot_was_set: book_restaurant_number_of_people: 4 (str)",
                "?                                                 ^\n",
                "+ slot_was_set: book_restaurant_number_of_people: 5 (str)",
                "?                                                 ^\n",
                "  slot_was_set: book_restaurant_date: today (str)",
                "  slot_was_set: book_restaurant_time: 7pm (str)",
                "  bot: Hum, the restaurant is not available.",
                "  bot: Do you want to alternatively book at 8pm?",
                "  user: Yes",
                "- slot_was_set: book_restaurant_time: 8PM (str)",
                "?                                      ^^\n",
                "+ slot_was_set: book_restaurant_time: 8pm (str)",
                "?                                      ^^\n",
            ],
        ),
    ],
)
def test_human_readable_diff_all_wrong(
    test_turns: Dict[int, Union[ActualStepOutput, TestStep]], expected_diff: List[Text]
) -> None:
    fail_positions = E2ETestRunner.find_test_failures(test_turns, TestCase("test", []))
    diff = E2ETestRunner.human_readable_diff(test_turns, fail_positions)

    assert diff == expected_diff


def test_fail_diff_with_wrong_order_of_bot_utterances() -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [SlotSet("trace", "23123")],
        ),
        0: TestStep.from_dict({"slot_was_set": {"trace": "23123"}}),
        1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("Hi!"),
                BotUttered("Please confirm your name."),
                BotUttered("Or provide some other identification."),
            ],
        ),
        2: TestStep.from_dict({"bot": "Please confirm your name."}),
        3: TestStep.from_dict({"bot": "By typing it bellow this message."}),
        4: TestStep.from_dict({"bot": "Or provide some other identification."}),
    }
    fail_positions = E2ETestRunner.find_test_failures(test_turns, TestCase("test", []))
    diff = E2ETestRunner.human_readable_diff(test_turns, fail_positions)

    assert diff == [
        "  slot_was_set: trace: 23123 (str)",
        "  user: Hi!",
        "  bot: Please confirm your name.",
        "- * No Bot Response *",
        "+ bot: By typing it bellow this message.",
    ]


def test_bot_did_not_catch_the_user_message() -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [SlotSet("trace", "23123")],
        ),
        0: TestStep.from_dict({"slot_was_set": {"trace": "23123"}}),
        1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [],
        ),
        2: TestStep.from_dict({"bot": "Please confirm your name."}),
        3: TestStep.from_dict({"bot": "Or provide some other identification."}),
    }
    fail_positions = E2ETestRunner.find_test_failures(test_turns, TestCase("test", []))
    with pytest.raises(RasaException) as error_info:
        E2ETestRunner.human_readable_diff(test_turns, fail_positions)

    assert "Bot did not catch user event: user: Hi!." in str(error_info)


@pytest.mark.parametrize(
    "test_turns",
    [
        {  # test for float type loaded with ruamel
            -1: ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Test"}),
                [],
            ),
            0: ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    BotUttered("How much money do you need?"),
                ],
            ),
            1: TestStep.from_dict({"bot": "How much money do you need?"}),
            2: ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "John Smith"}),
                [
                    UserUttered("$22.56"),
                    SlotSet("amount_of_money", "22.56"),
                    BotUttered("Sounds good!"),
                ],
            ),
            3: TestStep.from_dict(
                {"slot_was_set": {"amount_of_money": DoubleQuotedScalarString("22.56")}}
            ),
            4: TestStep.from_dict({"bot": "Sounds good!"}),
        },
        {  # Test for skipping slots correctly
            -1: ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Test"}),
                [
                    SlotSet("trace", "AB123-456DC-231xCS"),
                ],
            ),
            0: TestStep.from_dict({"slot_was_set": {"trace": "AB123-456DC-231xCS"}}),
            1: ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    SlotSet("user_number", "123456"),
                    SlotSet("cloud_inst", "x-aws-a231"),
                    SlotSet("random_num", "0x123456789"),
                    BotUttered("Please confirm your name."),
                ],
            ),
            2: TestStep.from_dict({"bot": "Please confirm your name."}),
            3: ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "John Smith"}),
                [
                    UserUttered("John Smith"),
                    BotUttered("Thank you."),
                ],
            ),
            4: TestStep.from_dict({"bot": "Thank you."}),
        },
        {  # Test for multiple slots, order agnostic
            -1: ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Test"}),
                [],
            ),
            0: ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    SlotSet("user_number", "123456"),
                    SlotSet("cloud_inst", "x-aws-a231"),
                    BotUttered("Please confirm your name."),
                ],
            ),
            1: TestStep.from_dict({"slot_was_set": {"user_number": "123456"}}),
            2: TestStep.from_dict({"slot_was_set": {"cloud_inst": "x-aws-a231"}}),
            3: TestStep.from_dict({"bot": "Please confirm your name."}),
            4: ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "John Smith"}),
                [
                    UserUttered("John Smith"),
                    BotUttered("Thank you."),
                ],
            ),
            5: TestStep.from_dict({"bot": "Thank you."}),
        },
        {  # Test for multiple slots, subset, order agnostic
            -1: ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Test"}),
                [],
            ),
            0: ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    SlotSet("user_number", "123456"),
                    SlotSet("trace", "AB123-456DC-231xCS"),
                    SlotSet("cloud_inst", "x-aws-a231"),
                    SlotSet("random_num", "0x123456789"),
                    BotUttered("Please confirm your name."),
                ],
            ),
            1: TestStep.from_dict({"slot_was_set": {"user_number": "123456"}}),
            2: TestStep.from_dict({"slot_was_set": {"cloud_inst": "x-aws-a231"}}),
            3: TestStep.from_dict({"bot": "Please confirm your name."}),
            4: ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "John Smith"}),
                [
                    UserUttered("John Smith"),
                    BotUttered("Thank you."),
                ],
            ),
            5: TestStep.from_dict({"bot": "Thank you."}),
        },
        {
            -1: ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Test"}),
                [],
            ),
            0: ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    BotUttered("Hey!"),
                ],
            ),
            1: TestStep.from_dict({"bot": "Hey!"}),
        },
        {
            -1: ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Test"}),
                [],
            ),
        },  # empty test case
        {
            -1: ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Test"}),
                [],
            ),
            0: ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    BotUttered("Hey!"),
                    SlotSet("name", "Rasa"),
                    SlotSet("age", -1),
                    SlotSet("city", "Undisclosed"),
                ],
            ),
            1: TestStep.from_dict({"bot": "Hey!"}),
        },
    ],
)
def test_find_test_failures_pass(test_turns: TEST_TURNS_TYPE) -> None:
    fail_positions = E2ETestRunner.find_test_failures(test_turns, TestCase("test", []))
    assert len(fail_positions) == 0


def test_find_test_failure_slots_ending() -> None:
    """Test when the actual events list ends with a SlotSet event.

    This should fail the test case when the test case still has more expected events.
    """
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("Hi!"),
                BotUttered("Hey!"),
                SlotSet("name", "Rasa"),
                SlotSet("age", -1),
                SlotSet("city", "Undisclosed"),
            ],
        ),
        1: TestStep.from_dict({"bot": "Hey!"}),
        2: TestStep.from_dict({"bot": "How are you doing?"}),
    }
    failures = E2ETestRunner.find_test_failures(test_turns, TestCase("test", []))
    first_failure_index = failures[0][1]
    assert len(failures) != 0
    assert first_failure_index == 2


@pytest.mark.parametrize(
    "test_case_name, slot_name, expected_slot_value",
    [
        ("test_premium_booking", "membership_type", "premium"),
        ("test_standard_booking", "membership_type", "standard"),
        ("test_mood_great", "", None),
    ],
)
async def test_set_up_fixtures(
    monkeypatch: MonkeyPatch,
    test_case_name: Text,
    slot_name: Text,
    expected_slot_value: Text,
) -> None:
    def mock_init(self: Any, *args: Any, **kwargs: Any) -> None:
        domain = Domain.from_dict(
            {
                "entities": ["membership_type"],
                "slots": {
                    "membership_type": {
                        "type": "text",
                        "mappings": [
                            {"type": "from_entity", "entity": "membership_type"}
                        ],
                    }
                },
            }
        )
        self.agent = Agent(
            domain=domain,
            tracker_store=InMemoryTrackerStore(domain=domain),
        )
        processor = AsyncMock()
        # using the actual tracker store instead of a mocked one
        processor.fetch_tracker_with_initial_session = (
            self.agent.tracker_store.get_or_create_tracker
        )
        self.agent.processor = processor

    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.__init__", mock_init
    )

    runner = E2ETestRunner()

    assert runner.agent is not None
    assert runner.agent.tracker_store is not None

    fixture_path = (
        Path(__file__).parent.parent.parent / "data" / "end_to_end_testing_input_files"
    )
    test_suite = rasa.cli.e2e_test.read_test_cases(str(fixture_path))

    test_case = next(
        iter(filter(lambda x: x.name == test_case_name, test_suite.test_cases))
    )
    test_fixtures = runner.filter_fixtures_for_test_case(test_case, test_suite.fixtures)
    sender_id = f"{test_case.name}_{datetime.datetime.now()}"

    await runner.set_up_fixtures(test_fixtures, sender_id=sender_id)

    tracker = await runner.agent.tracker_store.get_or_create_tracker(sender_id)
    assert tracker.sender_id == sender_id
    assert tracker.get_slot(slot_name) == expected_slot_value


@pytest.mark.parametrize("slot_was_set", ["location", {"location": "Paris"}])
def test_find_test_failure_with_slot_was_set_step(
    slot_was_set: Union[Text, Dict],
) -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                SlotSet("membership_type", "premium"),
                UserUttered("Hi!"),
                BotUttered("Hey! How can I help?"),
            ],
        ),
        1: TestStep.from_dict({"bot": "Hey! How can I help?"}),
        2: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I would like to book a trip."}),
            [
                UserUttered("I would like to book a trip."),
                BotUttered("Ok, where would you like to travel?"),
            ],
        ),
        3: TestStep.from_dict({"bot": "Ok, where would you like to travel?"}),
        4: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I want to go to Paris."}),
            [
                UserUttered("I want to go to Paris."),
                SlotSet("location", "Paris"),
                BotUttered("Paris is a great city! Let me check the flights."),
            ],
        ),
        5: TestStep.from_dict({"slot_was_set": slot_was_set}),
        6: TestStep.from_dict(
            {"bot": "Paris is a great city! Let me check the flights."}
        ),
    }

    assert len(E2ETestRunner.find_test_failures(test_turns, TestCase("test", []))) == 0


@pytest.mark.parametrize("slot_was_set", ["location", {"location": "Paris"}])
def test_find_test_failure_with_slot_was_set_step_fail(
    slot_was_set: Union[Text, Dict],
) -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                SlotSet("membership_type", "premium"),
                UserUttered("Hi!"),
                BotUttered("Hey! How can I help?"),
            ],
        ),
        1: TestStep.from_dict({"bot": "Hey! How can I help?"}),
        2: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I would like to book a trip."}),
            [
                UserUttered("I would like to book a trip."),
                BotUttered("Ok, where would you like to travel?"),
            ],
        ),
        3: TestStep.from_dict({"bot": "Ok, where would you like to travel?"}),
        4: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I want to go to Paris."}),
            [
                UserUttered("I want to go to Paris."),
                SlotSet("current_location", "Berlin"),
                # missing SetSlot event by design
                BotUttered("Paris is a great city! Let me check the flights."),
            ],
        ),
        5: TestStep.from_dict({"slot_was_set": slot_was_set}),
        6: TestStep.from_dict(
            {"bot": "Paris is a great city! Let me check the flights."}
        ),
    }

    failures = E2ETestRunner.find_test_failures(test_turns, TestCase("test", []))
    assert len(failures) == 1
    first_failure_index = failures[0][1]
    assert first_failure_index == 5


async def test_run_prediction_loop(
    test_suite_metadata: List[Metadata],
    test_case_metadata: Metadata,
    mock_e2e_test_runner: E2ETestRunner,
) -> None:
    assert mock_e2e_test_runner.agent is not None
    assert mock_e2e_test_runner.agent.tracker_store is not None

    collector = CollectingOutputChannel()
    steps = [
        TestStep.from_dict({"user": "Hi!"}),
        TestStep.from_dict({"bot": "Hey! How can I help?"}),
        TestStep.from_dict(
            {"user": "I would like to book a trip.", "metadata": "user_info"}
        ),
        TestStep.from_dict({"bot": "Ok, where would you like to travel?"}),
        TestStep.from_dict({"user": "I want to go to Paris."}),
        TestStep.from_dict({"bot": "Paris is a great city! Let me check the flights."}),
    ]
    sender_id = "test_run_prediction_loop"
    test_turns = await mock_e2e_test_runner.run_prediction_loop(
        collector, steps, sender_id, test_case_metadata, test_suite_metadata
    )
    # there should be one more turn than steps because we capture events before
    # the first step in -1 index
    assert len(test_turns) == len(steps) + 1
    for i in [pos for pos, _ in enumerate(steps) if steps[pos].actor == "user"]:
        assert isinstance(test_turns[i], ActualStepOutput)


async def test_run_prediction_loop_warning_for_no_user_text(
    mock_e2e_test_runner: E2ETestRunner,
) -> None:
    collector = CollectingOutputChannel()
    steps = [
        TestStep.from_dict({"user": ""}),
        TestStep.from_dict({"bot": "Hey! How can I help?"}),
    ]
    sender_id = "test_run_prediction_loop_with_warning"

    match_msg = (
        f"The test case '{sender_id}' contains a `user` step in line "
        f"1 without a text value. Skipping this step and proceeding "
        f"to the next user step."
    )

    with pytest.warns(UserWarning, match=match_msg):
        await mock_e2e_test_runner.run_prediction_loop(collector, steps, sender_id)


@pytest.mark.parametrize("fail_fast, expected_len", [(True, 1), (False, 2)])
async def test_run_tests_with_fail_fast(
    monkeypatch: MonkeyPatch,
    fail_fast: bool,
    expected_len: int,
    test_suite_metadata: List[Metadata],
) -> None:
    test_cases = [
        TestCase(
            steps=[
                TestStep.from_dict({"user": "Hi!"}),
                TestStep.from_dict({"bot": "Hey! How can I help?"}),
                TestStep.from_dict({"user": "Hello!", "metadata": "user_info"}),
            ],
            name="test_hi",
            fixture_names=["premium"],
            metadata_name="device_info",
        ),
        TestCase(
            steps=[
                TestStep.from_dict({"user": "Who are you?"}),
                TestStep.from_dict({"bot": "I am a bot powered by Rasa."}),
            ],
            name="test_bot",
        ),
    ]
    test_fixtures = [Fixture(name="premium", slots_set={"premium": True})]

    def mock_init(self: Any, *args: Any, **kwargs: Any) -> None:
        domain = Domain.empty()
        self.agent = Agent(
            domain=domain, tracker_store=InMemoryTrackerStore(domain=domain)
        )
        processor = AsyncMock()
        # using the actual tracker store instead of a mocked one
        processor.fetch_tracker_with_initial_session = (
            self.agent.tracker_store.get_or_create_tracker
        )
        self.agent.processor = processor

    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.__init__", mock_init
    )
    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.run_prediction_loop",
        AsyncMock(),
    )

    generate_test_result_mock = Mock()
    generate_test_result_mock.return_value = TestResult(
        test_case=test_cases[0], pass_status=False, difference=[]
    )

    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.generate_test_result",
        generate_test_result_mock,
    )

    runner = E2ETestRunner()

    results = await runner.run_tests(
        test_cases,
        test_fixtures,
        fail_fast=fail_fast,
        input_metadata=test_suite_metadata,
    )

    assert len(results) == expected_len
    assert results[0] == TestResult(
        test_case=test_cases[0], pass_status=False, difference=[]
    )


async def test_run_tests_for_fine_tuning(monkeypatch: MonkeyPatch):
    test_cases = [
        TestCase(
            steps=[
                TestStep.from_dict({"user": "Hi!"}),
                TestStep.from_dict({"bot": "Hey! How can I help?"}),
                TestStep.from_dict({"user": "Hello!", "metadata": "user_info"}),
            ],
            name="test_case_1",
        ),
        TestCase(
            steps=[
                TestStep.from_dict({"user": "Who are you?"}),
                TestStep.from_dict({"bot": "I am a bot powered by Rasa."}),
            ],
            name="test_case_2",
        ),
    ]

    def mock_init(self: Any, *args: Any, **kwargs: Any) -> None:
        domain = Domain.empty()
        self.agent = Agent(
            domain=domain, tracker_store=InMemoryTrackerStore(domain=domain)
        )
        processor = AsyncMock()
        # using the actual tracker store instead of a mocked one
        processor.fetch_tracker_with_initial_session = (
            self.agent.tracker_store.get_or_create_tracker
        )
        self.agent.processor = processor

    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.__init__", mock_init
    )
    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.run_prediction_loop",
        AsyncMock(),
    )

    mock_test_results = MagicMock(
        side_effect=[TestResult(Mock(), True, []), TestResult(Mock(), False, [])]
    )
    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.generate_test_result",
        mock_test_results,
    )

    # Mock generate_conversation to return a conversation object
    mock_generate_conversation = Mock()
    mock_conversation_1 = MagicMock(spec=Conversation)
    mock_conversation_2 = None
    mock_generate_conversation.side_effect = [mock_conversation_1, mock_conversation_2]
    monkeypatch.setattr(
        "rasa.llm_fine_tuning.annotation_module.generate_conversation",
        mock_generate_conversation,
    )

    runner = E2ETestRunner()

    with capture_logs() as logs:
        result = await runner.run_tests_for_fine_tuning(
            test_cases,
            [],
            None,
        )

        assert len(logs) == 1
        assert logs[0]["log_level"] == "warning"
        assert logs[0]["test_case"] == "test_case_2"

    # Verify the result contains only the passing test case conversation
    assert result == [mock_conversation_1]


@pytest.mark.parametrize(
    "default_slot_set_event",
    [
        SlotSet(REQUESTED_SLOT, "location"),
        SlotSet(SESSION_START_METADATA_SLOT, {}),
        SlotSet(SLOT_LISTED_ITEMS, []),
        SlotSet(SLOT_LAST_OBJECT, {}),
        SlotSet(SLOT_LAST_OBJECT_TYPE, ""),
    ],
)
def test_generate_test_result_asserts_default_slots(
    default_slot_set_event: SlotSet,
) -> None:
    default_slot_key = default_slot_set_event.key
    default_slot_value = default_slot_set_event.value

    test_case = TestCase(
        steps=[
            TestStep.from_dict({"user": "Hi!"}),
            TestStep.from_dict({"bot": "Hey! How are you?"}),
            TestStep.from_dict({"user": "I would like to book a trip."}),
            TestStep.from_dict({"bot": "Where would you like to travel?"}),
            TestStep.from_dict(
                {"slot_was_set": {default_slot_key: default_slot_value}}
            ),
            TestStep.from_dict({"user": "I want to go to Lisbon."}),
            TestStep.from_dict({"slot_was_set": {"city": "Lisbon"}}),
            TestStep.from_dict({"bot": "Your trip to Lisbon has been booked."}),
        ],
        name="default",
    )
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
            ],
        ),
        1: TestStep.from_dict({"bot": "Hey! How are you?"}),
        2: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I would like to book a trip."}),
            [
                UserUttered("I would like to book a trip."),
                BotUttered("Where would you like to travel?"),
                default_slot_set_event,
            ],
        ),
        3: TestStep.from_dict({"bot": "Where would you like to travel?"}),
        4: TestStep.from_dict({"slot_was_set": {default_slot_key: default_slot_value}}),
        5: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I want to go to Lisbon."}),
            [
                UserUttered("I want to go to Lisbon."),
                SlotSet("city", "Lisbon"),
                BotUttered("Your trip to Lisbon has been booked."),
            ],
        ),
        6: TestStep.from_dict({"slot_was_set": {"city": "Lisbon"}}),
        7: TestStep.from_dict({"bot": "Your trip to Lisbon has been booked."}),
    }

    result = E2ETestRunner.generate_test_result(test_turns, test_case)

    assert result.pass_status is True
    assert result.difference == []


def test_generate_result_with_multiple_consecutive_slot_was_set_steps() -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
            ],
        ),
        1: TestStep.from_dict({"bot": "Hey! How are you?"}),
        2: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I would like to book a trip."}),
            [
                UserUttered("I would like to book a trip."),
                BotUttered("Where would you like to travel?"),
            ],
        ),
        3: TestStep.from_dict({"bot": "Where would you like to travel?"}),
        4: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I want to go to Lisbon from London."}),
            [
                UserUttered("I want to go to Lisbon from London."),
                SlotSet("departure", "London"),
                SlotSet("destination", "Lisbon"),
                BotUttered("Your trip to Lisbon has been booked."),
            ],
        ),
        5: TestStep.from_dict({"slot_was_set": {"departure": "London"}}),
        6: TestStep.from_dict({"slot_was_set": {"destination": "Lisbon"}}),
        7: TestStep.from_dict({"bot": "Your trip to Lisbon has been booked."}),
    }

    test_case = TestCase(
        steps=[
            TestStep.from_dict({"user": "Hi!"}),
            TestStep.from_dict({"bot": "Hey! How are you?"}),
            TestStep.from_dict({"user": "I would like to book a trip."}),
            TestStep.from_dict({"bot": "Where would you like to travel?"}),
            TestStep.from_dict({"user": "I want to go to Lisbon from London."}),
            TestStep.from_dict({"slot_was_set": {"departure": "London"}}),
            TestStep.from_dict({"slot_was_set": {"destination": "Lisbon"}}),
            TestStep.from_dict({"bot": "Your trip to Lisbon has been booked."}),
        ],
        name="book_trip",
    )

    result = E2ETestRunner.generate_test_result(test_turns, test_case)
    assert result.pass_status is True
    assert result.difference == []


def test_generate_result_with_slot_was_set_to_none() -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
            ],
        ),
        1: TestStep.from_dict({"bot": "Hey! How are you?"}),
        2: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I would like to cancel a trip."}),
            [
                UserUttered("I would like to cancel a trip."),
                BotUttered("Which trip would you like to cancel?"),
            ],
        ),
        3: TestStep.from_dict({"bot": "Which trip would you like to cancel?"}),
        4: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I want to cancel my trip to Lisbon."}),
            [
                UserUttered("I want to cancel my trip to Lisbon."),
                SlotSet("destination", None),
                BotUttered("Your trip to Lisbon has been cancelled."),
            ],
        ),
        5: TestStep.from_dict({"slot_was_set": {"destination": None}}),
        6: TestStep.from_dict({"bot": "Your trip to Lisbon has been cancelled."}),
    }

    test_case = TestCase(
        steps=[
            TestStep.from_dict({"user": "Hi!"}),
            TestStep.from_dict({"bot": "Hey! How are you?"}),
            TestStep.from_dict({"user": "I would like to cancel a trip."}),
            TestStep.from_dict({"bot": "Which trip would you like to cancel?"}),
            TestStep.from_dict({"user": "I want to cancel my trip to Lisbon."}),
            TestStep.from_dict({"slot_was_set": {"destination": None}}),
            TestStep.from_dict({"bot": "Your trip to Lisbon has been cancelled."}),
        ],
        name="cancel_trip",
    )

    result = E2ETestRunner.generate_test_result(test_turns, test_case)
    assert result.pass_status is True
    assert result.difference == []


def test_generate_result_with_slot_was_set_not_set() -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
            ],
        ),
        1: TestStep.from_dict({"bot": "Hey! How are you?"}),
        2: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I would like to cancel a trip."}),
            [
                UserUttered("I would like to cancel a trip."),
                BotUttered("Which trip would you like to cancel?"),
            ],
        ),
        3: TestStep.from_dict({"bot": "Which trip would you like to cancel?"}),
        4: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I want to cancel my trip to Lisbon."}),
            [
                UserUttered("I want to cancel my trip to Lisbon."),
                BotUttered("Your trip to Lisbon has been cancelled."),
            ],
        ),
        5: TestStep.from_dict({"slot_was_not_set": "destination"}),
        6: TestStep.from_dict({"bot": "Your trip to Lisbon has been cancelled."}),
    }

    test_case = TestCase(
        steps=[
            TestStep.from_dict({"user": "Hi!"}),
            TestStep.from_dict({"bot": "Hey! How are you?"}),
            TestStep.from_dict({"user": "I would like to cancel a trip."}),
            TestStep.from_dict({"bot": "Which trip would you like to cancel?"}),
            TestStep.from_dict({"user": "I want to cancel my trip to Lisbon."}),
            TestStep.from_dict({"slot_was_not_set": "destination"}),
            TestStep.from_dict({"bot": "Your trip to Lisbon has been cancelled."}),
        ],
        name="cancel_trip",
    )

    result = E2ETestRunner.generate_test_result(test_turns, test_case)
    assert result.pass_status is True
    assert result.difference == []


def test_generate_result_with_slot_was_set_to_none_failure() -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
            ],
        ),
        1: TestStep.from_dict({"bot": "Hey! How are you?"}),
        2: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("I would like to cancel a trip."),
                BotUttered("Which trip would you like to cancel?"),
            ],
        ),
        3: TestStep.from_dict({"bot": "Which trip would you like to cancel?"}),
        4: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("I want to cancel my trip to Lisbon."),
                SlotSet("destination", "lisbon"),
                BotUttered("Your trip to Lisbon has been cancelled."),
            ],
        ),
        5: TestStep.from_dict({"slot_was_set": {"destination": None}}),
        6: TestStep.from_dict({"bot": "Your trip to Lisbon has been cancelled."}),
    }

    test_case = TestCase(
        steps=[
            TestStep.from_dict({"user": "Hi!"}),
            TestStep.from_dict({"bot": "Hey! How are you?"}),
            TestStep.from_dict({"user": "I would like to cancel a trip."}),
            TestStep.from_dict({"bot": "Which trip would you like to cancel?"}),
            TestStep.from_dict({"user": "I want to cancel my trip to Lisbon."}),
            TestStep.from_dict({"slot_was_set": {"destination": None}}),
            TestStep.from_dict({"bot": "Your trip to Lisbon has been cancelled."}),
        ],
        name="cancel_trip",
    )

    result = E2ETestRunner.generate_test_result(test_turns, test_case)
    assert result.pass_status is False
    assert result.difference == [
        "  user: Hi!",
        "  bot: Hey! How are you?",
        "  user: I would like to cancel a trip.",
        "  bot: Which trip would you like to cancel?",
        "  user: I want to cancel my trip to Lisbon.",
        "- slot_was_set: destination: lisbon (str)",
        "?                            ^^^^    ^^^\n",
        "+ slot_was_set: destination: None (NoneType)",
        "?                            ^  +  ^^^^^^^^\n",
    ]


def test_generate_result_with_slot_was_set_several_success() -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
            ],
        ),
        1: TestStep.from_dict({"bot": "Hey! How are you?"}),
        2: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("I would like to cancel a trip."),
                BotUttered("Which trip would you like to cancel?"),
            ],
        ),
        3: TestStep.from_dict({"bot": "Which trip would you like to cancel?"}),
        4: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("I want to cancel my trip to Lisbon."),
                SlotSet("action", "cancel"),
                SlotSet("destination", "lisbon"),
                BotUttered("Your trip to Lisbon has been cancelled."),
            ],
        ),
        5: TestStep.from_dict({"slot_was_set": {"action": "cancel"}}),
        6: TestStep.from_dict({"slot_was_set": {"destination": "lisbon"}}),
        7: TestStep.from_dict({"bot": "Your trip to Lisbon has been cancelled."}),
    }

    test_case = TestCase(
        steps=[
            TestStep.from_dict({"user": "Hi!"}),
            TestStep.from_dict({"bot": "Hey! How are you?"}),
            TestStep.from_dict({"user": "I would like to cancel a trip."}),
            TestStep.from_dict({"bot": "Which trip would you like to cancel?"}),
            TestStep.from_dict({"user": "I want to cancel my trip to Lisbon."}),
            TestStep.from_dict({"slot_was_set": {"action": "cancel"}}),
            TestStep.from_dict({"slot_was_set": {"destination": "lisbon"}}),
            TestStep.from_dict({"bot": "Your trip to Lisbon has been cancelled."}),
        ],
        name="cancel_trip",
    )

    result = E2ETestRunner.generate_test_result(test_turns, test_case)
    assert result.pass_status is True
    assert result.difference == []


def test_generate_result_with_slot_was_set_to_number_failure() -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
            ],
        ),
        1: TestStep.from_dict({"bot": "Hey! How are you?"}),
        2: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I would like to book a restaurant"}),
            [
                UserUttered("I would like to book a restaurant"),
                BotUttered("For how many people?"),
            ],
        ),
        3: TestStep.from_dict({"bot": "For how many people?"}),
        4: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "we will be 5 people"}),
            [
                UserUttered("we will be 5 people"),
                SlotSet("party", 5),
                BotUttered("Great. What time?"),
            ],
        ),
        5: TestStep.from_dict({"slot_was_set": {"party": 4}}),
        6: TestStep.from_dict({"bot": "Great. What time?"}),
    }

    test_case = TestCase(
        steps=[
            TestStep.from_dict({"user": "Hi!"}),
            TestStep.from_dict({"bot": "Hey! How are you?"}),
            TestStep.from_dict({"user": "I would like to book a restaurant"}),
            TestStep.from_dict({"bot": "For how many people?"}),
            TestStep.from_dict({"user": "we will be 5 people"}),
            TestStep.from_dict({"slot_was_set": {"party": 4}}),
            TestStep.from_dict({"bot": "Great. What time?"}),
        ],
        name="book_restaurant",
    )

    result = E2ETestRunner.generate_test_result(test_turns, test_case)
    assert result.pass_status is False

    assert result.difference == [
        "  user: Hi!",
        "  bot: Hey! How are you?",
        "  user: I would like to book a restaurant",
        "  bot: For how many people?",
        "  user: we will be 5 people",
        "- slot_was_set: party: 5 (int)",
        "?                      ^\n",
        "+ slot_was_set: party: 4 (int)",
        "?                      ^\n",
    ]


def test_find_test_failure_with_less_actual_events() -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "I need access to building Sequoia."}),
            [
                UserUttered("I need access to building Sequoia."),
                SlotSet("building", "Sequoia"),
                BotUttered("To request building access, proceed via the link below."),
            ],
        ),
        1: TestStep.from_dict(
            {"bot": "To request building access, proceed via the link below."}
        ),
        2: TestStep.from_dict(
            {"bot": "Is there anything else I can help you with today?"}
        ),
    }

    failures = E2ETestRunner.find_test_failures(test_turns, TestCase("test", []))
    assert len(failures) == 1
    first_failure_index = failures[0][1]
    assert first_failure_index == 2


def test_empty_bot_response() -> None:
    dialog = DialogueStateTracker.from_events(
        "test",
        [],
    )
    test_step = TestStep.from_dict({"user": "Hi!"})

    result, cursor = E2ETestRunner.get_actual_step_output(dialog, test_step, 0)
    assert len(result.events) == 2
    assert isinstance(result.events[0], UserUttered)
    assert isinstance(result.events[1], BotUttered)
    assert cursor == 0


def test_action_server_is_reachable(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    endpoint = AvailableEndpoints(
        action=EndpointConfig(url="http://localhost:5055/webhook")
    )
    mock = MagicMock()
    mock.return_value.status_code = 200
    monkeypatch.setattr(rasa.e2e_test.e2e_test_runner.requests, "get", mock)

    with capture_logs() as logs:
        E2ETestRunner._action_server_is_reachable(endpoint)

        assert mock.called
        assert mock.call_args[0][0] == "http://localhost:5055/health"
        assert len(logs) == 2
        assert logs[1]["log_level"] == "debug"
        assert "Action endpoint has responded successfully." in logs[1]["message"]


def test_action_server_is_reachable_bad_response(monkeypatch: MonkeyPatch) -> None:
    endpoint = AvailableEndpoints(
        action=EndpointConfig(url="http://localhost:5055/webhook")
    )
    mock = MagicMock()
    mock.return_value.status_code = 404
    monkeypatch.setattr(rasa.e2e_test.e2e_test_runner.requests, "get", mock)
    with pytest.raises(RasaException) as excinfo:
        E2ETestRunner._action_server_is_reachable(endpoint)
        assert (
            "Action endpoint is responding, but health status responded with 404"
            in str(excinfo.value)
        )

    assert mock.called
    assert mock.call_args[0][0] == "http://localhost:5055/health"


def test_action_server_is_not_reachable(monkeypatch: MonkeyPatch) -> None:
    endpoint = AvailableEndpoints(
        action=EndpointConfig(url="http://localhost:5055/webhook")
    )
    mock = MagicMock()
    mock.return_value.side_effect = requests.exceptions.ConnectionError()
    monkeypatch.setattr(rasa.e2e_test.e2e_test_runner.requests, "get", mock)
    with pytest.raises(RasaException) as excinfo:
        E2ETestRunner._action_server_is_reachable(endpoint)
        assert (
            "Action endpoint could not be reached. "
            "Actions server URL is defined in your endpoint configuration as "
            "'http://localhost:5055/webhook'." in str(excinfo.value)
        )

    assert mock.called
    assert mock.call_args[0][0] == "http://localhost:5055/health"


def test_action_server_is_not_reachable_url_not_defined(
    caplog: LogCaptureFixture,
) -> None:
    endpoint = AvailableEndpoints(action=EndpointConfig())
    with capture_logs() as logs:
        E2ETestRunner._action_server_is_reachable(endpoint)

        assert len(logs) == 1
        assert logs[0]["log_level"] == "debug"
        assert (
            logs[0]["message"] == "Action endpoint URL is not defined in the endpoint "
            "configuration."
        )


def test_action_server_is_not_reachable_action_not_defined(
    caplog: LogCaptureFixture,
) -> None:
    endpoint = AvailableEndpoints()
    with capture_logs() as logs:
        E2ETestRunner._action_server_is_reachable(endpoint)

        assert len(logs) == 1
        assert logs[0]["log_level"] == "debug"
        assert (
            logs[0]["message"] == "No action endpoint configured. Skipping the health "
            "check of the action server."
        )


def test_bot_event_text_message_formatting() -> None:
    test_turns: TEST_TURNS_TYPE = {
        -1: ActualStepOutput.from_test_step(
            TestStep.from_dict({"bot": "Test"}),
            [],
        ),
        0: ActualStepOutput.from_test_step(
            TestStep.from_dict({"user": "Hi!"}),
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
            ],
        ),
        1: TestStep.from_dict({"bot": "\nHey! How are you?\n"}),
    }

    test_case = TestCase(
        steps=[
            TestStep.from_dict({"user": "Hi!"}),
            TestStep.from_dict({"bot": "\nHey! How are you?\n"}),
        ],
        name="bot_message_formatting",
    )

    result = E2ETestRunner.generate_test_result(test_turns, test_case)

    assert result.pass_status is True
    assert result.difference == []


@pytest.mark.parametrize(
    "metadata_name, expected",
    [
        (
            "device_info",
            Metadata(name="device_info", metadata={"os": "linux"}),
        ),
        (
            "incorrect_metadata_name",
            None,
        ),
        ("", None),
    ],
)
def test_filter_metadata_for_input(
    metadata_name: Text, test_suite_metadata, expected: Optional[Metadata]
) -> None:
    result = E2ETestRunner.filter_metadata_for_input(metadata_name, test_suite_metadata)

    assert result == expected


def test_filter_metadata_for_input_undefined_metadata_name(
    caplog: LogCaptureFixture,
) -> None:
    metadata_name = "incorrect_metadata_name"
    with capture_logs() as logs:
        E2ETestRunner.filter_metadata_for_input(
            metadata_name,
            [Metadata(name="device_info", metadata={"os": "linux"})],
        )

        assert len(logs) == 1
        assert (
            f"Metadata '{metadata_name}' is not defined in the input metadata."
            in logs[0]["message"]
        )


@pytest.mark.parametrize(
    "test_case_metadata_dict, step_metadata_dict, expected",
    [
        ({"os": "linux"}, {"name": "Tom"}, {"os": "linux", "name": "Tom"}),
        (
            {"os": "linux"},
            {
                "name": "Tom",
                "os": "windows",
            },
            {"os": "windows", "name": "Tom"},
        ),
        ({}, {"name": "Tom"}, {"name": "Tom"}),
        ({"os": "linux"}, {}, {"os": "linux"}),
        ({}, {}, {}),
    ],
)
def test_merge_metadata(
    test_case_metadata_dict: Dict[Text, Text],
    step_metadata_dict: Dict[Text, Text],
    expected: Dict[Text, Text],
) -> None:
    result = E2ETestRunner.merge_metadata(
        "Test_case", "step_text", test_case_metadata_dict, step_metadata_dict
    )

    assert result == expected


def test_merge_metadata_warning(
    caplog: LogCaptureFixture,
) -> None:
    with capture_logs() as logs:
        E2ETestRunner.merge_metadata(
            "Test_case_name_123",
            "Hi!",
            {"os": "linux"},
            {"name": "Tom", "os": "windows"},
        )

        assert len(logs) == 1
        assert f"Metadata {['os']} exist in both the test case " in logs[0]["message"]
        assert "'Test_case_name' and the user step 'Hi!'. " in logs[0]["message"]


@pytest.mark.parametrize(
    "assertion",
    [
        {"flow_started": "transfer_money"},
        {
            "flow_cancelled": {
                "flow_id": "check_balance",
                "flow_step_id": "action_check_balance",
            }
        },
        {"slot_was_set": [{"name": "amount", "value": 100}]},
        {"slot_was_not_set": [{"name": "card_type"}]},
        {"slot_was_not_set": [{"name": "amount", "value": 200}]},
        {
            "bot_uttered": {
                "text_matches": "How would you like to transfer?",
                "utter_name": "utter_ask_transfer_method",
                "buttons": [
                    {"title": "Bank transfer", "payload": "/transfer_bank"},
                    {"title": "Card payment", "payload": "/card_payment"},
                ],
            }
        },
        {"action_executed": "action_transfer_money"},
        {
            "flow_completed": {
                "flow_id": "transfer_money",
                "flow_step_id": "action_transfer_money",
            }
        },
    ],
)
async def test_run_assertions_all_valid(
    assertion: Dict[str, Any], assertions_e2e_test_runner: E2ETestRunner
) -> None:
    test_case = TestCase(
        name="test_case_transfer_money",
        steps=[
            TestStep.from_dict({"user": "send money", "assertions": [assertion]}),
        ],
    )

    results = await assertions_e2e_test_runner.run_tests(
        [test_case], [], input_metadata=[]
    )
    assert len(results) == 1
    assert isinstance(results[0], TestResult)
    assert results[0].pass_status is True
    assert results[0].difference == []
    assert results[0].assertion_failure is None


async def test_run_assertions_with_duplicate_user_messages(
    default_agent: Agent,
    monkeypatch: MonkeyPatch,
    assertions_tracker_with_duplicate_user_msg: DialogueStateTracker,
) -> None:
    def mock_init(self, *args, **kwargs) -> None:
        self.agent = default_agent

    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.__init__", mock_init
    )

    test_runner = E2ETestRunner()

    async def mock_get_tracker(self, *args, **kwargs) -> DialogueStateTracker:
        return assertions_tracker_with_duplicate_user_msg

    monkeypatch.setattr(test_runner.agent.processor, "get_tracker", mock_get_tracker)

    test_case = TestCase(
        name="test_case_transfer_money",
        steps=[
            TestStep.from_dict(
                {
                    "user": "send money",
                    "assertions": [
                        {
                            "bot_uttered": {
                                "text_matches": "How much would you like to transfer?"
                            }
                        }
                    ],
                }
            ),
            TestStep.from_dict(
                {
                    "user": "100 dollars",
                    "assertions": [
                        {"slot_was_set": [{"name": "amount", "value": 100}]}
                    ],
                }
            ),
            TestStep.from_dict(
                {
                    "user": "Jane Doe",
                    "assertions": [
                        {
                            "slot_was_set": [
                                {"name": "recipient", "value": "Jane Doe"}
                            ],
                            "bot_uttered": {
                                "text_matches": "Please confirm if you'd like to "
                                "proceed with the transfer?"
                            },
                        }
                    ],
                }
            ),
            TestStep.from_dict(
                {
                    "user": "Yes",
                    "assertions": [
                        {
                            "bot_uttered": {
                                "text_matches": "Transfer completed, "
                                "anything else I can help you with?"
                            }
                        }
                    ],
                    "metadata": "turn_1",
                }
            ),
            TestStep.from_dict(
                {
                    "user": "Please make the same transfer to John Doe",
                    "assertions": [
                        {
                            "slot_was_set": [
                                {"name": "recipient", "value": "John Doe"},
                                {"name": "amount", "value": 100},
                            ],
                            "bot_uttered": {
                                "text_matches": "Please confirm if you'd like to "
                                "proceed with the transfer?"
                            },
                        }
                    ],
                }
            ),
            TestStep.from_dict(
                {
                    "user": "Yes",
                    "assertions": [
                        {
                            "flow_completed": {
                                "flow_id": "transfer_money",
                                "flow_step_id": "action_transfer_money",
                            }
                        }
                    ],
                    "metadata": "turn_2",
                }
            ),
        ],
    )

    input_metadata = [
        Metadata(name="turn_1", metadata={"turn_idx": 1}),
        Metadata(name="turn_2", metadata={"turn_idx": 2}),
    ]

    result = await test_runner.run_assertions(
        "test_assertions_tracker_duplicate_user_msg", test_case, input_metadata
    )
    assert isinstance(result, TestResult)
    assert result.pass_status is True
    assert result.assertion_failure is None


async def test_run_assertions_with_user_message_not_found(
    default_agent: Agent,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
) -> None:
    def mock_init(self, *args, **kwargs) -> None:
        self.agent = default_agent

    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.__init__", mock_init
    )

    test_runner = E2ETestRunner()

    async def mock_get_tracker(self, *args, **kwargs) -> DialogueStateTracker:
        return DialogueStateTracker("no_user_event", slots=[])

    monkeypatch.setattr(test_runner.agent.processor, "get_tracker", mock_get_tracker)

    test_case = TestCase(
        name="test_case_transfer_money",
        steps=[
            TestStep.from_dict(
                {
                    "user": "send money",
                    "assertions": [
                        {
                            "bot_uttered": {
                                "text_matches": "How much would you like to transfer?"
                            }
                        }
                    ],
                    "metadata": "test_info",
                }
            ),
        ],
    )

    input_metadata = [
        Metadata(name="test_info", metadata={"foo": "bar"}),
    ]

    result = await test_runner.run_assertions(
        "no_user_event", test_case, input_metadata
    )
    assert isinstance(result, TestResult)
    assert result.pass_status is False
    assert result.assertion_failure is None

    captured = capsys.readouterr()
    error_message = (
        "User message 'send money' was not found in the actual events. "
        "The user message properties which were searched: "
        "{'text': 'send money', 'metadata': {'foo': 'bar', "
        "'model_id': None, 'assistant_id': None}}\n"
    )
    assert error_message in captured.out


def test_get_additional_splitting_conditions(
    assertions_tracker_with_duplicate_user_msg: DialogueStateTracker,
    default_agent: Agent,
    monkeypatch: MonkeyPatch,
) -> None:
    def mock_init(self, *args, **kwargs) -> None:
        self.agent = default_agent

    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.__init__", mock_init
    )

    test_runner = E2ETestRunner()

    async def mock_get_tracker(self, *args, **kwargs) -> DialogueStateTracker:
        return assertions_tracker_with_duplicate_user_msg

    monkeypatch.setattr(test_runner.agent.processor, "get_tracker", mock_get_tracker)

    test_step = TestStep.from_dict(
        {
            "user": "send money",
            "assertions": [
                {
                    "bot_uttered": {
                        "text_matches": "How much would you like to transfer?"
                    }
                }
            ],
            "metadata": "step_info",
        }
    )
    test_case = TestCase(
        name="test_case_transfer_money",
        steps=[test_step],
        metadata_name="test_case_info",
    )

    input_metadata = [
        Metadata(name="step_info", metadata={"foo": "bar"}),
        Metadata(name="test_case_info", metadata={"baz": "qux"}),
    ]

    additional_splitting_conditions = test_runner._get_additional_splitting_conditions(
        test_step, input_metadata, assertions_tracker_with_duplicate_user_msg, test_case
    )

    assert additional_splitting_conditions == {
        "text": test_step.text,
        "metadata": {
            "foo": "bar",
            "baz": "qux",
            "model_id": assertions_tracker_with_duplicate_user_msg.model_id,
            "assistant_id": assertions_tracker_with_duplicate_user_msg.assistant_id,
        },
    }


def test_get_current_user_turn_and_prior_events(
    assertions_e2e_test_runner: E2ETestRunner,
    assertions_tracker: DialogueStateTracker,
) -> None:
    test_step = TestStep.from_dict(
        {
            "user": "send money",
            "assertions": [
                {
                    "bot_uttered": {
                        "text_matches": "How much would you like to transfer?"
                    }
                }
            ],
        }
    )

    additional_splitting_conditions = {"text": test_step.text}

    current_user_turn, prior_events = (
        assertions_e2e_test_runner._get_current_user_turn_and_prior_events(
            assertions_tracker, additional_splitting_conditions, test_step
        )
    )

    assert prior_events == [assertions_tracker.events[0]]
    assert current_user_turn == list(assertions_tracker.events)[1:]


def test_slice_turn_events(
    assertions_e2e_test_runner: E2ETestRunner,
    assertions_tracker: DialogueStateTracker,
) -> None:
    test_step = TestStep.from_dict(
        {
            "user": "send money",
            "assertions": [{"flow_started": "transfer_money"}],
            "assertion_order_enabled": True,
        }
    )

    additional_splitting_conditions = {"text": test_step.text}

    current_user_turn, prior_events = (
        assertions_e2e_test_runner._get_current_user_turn_and_prior_events(
            assertions_tracker, additional_splitting_conditions, test_step
        )
    )

    matching_event = FlowStarted("transfer_money")
    sliced_turn_events, new_prior_events = (
        assertions_e2e_test_runner._slice_turn_events(
            test_step, matching_event, current_user_turn, copy.deepcopy(prior_events)
        )
    )

    # we skip the first event in the current user turn, as it is the user
    # message that we are splitting on
    # we also skip the second event in the user turn, because it was already
    # verified by the assertion
    assert sliced_turn_events == current_user_turn[2:]

    # the skipped events should be added to the prior events
    assert new_prior_events == prior_events + current_user_turn[:2]


@pytest.mark.parametrize(
    "events, expected_flow_paths, expected_tested_commands",
    [
        (
            # No flows started, e.g. a nlu-based assistant was used
            [
                UserUttered("Hi!"),
                BotUttered("Hey! How are you?"),
                UserUttered("I would like to book a trip."),
                BotUttered("Where would you like to travel?"),
                UserUttered("I want to go to Lisbon."),
                ActionExecuted("action_one"),
                ActionExecuted("action_two"),
                BotUttered("Your trip to Madrid has been booked."),
            ],
            [],  # No flow paths
            {},  # No commands were tested
        ),
        (
            # one flow with a bot utterance and a custom action
            [
                UserUttered(
                    "What is the weather like?",
                    parse_data={"commands": [{"command": "ask_weather"}]},
                ),
                DialogueStackUpdated(
                    '[{"op": "add", "path": "/0", "value": {"flow_id": "flow_a", '
                    '"step_id": "START", "frame_type": "regular", "type": "flow"}}]'
                ),
                BotUttered(
                    metadata={
                        ACTIVE_FLOW_METADATA_KEY: "flow_a",
                        STEP_ID_METADATA_KEY: "utter_1",
                    }
                ),
                ActionExecuted(
                    "custom_action_1",
                    metadata={
                        ACTIVE_FLOW_METADATA_KEY: "flow_a",
                        STEP_ID_METADATA_KEY: "custom_action_1",
                    },
                ),
                FlowCompleted("flow_a", "END"),
            ],
            [
                FlowPath(
                    "flow_a",
                    nodes=[
                        PathNode(step_id="utter_1", flow="flow_a"),
                        PathNode(step_id="custom_action_1", flow="flow_a"),
                    ],
                )
            ],
            {"no_flow": {"ask_weather": 1}},  # Commands were tested within flow_a
        ),
        (
            # nested flows with different utterances; flow a not completed
            [
                FlowStarted("flow_a"),
                UserUttered(
                    "Start flow", parse_data={"commands": [{"command": "start_flow"}]}
                ),
                BotUttered(metadata={"utter_action": "utter_1"}),
                FlowStarted("flow_b"),
                UserUttered(
                    "Continue flow",
                    parse_data={"commands": [{"command": "continue_flow"}]},
                ),
                DialogueStackUpdated(
                    '[{"op": "add", "path": "/0", "value": {"flow_id": "flow_a", '
                    '"step_id": "START", "frame_type": "regular", "type": "flow"}}]'
                ),
                BotUttered(
                    metadata={
                        ACTIVE_FLOW_METADATA_KEY: "flow_a",
                        STEP_ID_METADATA_KEY: "utter_1",
                    }
                ),
                DialogueStackUpdated(
                    '[{"op": "add", "path": "/0", "value": {"flow_id": "flow_b", '
                    '"step_id": "START", "frame_type": "regular", "type": "flow"}}]'
                ),
                BotUttered(
                    metadata={
                        ACTIVE_FLOW_METADATA_KEY: "flow_b",
                        STEP_ID_METADATA_KEY: "utter_2",
                    }
                ),
                FlowCompleted("flow_b", "END"),
                BotUttered(
                    metadata={
                        ACTIVE_FLOW_METADATA_KEY: "flow_a",
                        STEP_ID_METADATA_KEY: "utter_3",
                    }
                ),
            ],
            [
                FlowPath(
                    "flow_b",
                    nodes=[PathNode(step_id="utter_2", flow="flow_b")],
                ),
                FlowPath(
                    "flow_a",
                    nodes=[
                        PathNode(step_id="utter_1", flow="flow_a"),
                        PathNode(step_id="utter_3", flow="flow_a"),
                    ],
                ),
            ],
            {
                "no_flow": {"start_flow": 1, "continue_flow": 1}
            },  # Commands were tested within flow_a and flow_b
        ),
        (
            # flow with patterns
            [
                DialogueStackUpdated(
                    '[{"op": "add", "path": "/0", "value": {"flow_id": "flow_a", '
                    '"step_id": "START", "frame_type": "regular", "type": "flow"}}]'
                ),
                BotUttered(
                    metadata={
                        ACTIVE_FLOW_METADATA_KEY: "flow_a",
                        STEP_ID_METADATA_KEY: "utter_1",
                    }
                ),
                DialogueStackUpdated(
                    '[{"op": "add", "path": "/0", "value": {"flow_id": '
                    '"pattern_collect_information", '
                    '"step_id": "START", "frame_type": "regular", "type": "flow"}}]'
                ),
                BotUttered(
                    metadata={
                        ACTIVE_FLOW_METADATA_KEY: "flow_a",
                        STEP_ID_METADATA_KEY: "utter_2",
                    }
                ),
                FlowCompleted("pattern_collect_information", "END"),
                BotUttered(
                    metadata={
                        ACTIVE_FLOW_METADATA_KEY: "flow_a",
                        STEP_ID_METADATA_KEY: "utter_3",
                    }
                ),
                FlowCompleted("flow_a", "END"),
            ],
            [
                FlowPath(
                    "flow_a",
                    nodes=[
                        PathNode(step_id="utter_1", flow="flow_a"),
                        PathNode(step_id="utter_2", flow="flow_a"),
                        PathNode(step_id="utter_3", flow="flow_a"),
                    ],
                )
            ],
            {},  # No commands were tested
        ),
        (
            # flow with direct call step
            [
                DialogueStackUpdated(
                    '[{"op": "add", "path": "/0", "value": {"flow_id": "flow_a", '
                    '"step_id": "START", "frame_type": "regular", "type": "flow"}}]'
                ),
                BotUttered(
                    metadata={
                        ACTIVE_FLOW_METADATA_KEY: "flow_a",
                        STEP_ID_METADATA_KEY: "utter_1",
                    }
                ),
                DialogueStackUpdated(
                    '[{"op": "add", "path": "/0", "value": {"flow_id": "flow_b", '
                    '"step_id": "START", "frame_type": "call", "type": "flow"}}]'
                ),
                BotUttered(
                    metadata={
                        ACTIVE_FLOW_METADATA_KEY: "flow_b",
                        STEP_ID_METADATA_KEY: "utter_2",
                    }
                ),
                FlowCompleted("flow_b", "END"),
                BotUttered(
                    metadata={
                        ACTIVE_FLOW_METADATA_KEY: "flow_a",
                        STEP_ID_METADATA_KEY: "utter_3",
                    }
                ),
                FlowCompleted("flow_a", "END"),
            ],
            [
                FlowPath(
                    "flow_b",
                    nodes=[
                        PathNode(step_id="utter_2", flow="flow_b"),
                    ],
                ),
                FlowPath(
                    "flow_a",
                    nodes=[
                        PathNode(step_id="utter_1", flow="flow_a"),
                        PathNode(step_id="utter_3", flow="flow_a"),
                    ],
                ),
            ],
            {},
        ),
    ],
)
def test_get_tested_flow_paths_and_commands(
    events: List[Event],
    expected_flow_paths: List[FlowPath],
    expected_tested_commands: Dict[str, Dict[str, int]],
    mock_e2e_test_runner: E2ETestRunner,
):
    test_result = TestResult(TestCase("test_case", []), pass_status=True, difference=[])

    actual_flow_paths, actual_tested_commands = (
        mock_e2e_test_runner._get_tested_flow_paths_and_commands(events, test_result)
    )

    assert actual_flow_paths == expected_flow_paths
    assert actual_tested_commands == expected_tested_commands
