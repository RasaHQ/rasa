from typing import Any, Dict, List, Union
from collections import OrderedDict

import pytest

from rasa.e2e_test.constants import KEY_SLOT_NOT_SET, KEY_SLOT_SET
from rasa.shared.core.events import (
    BotUttered,
    Event,
    SlotSet,
    UserUttered,
)
from rasa.e2e_test.e2e_test_case import (
    ActualStepOutput,
    Fixture,
    Metadata,
    TestCase,
    TestStep,
)
from rasa.shared.exceptions import RasaException


def test_create_test_fixture_from_dict() -> None:
    """Test creating a test fixture from a dictionary."""
    result = Fixture.from_dict(
        {"some_fixture_name": [{"slot_a": 1}, {"slot_b": "some_value"}]}
    )
    assert result.name == "some_fixture_name"
    assert result.slots_set == {"slot_a": 1, "slot_b": "some_value"}


def test_create_test_fixture_from_dict_invalid() -> None:
    """Test creating an invalid fixture.

    The value type associated with the fixture name is a dict
    instead of a list of dicts.
    """
    with pytest.raises(AttributeError):
        Fixture.from_dict({"some_fixture_name": {"slot_a": 1, "slot_b": "some_value"}})


def test_create_test_metadata_from_dict() -> None:
    """Test creating a test metadata from a dictionary."""
    result = Metadata.from_dict({"some_metadata_name": {"os": "linux", "name": "Tom"}})
    assert result.name == "some_metadata_name"
    assert result.metadata == {"os": "linux", "name": "Tom"}


@pytest.mark.parametrize(
    "input, metadata_name",
    [({"user": "Hi!"}, ""), ({"user": "Hi!", "metadata": "user_info"}, "user_info")],
)
def test_create_test_step_user_from_dict(input: Dict, metadata_name: str) -> None:
    """Test creating a test step from a dictionary."""
    result = TestStep.from_dict(input)
    assert result.text == "Hi!"
    assert result.actor == "user"
    assert result.metadata_name == metadata_name


def test_create_test_step_bot_from_dict() -> None:
    """Test creating a test step from a dictionary."""
    result = TestStep.from_dict({"bot": "Hi!"})
    assert result.text == "Hi!"
    assert result.actor == "bot"


@pytest.mark.parametrize(
    "input",
    [
        {"utter": "utter_greet"},
        {"slot_was_set": {"slot_a": 1}},
        {"slot_was_not_set": {"slot_a": 1}},
    ],
)
def test_create_test_step_from_dict_text_is_none(input: Dict) -> None:
    """Test creating a test step from a dictionary.

    When "user" or "bot" keys are omitted, `text` property should be None.
    """
    result = TestStep.from_dict(input)
    assert result.text is None


def test_create_test_step_utter_from_dict() -> None:
    """Test creating a test step from a dictionary."""
    result = TestStep.from_dict({"utter": "utter_greet"})
    assert result.text is None
    assert result.template == "utter_greet"
    assert result.actor == "bot"


def test_create_test_step_from_dict_invalid() -> None:
    """Test creating a test step from a dictionary."""
    with pytest.raises(RasaException):
        TestStep.from_dict({"invalid": "Hi!"})


def test_create_test_case_from_dict() -> None:
    result = TestCase.from_dict(
        {
            "test_case": "book_trip",
            "steps": [
                {"user": "Hi!"},
                {"bot": "Hey! How are you?"},
            ],
            "metadata": "some_metadata",
        }
    )
    assert result.name == "book_trip"

    assert result.steps[0].actor == "user"
    assert result.steps[0].text == "Hi!"

    assert result.steps[1].actor == "bot"
    assert result.steps[1].text == "Hey! How are you?"


@pytest.mark.parametrize(
    "test_input",
    [
        {"user": "Hi!"},
        {"user": "Hi!", "metadata": "user_info"},
        {"bot": "Hi!"},
        {"utter": "utter_greet"},
        {"slot_was_set": {"slot_a": 1}},
        {"slot_was_set": "slot_a"},
        {"slot_was_not_set": {"slot_a": 1}},
        {"slot_was_not_set": "slot_a"},
        {
            "user": "How can I pay for my credit card?",
            "assertions": [{"flow_started": "pay_credit_card_bill"}],
        },
        {
            "user": "How can I pay for my credit card?",
            "assertions": [
                {
                    "pattern_clarification_contains": [
                        "transfer_money",
                        "apply_credit_card",
                        "pay_credit_card_bill",
                    ]
                }
            ],
            "assertion_order_enabled": True,
        },
    ],
)
def test_test_step_from_dict_validation_pass(test_input: Dict[str, Any]) -> None:
    step = TestStep.from_dict(test_input)
    assert step.as_dict() == test_input


@pytest.mark.parametrize(
    "test_input, expected_error",
    [
        ({}, "Test step is missing either the"),
        (
            {"slot_was_set": {"slot_a": 1}, "slot_was_not_set": {"slot_a": 1}},
            "Test step has both slot_was_set and slot_was_not_set keys",
        ),
        (
            {"slot_was_set": "slot_a", "slot_was_not_set": "slot_b"},
            "Test step has both slot_was_set and slot_was_not_set keys",
        ),
        ({"random": 123}, "Test step is missing either the"),
        (
            {"bot": "Hello!", "assertions": [{"flow_started": "welcome_flow"}]},
            "Test step with assertions must only be used with the 'user' key:",
        ),
        (
            {"bot": "Hello!", "assertion_order_enabled": True},
            "Test step with 'assertion_order_enabled' key must only "
            "be used with the 'user' key:",
        ),
        (
            {"user": "Hello!", "assertion_order_enabled": True},
            "You must specify the 'assertions' key in the user "
            "test step where you are using 'assertion_order_enabled' key:",
        ),
    ],
)
def test_test_step_from_dict_validation_exception(
    test_input: Dict, expected_error: str
) -> None:
    with pytest.raises(RasaException, match=expected_error):
        TestStep.from_dict(test_input)


@pytest.mark.parametrize(
    "test_step, event",
    [
        (TestStep.from_dict({"bot": "Hi!"}), BotUttered("Hi!")),
        (
            TestStep.from_dict({"utter": "utter_greet"}),
            BotUttered("Hello!", metadata={"utter_action": "utter_greet"}),
        ),
        (TestStep.from_dict({"slot_was_set": {"slot_a": 1}}), SlotSet("slot_a", 1)),
        (
            TestStep.from_dict({"slot_was_set": {"slot_a": None}}),
            SlotSet("slot_a", None),
        ),
        (TestStep.from_dict({"slot_was_set": "slot_a"}), SlotSet("slot_a", 1)),
        (TestStep.from_dict({"slot_was_set": "slot_a"}), SlotSet("slot_a", None)),
        (TestStep.from_dict({"slot_was_not_set": "slot_a"}), SlotSet("slot_a", 1)),
        (TestStep.from_dict({"slot_was_not_set": "slot_a"}), SlotSet("slot_a", None)),
        (TestStep.from_dict({"slot_was_not_set": {"slot_a": 1}}), SlotSet("slot_a", 1)),
        (
            TestStep.from_dict({"slot_was_not_set": {"slot_a": None}}),
            SlotSet("slot_a", None),
        ),
    ],
)
def test_matches_event_true(test_step: TestStep, event: Event) -> None:
    assert test_step.matches_event(event)


@pytest.mark.parametrize(
    "test_step, event",
    [
        (TestStep.from_dict({"bot": "Hi!"}), BotUttered("Hello!")),
        (
            TestStep.from_dict({"utter": "utter_greet"}),
            BotUttered("Hello!", metadata={"utter_action": "utter_welcome"}),
        ),
        (TestStep.from_dict({"slot_was_set": {"slot_a": 1}}), SlotSet("slot_b", 1)),
        (TestStep.from_dict({"slot_was_set": {"slot_a": 1}}), SlotSet("slot_a", 2)),
        (TestStep.from_dict({"slot_was_set": "slot_a"}), SlotSet("slot_b", 1)),
        (TestStep.from_dict({"slot_was_not_set": "slot_a"}), SlotSet("slot_b", 1)),
        (TestStep.from_dict({"slot_was_not_set": {"slot_a": 1}}), SlotSet("slot_b", 1)),
        (TestStep.from_dict({"slot_was_not_set": {"slot_a": 1}}), SlotSet("slot_a", 2)),
    ],
)
def test_matches_event_false(test_step: TestStep, event: Event) -> None:
    assert not test_step.matches_event(event)


@pytest.mark.parametrize(
    "test_step, value",
    [
        (TestStep.from_dict({"slot_was_set": {"slot_a": 1}}), "slot_a"),
        (TestStep.from_dict({"slot_was_set": "slot_a"}), "slot_a"),
    ],
)
def test_get_slot_name(test_step: TestStep, value: str) -> None:
    assert test_step.get_slot_name() == value


@pytest.mark.parametrize(
    "test_step, value",
    [
        (TestStep.from_dict({"slot_was_set": {"slot_a": 1}}), 1),
        (TestStep.from_dict({"slot_was_set": "slot_a"}), None),
    ],
)
def test_get_slot_value(test_step: TestStep, value: Union[str, None]) -> None:
    assert test_step.get_slot_value() == value


@pytest.mark.parametrize(
    "actual_step_output, remove, after",
    [
        (
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Hi!"}),
                [UserUttered("Hi!"), BotUttered("Hi!")],
            ),
            [BotUttered("Hi!")],
            [UserUttered("Hi!")],
        ),
        (
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    BotUttered("Hi!"),
                    BotUttered("How are you doing!"),
                ],
            ),
            [BotUttered("Hi!")],
            [UserUttered("Hi!"), BotUttered("How are you doing!")],
        ),
        (
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    BotUttered("Hi!"),
                    BotUttered("How are you doing!"),
                ],
            ),
            [BotUttered("Hi!"), BotUttered("How are you doing!")],
            [UserUttered("Hi!")],
        ),
        (
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    BotUttered("Hi!"),
                    BotUttered("How are you doing!"),
                ],
            ),
            [BotUttered("Hello!"), BotUttered("Welcome to xyz!")],
            [UserUttered("Hi!"), BotUttered("Hi!"), BotUttered("How are you doing!")],
        ),
    ],
)
def test_remove_bot_uttered_event(
    actual_step_output: ActualStepOutput, remove: List[BotUttered], after: List[Event]
) -> None:
    for event in remove:
        actual_step_output.remove_bot_uttered_event(event)

    assert actual_step_output.events == after
    for event in remove:
        assert event not in actual_step_output.bot_uttered_events


@pytest.mark.parametrize(
    "actual_step_output, remove, after",
    [
        (
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Hi!"}),
                [UserUttered("Hi!"), BotUttered("Hi!")],
            ),
            [UserUttered("Hi!")],
            [BotUttered("Hi!")],
        ),
        (
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    BotUttered("Hi!"),
                    BotUttered("How are you doing!"),
                ],
            ),
            [UserUttered("Hi!")],
            [
                BotUttered("Hi!"),
                BotUttered("How are you doing!"),
            ],
        ),
        (
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    BotUttered("Hi!"),
                    BotUttered("How are you doing!"),
                ],
            ),
            [UserUttered("Hello!")],
            [
                UserUttered("Hi!"),
                BotUttered("Hi!"),
                BotUttered("How are you doing!"),
            ],
        ),  # there is only one user utterance in the ActualStepOutput
    ],  # since it is created from a TestStep
)
def test_remove_user_uttered_event(
    actual_step_output: ActualStepOutput, remove: List[BotUttered], after: List[Event]
) -> None:
    for event in remove:
        actual_step_output.remove_user_uttered_event(event)

    assert actual_step_output.events == after
    for event in remove:
        assert event not in actual_step_output.user_uttered_events


@pytest.mark.parametrize(
    "actual_step_output, remove, after",
    [
        (
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Hi!"}),
                [UserUttered("Hi!"), BotUttered("Hi!"), SlotSet("slot_a", 1)],
            ),
            [SlotSet("slot_a", 1)],
            [UserUttered("Hi!"), BotUttered("Hi!")],
        ),
        (
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    BotUttered("Hi!"),
                    BotUttered("How are you doing!"),
                    SlotSet("slot_a", 1),
                    SlotSet("slot_b", 2),
                ],
            ),
            [SlotSet("slot_a", 1), SlotSet("slot_b", 2)],
            [
                UserUttered("Hi!"),
                BotUttered("Hi!"),
                BotUttered("How are you doing!"),
            ],
        ),
        (
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"bot": "Hi!"}),
                [
                    UserUttered("Hi!"),
                    BotUttered("Hi!"),
                    BotUttered("How are you doing!"),
                    SlotSet("slot_a", 1),
                    SlotSet("slot_b", 2),
                ],
            ),
            [SlotSet("slot_a", 3), SlotSet("slot_b", 4)],
            [
                UserUttered("Hi!"),
                BotUttered("Hi!"),
                BotUttered("How are you doing!"),
                SlotSet("slot_a", 1),
                SlotSet("slot_b", 2),
            ],
        ),
    ],
)
def test_remove_slot_set_event(
    actual_step_output: ActualStepOutput, remove: List[BotUttered], after: List[Event]
) -> None:
    for event in remove:
        actual_step_output.remove_slot_set_event(event)

    assert actual_step_output.events == after
    for event in remove:
        assert event not in actual_step_output.slot_set_events


@pytest.mark.parametrize(
    "test_step, expected_dict",
    [
        (TestStep.from_dict({"bot": "Hi!"}), {"bot": "Hi!"}),
        (TestStep.from_dict({"user": "Hello"}), {"user": "Hello"}),
        (
            TestStep.from_dict({KEY_SLOT_NOT_SET: [{"slot_name": "slot_value"}]}),
            {KEY_SLOT_NOT_SET: [{"slot_name": "slot_value"}]},
        ),
        (
            TestStep.from_dict(
                {KEY_SLOT_NOT_SET: OrderedDict([("slot_name", "slot_value")])}
            ),
            {KEY_SLOT_NOT_SET: [{"slot_name": "slot_value"}]},
        ),
        (
            TestStep.from_dict({KEY_SLOT_NOT_SET: ["slot_name", "another_slot"]}),
            {KEY_SLOT_NOT_SET: ["slot_name", "another_slot"]},
        ),
        (
            TestStep.from_dict({KEY_SLOT_NOT_SET: "slot_name"}),
            {KEY_SLOT_NOT_SET: ["slot_name"]},
        ),
        (
            TestStep.from_dict({KEY_SLOT_SET: [{"slot_name": "slot_value"}]}),
            {KEY_SLOT_SET: [{"slot_name": "slot_value"}]},
        ),
        (
            TestStep.from_dict(
                {KEY_SLOT_SET: OrderedDict([("another_slot", "another_value")])}
            ),
            {KEY_SLOT_SET: [{"another_slot": "another_value"}]},
        ),
        (
            TestStep.from_dict({KEY_SLOT_SET: ["slot_name", "another_slot"]}),
            {KEY_SLOT_SET: ["slot_name", "another_slot"]},
        ),
        (
            TestStep.from_dict({KEY_SLOT_SET: "slot_name"}),
            {KEY_SLOT_SET: ["slot_name"]},
        ),
    ],
)
def test_test_step_as_dict_in_yaml_format(
    test_step: TestStep, expected_dict: Dict[str, Any]
):
    assert expected_dict == test_step.as_dict_yaml_format()
