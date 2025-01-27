from typing import List, Optional

import pytest

from rasa.dialogue_understanding.commands import StartFlowCommand
from rasa.dialogue_understanding_test.constants import (
    ACTOR_BOT,
    ACTOR_USER,
    PLACEHOLDER_GENERATED_ANSWER_TEMPLATE,
)
from rasa.dialogue_understanding_test.du_test_case import DialogueUnderstandingTestStep
from rasa.e2e_test.e2e_test_case import (
    ActualStepOutput,
    Fixture,
    Metadata,
    TestCase,
    TestStep,
)
from rasa.shared.core.constants import USER
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.nlu.constants import PREDICTED_COMMANDS
from scripts.convert_e2e_tests_to_du_tests import (
    DialogueUnderstandingTestCase,
    TestSuite,
    _convert_to_bot_test_steps,
    _extract_commands,
    _filter_fixtures,
    _filter_metadata,
    convert_test_case,
)


@pytest.fixture
def test_suite() -> TestSuite:
    return TestSuite(test_cases=[], fixtures=[], metadata=[], stub_custom_actions={})


@pytest.fixture
def du_test_cases() -> List[DialogueUnderstandingTestCase]:
    return [
        DialogueUnderstandingTestCase(
            name="test_case_1",
            steps=[DialogueUnderstandingTestStep(actor="user", text="hello")],
            file="test_file_1.yml",
            fixture_names=["fixture_1"],
            metadata_name="metadata_1",
        ),
        DialogueUnderstandingTestCase(
            name="test_case_2",
            steps=[DialogueUnderstandingTestStep(actor="user", text="hello")],
            file="test_file_2.yml",
            fixture_names=["fixture_2", "fixture_3"],
            metadata_name="metadata_2",
        ),
        DialogueUnderstandingTestCase(
            name="test_case_3",
            steps=[DialogueUnderstandingTestStep(actor="user", text="hello")],
            file="test_file_3.yml",
            fixture_names=[],
            metadata_name=None,
        ),
    ]


@pytest.fixture
def fixtures() -> List[Fixture]:
    return [
        Fixture(name="fixture_1", slots_set={}),
        Fixture(name="fixture_2", slots_set={}),
        Fixture(name="fixture_3", slots_set={}),
    ]


@pytest.fixture
def metadata() -> List[Metadata]:
    return [
        Metadata(name="metadata_1", metadata={}),
        Metadata(name="metadata_2", metadata={}),
        Metadata(name="metadata_3", metadata={}),
    ]


def test_convert_passing_test_case():
    dummy_test_turn = ActualStepOutput(
        actor=ACTOR_BOT,
        text="dummy test turn",
        user_uttered_events=[],
        bot_uttered_events=[],
        slot_set_events=[],
        events=[],
    )
    test_turns = ActualStepOutput(
        actor=ACTOR_USER,
        text="Hello",
        user_uttered_events=[
            UserUttered(
                "hello",
                parse_data={
                    PREDICTED_COMMANDS: {
                        "component": [StartFlowCommand("flow").as_dict()]
                    }
                },
            )
        ],
        bot_uttered_events=[],
        slot_set_events=[],
        events=[],
    )
    e2e_test_case = TestCase(
        name="test_case_1",
        steps=[TestStep(actor=ACTOR_USER, text="Hello")],
        file="test_file.yml",
        line=1,
        fixture_names=[],
        metadata_name=None,
    )
    du_test_case = convert_test_case(
        {-1: dummy_test_turn, 0: test_turns},
        e2e_test_case,
        assertions_used=False,
        test_passing=True,
    )

    assert du_test_case is not None
    assert du_test_case.name == e2e_test_case.name
    assert len(du_test_case.steps) == 1
    assert du_test_case.steps[0].actor == ACTOR_USER
    assert du_test_case.steps[0].commands[0] == StartFlowCommand("flow")


def test_convert_passing_test_case_with_no_commands():
    test_turns = ActualStepOutput(
        actor=ACTOR_USER,
        text="Hello",
        user_uttered_events=[UserUttered("hello", parse_data={})],
        bot_uttered_events=[],
        slot_set_events=[],
        events=[],
    )
    e2e_test_case = TestCase(
        name="test_case_1",
        steps=[TestStep(actor=ACTOR_USER, text="hello")],
        file="test_file.yml",
        line=1,
        fixture_names=[],
        metadata_name=None,
    )

    du_test_case = convert_test_case(
        {0: test_turns}, e2e_test_case, assertions_used=False, test_passing=True
    )

    assert du_test_case is None


def test_convert_failing_test_case_with_no_commands():
    test_turns = ActualStepOutput(
        actor=ACTOR_USER,
        text="Hello",
        user_uttered_events=[UserUttered("hello", parse_data={})],
        bot_uttered_events=[],
        slot_set_events=[],
        events=[],
    )
    e2e_test_case = TestCase(
        name="test_case_1",
        steps=[TestStep(actor=USER, text="hello")],
        file="test_file.yml",
        line=1,
        fixture_names=[],
        metadata_name=None,
    )

    du_test_case = convert_test_case(
        {0: test_turns}, e2e_test_case, assertions_used=False, test_passing=False
    )

    assert du_test_case is not None
    assert du_test_case.name == e2e_test_case.name
    assert len(du_test_case.steps) == 1
    assert du_test_case.steps[0].actor == ACTOR_USER
    assert du_test_case.steps[0].commands is None


def test_convert_passing_test_case_with_assertions():
    test_turns = ActualStepOutput(
        actor=ACTOR_USER,
        text="Hello",
        user_uttered_events=[
            UserUttered(
                "hello",
                parse_data={
                    PREDICTED_COMMANDS: {
                        "component": [StartFlowCommand("flow").as_dict()]
                    }
                },
            )
        ],
        bot_uttered_events=[BotUttered("hi")],
        slot_set_events=[],
        events=[],
    )
    e2e_test_case = TestCase(
        name="test_case_1",
        steps=[TestStep(actor=ACTOR_USER, text="hello")],
        file="test_file.yml",
        line=1,
        fixture_names=[],
        metadata_name=None,
    )

    du_test_case = convert_test_case(
        {0: test_turns}, e2e_test_case, assertions_used=True, test_passing=False
    )

    assert du_test_case is not None
    assert du_test_case.name == e2e_test_case.name
    assert len(du_test_case.steps) == 2
    assert du_test_case.steps[0].actor == ACTOR_USER
    assert du_test_case.steps[0].commands is None
    assert du_test_case.steps[1].actor == ACTOR_BOT


@pytest.mark.parametrize(
    "test_case_index, expected",
    [
        (0, ["fixture_1"]),
        (1, ["fixture_2", "fixture_3"]),
        (2, []),
        (None, ["fixture_1", "fixture_2", "fixture_3"]),
    ],
)
def test_filter_fixtures(
    test_case_index: Optional[int],
    expected: List[str],
    fixtures: List[Fixture],
    du_test_cases: List[DialogueUnderstandingTestCase],
):
    if test_case_index is not None:
        test_cases = [du_test_cases[test_case_index]]
    else:
        test_cases = du_test_cases

    print(test_cases)

    filtered = _filter_fixtures(fixtures, test_cases)

    assert len(filtered) == len(expected)
    assert {f.name for f in filtered} == set(expected)


@pytest.mark.parametrize(
    "test_case_index, expected",
    [
        (0, ["metadata_1"]),
        (1, ["metadata_2"]),
        (2, []),
        (None, ["metadata_1", "metadata_2"]),
    ],
)
def test_filter_metadata(
    test_case_index: Optional[int],
    expected: List[str],
    metadata: List[Metadata],
    du_test_cases: List[DialogueUnderstandingTestCase],
):
    if test_case_index is not None:
        test_cases = [du_test_cases[test_case_index]]
    else:
        test_cases = du_test_cases

    filtered = _filter_metadata(metadata, test_cases)

    assert len(filtered) == len(expected)
    assert {m.name for m in filtered} == set(expected)


@pytest.mark.parametrize(
    "user_uttered_events, expected_commands",
    [
        ([], None),
        ([UserUttered()], None),
        ([UserUttered(parse_data=None)], None),
        ([UserUttered(parse_data={})], None),
        ([UserUttered(parse_data={PREDICTED_COMMANDS: {}})], []),
        (
            [UserUttered(parse_data={PREDICTED_COMMANDS: {"component": []}})],
            [],
        ),
        (
            [
                UserUttered(
                    parse_data={
                        PREDICTED_COMMANDS: {
                            "component": [StartFlowCommand("flow").as_dict()]
                        }
                    }
                )
            ],
            [StartFlowCommand("flow")],
        ),
    ],
)
def test_extract_commands_no_user_uttered_events(
    user_uttered_events: List[UserUttered],
    expected_commands: Optional[List[StartFlowCommand]],
):
    turn = ActualStepOutput(
        actor=ACTOR_USER,
        text="foobar",
        user_uttered_events=user_uttered_events,
        bot_uttered_events=[BotUttered("hi")],
        slot_set_events=[],
        events=[],
    )
    assert _extract_commands(turn) == expected_commands


@pytest.mark.parametrize(
    "bot_uttered_events, expected_steps",
    [
        ([], []),
        (
            [BotUttered("hello")],
            [DialogueUnderstandingTestStep(actor=ACTOR_BOT, text="hello")],
        ),
        (
            [BotUttered(metadata={"utter_action": "utter_greet"})],
            [DialogueUnderstandingTestStep(actor=ACTOR_BOT, template="utter_greet")],
        ),
        (
            [
                BotUttered(
                    "This is a knowledge answer response.",
                    metadata={"utter_source": "EnterpriseSearchPolicy"},
                )
            ],
            [
                DialogueUnderstandingTestStep(
                    actor=ACTOR_BOT,
                    text="This is a knowledge answer response.",
                    template=PLACEHOLDER_GENERATED_ANSWER_TEMPLATE,
                )
            ],
        ),
    ],
)
def test_convert_to_bot_test_steps(
    bot_uttered_events: List[BotUttered],
    expected_steps: Optional[List[DialogueUnderstandingTestStep]],
):
    turn = ActualStepOutput(
        actor=ACTOR_USER,
        text="foobar",
        user_uttered_events=[UserUttered("foobar", parse_data={})],
        bot_uttered_events=bot_uttered_events,
        slot_set_events=[],
        events=[],
    )

    actual_steps = _convert_to_bot_test_steps(turn)

    assert actual_steps == expected_steps
