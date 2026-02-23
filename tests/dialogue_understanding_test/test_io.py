from pathlib import Path

import pytest

from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingOutput,
    DialogueUnderstandingTestStep,
)
from rasa.dialogue_understanding_test.io import read_test_suite
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.utils.yaml import YamlValidationException


@pytest.fixture
def sample_output() -> DialogueUnderstandingOutput:
    return DialogueUnderstandingOutput(
        commands={
            "component1": [SetSlotCommand("bar", "baz"), SetSlotCommand("foo", "bar")],
            "component2": [StartFlowCommand("foo")],
        },
        prompts={
            "component1": [("system1", {"user_prompt": "user1"})],
        },
    )


@pytest.fixture
def sample_test_step(
    sample_output: DialogueUnderstandingOutput,
) -> DialogueUnderstandingTestStep:
    return DialogueUnderstandingTestStep(
        actor="user",
        text="Hello",
        template=None,
        line=42,
        metadata_name=None,
        commands=[StartFlowCommand("bar")],
        dialogue_understanding_output=sample_output,
    )


def test_read_test_suite(dialogue_understanding_tests_input_folder: Path) -> None:
    flows = FlowsList(underlying_flows=[Flow(id="transfer_money")])

    test_suite = read_test_suite(
        str(dialogue_understanding_tests_input_folder / "valid_test_case.yml"), flows
    )
    # Assert the TestSuite object is not None
    assert test_suite is not None

    # Assert test cases
    assert len(test_suite.test_cases) == 1
    test_case = test_suite.test_cases[0]
    assert test_case.name == "cancellation respects scope"
    assert test_case.file == str(
        dialogue_understanding_tests_input_folder / "valid_test_case.yml"
    )
    assert test_case.line == 15
    assert test_case.fixture_names == ["standard"]
    assert test_case.metadata_name == "user_info"
    assert len(test_case.steps) == 7

    # Assert each step in the test case
    step_1 = test_case.steps[0]
    assert step_1.actor == "user"
    assert step_1.text == "send money to John"
    assert step_1.line == 20
    assert step_1.commands == [
        StartFlowCommand(flow="transfer_money"),
        SetSlotCommand(name="transfer_money_recipient", value="John", extractor="LLM"),
    ]

    step_2 = test_case.steps[1]
    assert step_2.actor == "bot"
    assert step_2.template == "utter_ask_transfer_money_amount_of_money"
    assert step_2.commands == []

    step_3 = test_case.steps[2]
    assert step_3.actor == "user"
    assert step_3.text == "cancel"
    assert step_3.commands == [CancelFlowCommand()]

    step_4 = test_case.steps[3]
    assert step_4.actor == "bot"
    assert step_4.template == "utter_flow_cancelled_rasa"

    step_5 = test_case.steps[4]
    assert step_5.actor == "bot"
    assert step_5.template == "utter_can_do_something_else"

    step_6 = test_case.steps[5]
    assert step_6.actor == "user"
    assert step_6.text == "send money"
    assert step_6.commands == [StartFlowCommand(flow="transfer_money")]

    step_7 = test_case.steps[6]
    assert step_7.actor == "bot"
    assert step_7.template == "utter_ask_transfer_money_recipient"

    assert len(test_suite.fixtures_per_test) == 1
    tc_fixtures = test_suite.fixtures_per_test[0]
    assert tc_fixtures.test_case_name == "cancellation respects scope"
    assert Path(tc_fixtures.file).name == "valid_test_case.yml"
    # fixtures_per_test only includes fixtures used by this test case (standard)
    # not all file fixtures
    fixtures_by_name = {f.name: f for f in tc_fixtures.fixtures}
    assert len(fixtures_by_name) == 1
    fixture_standard = fixtures_by_name["standard"]
    assert fixture_standard.slots_set == {"membership_type": "standard"}

    # Assert metadata
    assert len(test_suite.metadata) == 2
    metadata_user_info = test_suite.metadata[0]
    assert metadata_user_info.name == "user_info"
    assert metadata_user_info.metadata == {"language": "English", "location": "Europe"}

    metadata_device_info = test_suite.metadata[1]
    assert metadata_device_info.name == "device_info"
    assert metadata_device_info.metadata == {"os": "linux"}

    # Assert stub_custom_actions is empty
    assert test_suite.stub_custom_actions == {}


@pytest.mark.parametrize(
    "file_name", ["missing_commands_test_case.yml", "missing_steps_test_case.yml"]
)
def test_read_test_suite_fails_due_to_invalid_test_case(
    dialogue_understanding_tests_input_folder: Path, file_name: str
) -> None:
    with pytest.raises(YamlValidationException):
        read_test_suite(
            str(dialogue_understanding_tests_input_folder / file_name), FlowsList([])
        )
