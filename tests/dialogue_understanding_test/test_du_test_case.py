from typing import List

import pydantic_core
import pytest

from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy
from rasa.dialogue_understanding.commands import SetSlotCommand, StartFlowCommand
from rasa.dialogue_understanding_test.constants import (
    ACTOR_BOT,
    ACTOR_USER,
    KEY_BOT_INPUT,
    KEY_BOT_UTTERED,
    KEY_USER_INPUT,
)
from rasa.dialogue_understanding_test.du_test_case import (
    KEY_CHOICES,
    KEY_COMPLETION_TOKENS,
    KEY_PROMPT_TOKENS,
    DialogueUnderstandingOutput,
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.nlu.constants import (
    KEY_COMPONENT_NAME,
    KEY_LATENCY,
    KEY_LLM_RESPONSE_METADATA,
    KEY_PROMPT_NAME,
    KEY_SYSTEM_PROMPT,
    KEY_USER_PROMPT,
)
from tests.utilities import flows_from_str


@pytest.fixture
def sample_output() -> DialogueUnderstandingOutput:
    return DialogueUnderstandingOutput(
        latency=0.1234,
        commands={
            "component1": [SetSlotCommand("bar", "baz"), SetSlotCommand("foo", "bar")],
            "component2": [StartFlowCommand("foo")],
        },
        prompts=[
            {
                KEY_COMPONENT_NAME: "component1",
                KEY_USER_PROMPT: "prompt_content",
                KEY_PROMPT_NAME: "prompt_name",
                KEY_LLM_RESPONSE_METADATA: {
                    KEY_LATENCY: 1.23,
                    "usage": {
                        KEY_PROMPT_TOKENS: 1234,
                        KEY_COMPLETION_TOKENS: 4,
                    },
                },
            },
            {
                KEY_COMPONENT_NAME: "component2",
                KEY_USER_PROMPT: "prompt_content",
                KEY_PROMPT_NAME: "other_prompt_name",
                KEY_LLM_RESPONSE_METADATA: {
                    KEY_LATENCY: 1.55,
                    "usage": {
                        KEY_PROMPT_TOKENS: 1543,
                        KEY_COMPLETION_TOKENS: 6,
                    },
                },
            },
        ],
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


@pytest.fixture
def sample_correct_user_step(
    sample_output: DialogueUnderstandingOutput,
) -> DialogueUnderstandingTestStep:
    commands = [
        SetSlotCommand("bar", "baz"),
        SetSlotCommand("foo", "bar"),
        StartFlowCommand("foo"),
    ]
    return DialogueUnderstandingTestStep(
        actor="user",
        text="Hi there!",
        template=None,
        line=42,
        metadata_name=None,
        commands=commands,
        dialogue_understanding_output=sample_output,
    )


@pytest.fixture
def sample_correct_bot_step() -> DialogueUnderstandingTestStep:
    return DialogueUnderstandingTestStep(
        actor="bot",
        text="Hi there!",
        template=None,
        line=42,
        metadata_name=None,
        commands=None,
        dialogue_understanding_output=None,
    )


@pytest.fixture
def sample_flow_list() -> FlowsList:
    return flows_from_str(
        """
            flows:
              foo_flow:
                description: This is a test flow.
                steps:
                - id: start_step
                  collect: foo
                - id: second_step
                  collect: bar

              bar_flow:
                description: Another test flow.
                steps:
                - id: ask_abc
                  collect: abc
                - set_slots:
                  - def: ghi
            """
    )


class TestDialogueUnderstandingOutput:
    def test_valid_creation(self, sample_output: DialogueUnderstandingOutput):
        """Test that a valid output can be created."""
        assert isinstance(sample_output, DialogueUnderstandingOutput)
        assert len(sample_output.prompts) == 2
        assert len(sample_output.commands) == 2

    def test_get_predicted_commands(self, sample_output: DialogueUnderstandingOutput):
        """Test getting predicted commands."""
        commands = sample_output.get_predicted_commands()
        assert len(commands) == 3
        assert commands[0].as_dict() == SetSlotCommand("bar", "baz").as_dict()

    def test_get_predicted_commands_single_component(self):
        """Test get_predicted_commands returns all commands across all components."""
        output = DialogueUnderstandingOutput(
            commands={
                "component1": [
                    SetSlotCommand("slot1", "value1"),
                    SetSlotCommand("slot2", "value2"),
                ]
            }
        )
        predicted = output.get_predicted_commands()
        assert len(predicted) == 2
        assert isinstance(predicted[0], SetSlotCommand)
        assert predicted[0].name == "slot1" and predicted[0].value == "value1"

    def test_get_predicted_commands_multiple_components(self):
        """Test get_predicted_commands returns commands
        from multiple components combined.
        """
        output = DialogueUnderstandingOutput(
            commands={
                "component1": [SetSlotCommand("slot1", "value1")],
                "component2": [
                    StartFlowCommand("flow_1"),
                    SetSlotCommand("slot2", "value2"),
                ],
            }
        )
        predicted = output.get_predicted_commands()
        # Should be 3 commands total from both components.
        assert len(predicted) == 3
        assert any(isinstance(cmd, StartFlowCommand) for cmd in predicted)
        assert len([cmd for cmd in predicted if isinstance(cmd, SetSlotCommand)]) == 2

    def test_get_component_names_of_commands(self):
        output = DialogueUnderstandingOutput(
            commands={
                "componentA": [SetSlotCommand("slotA", "valA")],
                "componentB": [],
            }
        )
        component_names = (
            output.get_component_names_that_predicted_commands_or_have_llm_response()
        )
        assert sorted(component_names) == ["componentA"]

    def test_get_component_names_without_commands_but_with_llm_response(self):
        output = DialogueUnderstandingOutput(
            commands={
                "ComponentA": [SetSlotCommand("slotA", "valA")],
                "ComponentB": [],
            },
            prompts=[
                {
                    KEY_COMPONENT_NAME: "ComponentB",
                    KEY_LLM_RESPONSE_METADATA: {KEY_CHOICES: ["test LLM response"]},
                },
                {
                    KEY_COMPONENT_NAME: EnterpriseSearchPolicy.__name__,
                    KEY_LLM_RESPONSE_METADATA: {KEY_CHOICES: ["test LLM response"]},
                },
            ],
        )

        result = (
            output.get_component_names_that_predicted_commands_or_have_llm_response()
        )

        assert set(result) == {"ComponentA", "ComponentB"}

    def test_get_component_name_to_prompt_info(self):
        output = DialogueUnderstandingOutput(
            commands={
                "componentA": [SetSlotCommand("slotA", "valA")],
            },
            prompts=[
                {
                    KEY_COMPONENT_NAME: "componentA",
                    KEY_PROMPT_NAME: "promptA",
                    KEY_USER_PROMPT: "User prompt content A",
                    KEY_SYSTEM_PROMPT: "System prompt content A",
                    KEY_LLM_RESPONSE_METADATA: {
                        KEY_LATENCY: 1.23,
                        "usage": {
                            KEY_PROMPT_TOKENS: 1234,
                            KEY_COMPLETION_TOKENS: 4,
                        },
                    },
                },
                {
                    KEY_COMPONENT_NAME: "componentA",
                    KEY_PROMPT_NAME: "promptB",
                    KEY_USER_PROMPT: "User prompt content B",
                    KEY_LLM_RESPONSE_METADATA: {
                        "usage": {
                            KEY_PROMPT_TOKENS: 7834,
                            KEY_COMPLETION_TOKENS: 34,
                        }
                    },
                },
            ],
        )
        result = output.get_component_name_to_prompt_info()
        assert list(result.keys()) == ["componentA"]
        assert result["componentA"][0] == {
            KEY_PROMPT_NAME: "promptA",
            KEY_USER_PROMPT: "User prompt content A",
            KEY_SYSTEM_PROMPT: "System prompt content A",
            KEY_LATENCY: 1.23,
            KEY_COMPLETION_TOKENS: 4,
            KEY_PROMPT_TOKENS: 1234,
        }
        assert result["componentA"][1] == {
            KEY_PROMPT_NAME: "promptB",
            KEY_USER_PROMPT: "User prompt content B",
            KEY_PROMPT_TOKENS: 7834,
            KEY_COMPLETION_TOKENS: 34,
        }

    def test_get_component_name_to_prompt_info_when_no_commands_were_predicted(self):
        output = DialogueUnderstandingOutput(
            commands={},
            prompts=[
                {
                    KEY_COMPONENT_NAME: "componentA",
                    KEY_PROMPT_NAME: "promptA",
                    KEY_USER_PROMPT: "User prompt content A",
                    KEY_SYSTEM_PROMPT: "System prompt content A",
                    KEY_LLM_RESPONSE_METADATA: {
                        KEY_LATENCY: 1.23,
                        "usage": {
                            KEY_PROMPT_TOKENS: 1234,
                            KEY_COMPLETION_TOKENS: 4,
                        },
                        KEY_CHOICES: ["test LLM response"],
                    },
                },
                {
                    KEY_COMPONENT_NAME: "componentA",
                    KEY_PROMPT_NAME: "promptB",
                    KEY_USER_PROMPT: "User prompt content B",
                    KEY_LLM_RESPONSE_METADATA: {
                        "usage": {
                            KEY_PROMPT_TOKENS: 7834,
                            KEY_COMPLETION_TOKENS: 34,
                        },
                        KEY_CHOICES: ["test LLM response 2"],
                    },
                },
            ],
        )
        result = output.get_component_name_to_prompt_info()
        assert list(result.keys()) == ["componentA"]
        assert result["componentA"][0] == {
            KEY_PROMPT_NAME: "promptA",
            KEY_USER_PROMPT: "User prompt content A",
            KEY_SYSTEM_PROMPT: "System prompt content A",
            KEY_LATENCY: 1.23,
            KEY_COMPLETION_TOKENS: 4,
            KEY_PROMPT_TOKENS: 1234,
            KEY_CHOICES: "test LLM response",
        }
        assert result["componentA"][1] == {
            KEY_PROMPT_NAME: "promptB",
            KEY_USER_PROMPT: "User prompt content B",
            KEY_PROMPT_TOKENS: 7834,
            KEY_COMPLETION_TOKENS: 34,
            KEY_CHOICES: "test LLM response 2",
        }


class TestDialogueUnderstandingTestStep:
    def test_valid_creation(self, sample_test_step: DialogueUnderstandingTestStep):
        """Test that a valid test step can be created."""
        assert isinstance(sample_test_step, DialogueUnderstandingTestStep)
        assert sample_test_step.actor == "user"
        assert sample_test_step.line == 42

    def test_optional_fields(self):
        """Test creation with minimal required fields."""
        step = DialogueUnderstandingTestStep(
            actor="user",
        )
        assert step.text is None
        assert step.template is None
        assert step.line is None
        assert step.commands is None
        assert step.metadata_name is None
        assert step.dialogue_understanding_output is None

    def test_as_dict(self, sample_test_step: DialogueUnderstandingTestStep):
        """Test converting a test step to a dictionary."""
        assert sample_test_step.as_dict() == {
            "user": "Hello",
            "commands": [StartFlowCommand("bar").to_dsl()],
        }

    def test_get_predicted_commands(
        self, sample_test_step: DialogueUnderstandingTestStep
    ):
        """Test getting predicted commands from a test step."""
        commands = sample_test_step.get_predicted_commands()
        assert len(commands) == 3
        assert commands[0] == SetSlotCommand("bar", "baz")
        assert commands[1] == SetSlotCommand("foo", "bar")
        assert commands[2] == StartFlowCommand("foo")

    def test_has_passed_true(self, sample_test_step: DialogueUnderstandingTestStep):
        sample_test_step.commands = [StartFlowCommand("foo")]
        sample_test_step.dialogue_understanding_output = DialogueUnderstandingOutput(
            commands={"component": [StartFlowCommand("foo")]}
        )
        assert sample_test_step.has_passed() is True

    def test_has_passed_false(self, sample_test_step: DialogueUnderstandingTestStep):
        sample_test_step.commands = [StartFlowCommand("bar")]
        sample_test_step.dialogue_understanding_output = DialogueUnderstandingOutput(
            commands={"component": [StartFlowCommand("foo")]}
        )
        assert sample_test_step.has_passed() is False

    def test_has_passed_no_expected_commands(self):
        """Test has_passed returns True if there are
        no expected commands and predicted is empty.
        """
        step = DialogueUnderstandingTestStep(
            actor=ACTOR_USER, text="Hello", commands=None
        )
        # If there's no output, predicted commands is also empty
        step.dialogue_understanding_output = None
        assert step.has_passed() is True

    def test_has_passed_mismatch_in_commands(self):
        """Verify has_passed is False if the expected and predicted commands differ."""
        step = DialogueUnderstandingTestStep(
            actor=ACTOR_USER,
            commands=[StartFlowCommand("foo")],
            dialogue_understanding_output=DialogueUnderstandingOutput(
                commands={"component1": [StartFlowCommand("bar")]}
            ),
        )
        assert step.has_passed() is False

    def test_has_passed_match_in_commands(self):
        """Verify has_passed is True if the expected and predicted commands match."""
        step = DialogueUnderstandingTestStep(
            actor=ACTOR_USER,
            commands=[SetSlotCommand("slot_name", "slot_value")],
            dialogue_understanding_output=DialogueUnderstandingOutput(
                commands={"component1": [SetSlotCommand("slot_name", "slot_value")]}
            ),
        )
        assert step.has_passed() is True

    @pytest.mark.parametrize(
        "actor,text,template,expected_str",
        [
            (
                ACTOR_BOT,
                "Hello from bot!",
                None,
                f"{KEY_BOT_INPUT}: Hello from bot!",
            ),
            (
                ACTOR_BOT,
                None,
                "utter_greeting",
                f"{KEY_BOT_UTTERED}: utter_greeting",
            ),
            (
                ACTOR_USER,
                "User input text",
                None,
                f"{KEY_USER_INPUT}: User input text",
            ),
        ],
    )
    def test_to_str_bot_and_user(
        self, actor: str, text: str, template: str, expected_str: str
    ):
        """Test the to_str output for both bot and user steps."""
        step = DialogueUnderstandingTestStep(
            actor=actor,
            text=text,
            template=template,
        )
        assert step.to_str() == expected_str

    def test_get_latencies(self, sample_test_step: DialogueUnderstandingTestStep):
        """Test getting latencies from a test step."""
        assert sample_test_step.get_latencies() == {
            "component1": [1.23],
            "component2": [1.55],
        }

    def test_get_completion_tokens(
        self, sample_test_step: DialogueUnderstandingTestStep
    ):
        """Test getting latencies from a test step."""
        assert sample_test_step.get_completion_tokens() == {
            "component1": [4],
            "component2": [6],
        }

    def test_get_prompt_tokens(self, sample_test_step: DialogueUnderstandingTestStep):
        """Test getting latencies from a test step."""
        assert sample_test_step.get_prompt_tokens() == {
            "component1": [1234],
            "component2": [1543],
        }


class TestDialogueUnderstandingTestCase:
    def test_valid_creation(self, sample_test_step: DialogueUnderstandingTestStep):
        """Test that a valid test case can be created."""
        test_case = DialogueUnderstandingTestCase(
            name="test", file="test.yml", line=1, steps=[sample_test_step]
        )
        assert isinstance(test_case, DialogueUnderstandingTestCase)
        assert len(test_case.steps) == 1

    def test_empty_steps_raises_error(self):
        """Test that creating a test case with empty steps raises an error."""
        with pytest.raises(pydantic_core._pydantic_core.ValidationError):
            DialogueUnderstandingTestCase(
                name="test", file="test.yml", line=1, steps=[]
            )

    def test_multiple_steps(self, sample_test_step: DialogueUnderstandingTestStep):
        """Test creation with multiple steps."""
        test_case = DialogueUnderstandingTestCase(
            name="test",
            file="test.yml",
            line=1,
            steps=[sample_test_step, sample_test_step],
        )
        assert len(test_case.steps) == 2

    def test_full_name(self, sample_test_step: DialogueUnderstandingTestStep):
        """Test creation with multiple steps."""
        test_case = DialogueUnderstandingTestCase(
            name="test",
            file="test.yml",
            line=1,
            steps=[sample_test_step, sample_test_step],
        )
        assert test_case.full_name() == "test.yml::test"

    def test_as_dict(self, sample_test_step: DialogueUnderstandingTestStep):
        """Test converting a test case to a dictionary."""
        test_case = DialogueUnderstandingTestCase(
            name="test", file="test.yml", line=1, steps=[sample_test_step]
        )
        assert test_case.as_dict() == {
            "test_case": "test",
            "steps": [sample_test_step.as_dict()],
        }

    def test_get_expected_commands(
        self, sample_test_step: DialogueUnderstandingTestStep
    ):
        """Test getting expected commands from a test case."""
        test_case = DialogueUnderstandingTestCase(
            name="test", file="test.yml", line=1, steps=[sample_test_step]
        )
        commands = test_case.get_expected_commands()
        assert len(commands) == 1
        assert commands[0].as_dict() == StartFlowCommand("bar").as_dict()

    def test_iterate_over_user_steps(
        self, sample_test_step: DialogueUnderstandingTestStep
    ):
        """Test iterating over user steps."""
        test_case = DialogueUnderstandingTestCase(
            name="test",
            file="test.yml",
            line=1,
            steps=[
                sample_test_step,
                DialogueUnderstandingTestStep(actor="bot", text="Hello"),
                sample_test_step,
            ],
        )
        assert list(test_case.iterate_over_user_steps()) == [
            sample_test_step,
            sample_test_step,
        ]

    @pytest.mark.parametrize(
        "steps, from_index, expected_user_step, expected_bot_steps",
        [
            (
                [
                    DialogueUnderstandingTestStep(actor=ACTOR_USER, text="User step 1"),
                    DialogueUnderstandingTestStep(actor=ACTOR_BOT, text="Bot step 1"),
                    DialogueUnderstandingTestStep(actor=ACTOR_BOT, text="Bot step 2"),
                    DialogueUnderstandingTestStep(actor=ACTOR_USER, text="User step 2"),
                ],
                0,
                DialogueUnderstandingTestStep(actor=ACTOR_USER, text="User step 1"),
                [
                    DialogueUnderstandingTestStep(actor=ACTOR_BOT, text="Bot step 1"),
                    DialogueUnderstandingTestStep(actor=ACTOR_BOT, text="Bot step 2"),
                ],
            ),
            (
                [
                    DialogueUnderstandingTestStep(actor=ACTOR_BOT, text="Bot step 1"),
                    DialogueUnderstandingTestStep(actor=ACTOR_USER, text="User step 1"),
                    DialogueUnderstandingTestStep(actor=ACTOR_BOT, text="Bot step 2"),
                    DialogueUnderstandingTestStep(actor=ACTOR_USER, text="User step 2"),
                ],
                1,
                DialogueUnderstandingTestStep(actor=ACTOR_USER, text="User step 1"),
                [DialogueUnderstandingTestStep(actor=ACTOR_BOT, text="Bot step 2")],
            ),
            (
                [
                    DialogueUnderstandingTestStep(actor=ACTOR_BOT, text="Bot step 1"),
                    DialogueUnderstandingTestStep(actor=ACTOR_USER, text="User step 1"),
                ],
                1,
                DialogueUnderstandingTestStep(actor=ACTOR_USER, text="User step 1"),
                [],
            ),
            (
                [
                    DialogueUnderstandingTestStep(actor=ACTOR_USER, text="User step 1"),
                    DialogueUnderstandingTestStep(actor=ACTOR_BOT, text="Bot step 1"),
                ],
                0,
                DialogueUnderstandingTestStep(actor=ACTOR_USER, text="User step 1"),
                [DialogueUnderstandingTestStep(actor=ACTOR_BOT, text="Bot step 1")],
            ),
        ],
    )
    def test_get_next_user_and_bot_steps(
        self,
        steps: List[DialogueUnderstandingTestStep],
        from_index: int,
        expected_user_step: DialogueUnderstandingTestStep,
        expected_bot_steps: List[DialogueUnderstandingTestStep],
    ):
        test_case = DialogueUnderstandingTestCase(name="test_case", steps=steps)
        user_step, bot_steps = test_case.get_next_user_and_bot_steps(from_index)
        assert user_step == expected_user_step
        assert bot_steps == expected_bot_steps

    def test_from_dict(self):
        """Test creating a test case from a dictionary."""
        test_case = DialogueUnderstandingTestCase.from_dict(
            {
                "test_case": "test",
                "steps": [{"user": "Hello", "commands": ["StartFlow(bar)"]}],
            },
            FlowsList(underlying_flows=[Flow(id="bar")]),
        )
        assert isinstance(test_case, DialogueUnderstandingTestCase)
        assert test_case.name == "test"
        assert len(test_case.steps) == 1
        assert test_case.steps[0].text == "Hello"
        assert test_case.steps[0].actor == "user"
        assert test_case.steps[0].line is None
        assert test_case.steps[0].commands == [StartFlowCommand("bar")]

    def test_to_readable_conversation(
        self,
        sample_correct_user_step: DialogueUnderstandingTestStep,
        sample_test_step: DialogueUnderstandingTestStep,
    ):
        """Test that to_readable_conversation returns expected lines for each step."""
        test_case = DialogueUnderstandingTestCase(
            name="test_print_case", steps=[sample_correct_user_step, sample_test_step]
        )

        lines = test_case.to_readable_conversation()
        assert lines == ["user: Hi there!", "user: Hello"]

        truncated_lines = test_case.to_readable_conversation(until_step=1)
        assert truncated_lines == ["user: Hi there!"]

    def test_failed_steps(
        self,
        sample_correct_user_step: DialogueUnderstandingTestStep,
        sample_test_step: DialogueUnderstandingTestStep,
    ):
        """Test that failed_steps() returns only the failing steps."""
        test_case = DialogueUnderstandingTestCase(
            name="test_failed_steps",
            steps=[sample_correct_user_step, sample_test_step],
        )
        assert test_case.failed_user_steps() == [sample_test_step]

    def test_raise_error_on_parsing_start_flow_command_with_nonexisting_flow_arg(
        self, sample_flow_list: FlowsList
    ):
        with pytest.raises(ValueError) as exc_info:
            DialogueUnderstandingTestStep.from_dict(
                step={"user": "hello", "commands": ["StartFlow(non_existing_flow)"]},
                flows=sample_flow_list,
            )
        assert (
            "Failed to parse command 'StartFlow(non_existing_flow)': command parser "
            "returned None" in str(exc_info.value)
        )

    def test_do_not_raise_error_on_parsing_valid_start_flow_command(
        self, sample_flow_list: FlowsList
    ):
        # assert no exception is raised
        try:
            DialogueUnderstandingTestStep.from_dict(
                step={"user": "hello", "commands": ["StartFlow(foo_flow)"]},
                flows=sample_flow_list,
            )
        except ValueError:
            pytest.fail("ValueError raised unexpectedly.")

    def test_do_raise_error_on_parsing_valid_clarify_command(
        self, sample_flow_list: FlowsList
    ):
        # assert no exception is raised
        try:
            DialogueUnderstandingTestStep.from_dict(
                step={"user": "hello", "commands": ["Clarify(foo_flow, bar_flow)"]},
                flows=sample_flow_list,
            )
        except ValueError:
            pytest.fail("ValueError raised unexpectedly.")
