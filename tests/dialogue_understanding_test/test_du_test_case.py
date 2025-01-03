import pydantic_core
import pytest

# Assuming these imports are from your actual code
from rasa.dialogue_understanding.commands import StartFlowCommand
from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingOutput,
    DialogueUnderstandingTestStep,
    DialogueUnderstandingTestCase,
)


@pytest.fixture
def sample_output() -> DialogueUnderstandingOutput:
    return DialogueUnderstandingOutput(
        prompts={
            "component1": ("system1", {"user_prompt": "user1"}),
        },
        commands={
            "component1": [StartFlowCommand("bar")],
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


class TestDialogueUnderstandingOutput:
    def test_valid_creation(self, sample_output: DialogueUnderstandingOutput):
        """Test that a valid output can be created."""
        assert isinstance(sample_output, DialogueUnderstandingOutput)
        assert len(sample_output.prompts) == 1
        assert len(sample_output.commands) == 1

    def test_get_component_data_existing(
        self, sample_output: DialogueUnderstandingOutput
    ):
        """Test getting data for an existing component."""
        prompts, commands = sample_output.get_component_data("component1")
        assert prompts == ("system1", {"user_prompt": "user1"})
        assert len(commands) == 1
        assert commands[0].as_dict() == StartFlowCommand("bar").as_dict()

    def test_get_component_data_nonexistent(
        self, sample_output: DialogueUnderstandingOutput
    ):
        """Test getting data for a non-existent component."""
        prompts, commands = sample_output.get_component_data("nonexistent")
        assert prompts is None
        assert commands == []


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
            "commands": [StartFlowCommand("bar").as_dict()],
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
