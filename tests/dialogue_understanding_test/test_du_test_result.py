import pytest

# Assuming these imports are from your actual code
from rasa.dialogue_understanding.commands import StartFlowCommand
from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingOutput,
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.dialogue_understanding_test.du_test_result import (
    DialogueUnderstandingTestResult,
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


@pytest.fixture
def sample_test_case(
    sample_test_step: DialogueUnderstandingTestStep,
) -> DialogueUnderstandingTestCase:
    return DialogueUnderstandingTestCase(
        name="test", file="test.yml", line=1, steps=[sample_test_step]
    )


class TestDialogueUnderstandingTestCase:
    def test_valid_creation(self, sample_test_case: DialogueUnderstandingTestCase):
        """Test that a valid test case can be created."""
        test_result = DialogueUnderstandingTestResult(
            test_case=sample_test_case,
            passed=True,
        )

        assert isinstance(test_result, DialogueUnderstandingTestResult)
        assert test_result.passed is True
