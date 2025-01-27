import pytest

# Assuming these imports are from your actual code
from rasa.dialogue_understanding.commands import SetSlotCommand, StartFlowCommand
from rasa.dialogue_understanding_test.command_metric_calculation import CommandMetrics
from rasa.dialogue_understanding_test.constants import ACTOR_USER
from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingOutput,
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.dialogue_understanding_test.du_test_result import (
    DialogueUnderstandingTestResult,
    DialogueUnderstandingTestSuiteResult,
    FailedTestStep,
    get_command_comparison,
)


@pytest.fixture
def sample_output() -> DialogueUnderstandingOutput:
    return DialogueUnderstandingOutput(
        prompts={
            "component1": [("system1", {"user_prompt": "user1"})],
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

    def test_get_expected_commands(
        self, sample_test_case: DialogueUnderstandingTestCase
    ):
        """Test getting expected commands from a test case."""
        all_commands = sample_test_case.get_expected_commands()
        assert len(all_commands) == 1
        assert all_commands[0].as_dict() == StartFlowCommand("bar").as_dict()


class TestDialogueUnderstandingTestSuiteResult:
    def test_from_results_with_some_tests(self):
        """Test from_results() with one failing test and one passing test."""
        # Create a passing step (expected == predicted)
        passing_step = DialogueUnderstandingTestStep(
            actor=ACTOR_USER,
            text="hello",
            commands=[SetSlotCommand(name="slot1", value="value1")],
            dialogue_understanding_output=DialogueUnderstandingOutput(
                commands={
                    "dummy_component": [SetSlotCommand(name="slot1", value="value1")]
                }
            ),
        )
        passing_test_case = DialogueUnderstandingTestCase(
            name="test_case_pass",
            steps=[passing_step],
            file="test_file_pass.yml",
        )
        passing_test_result = DialogueUnderstandingTestResult(
            test_case=passing_test_case,
            passed=True,
        )

        # Create a failing step (expected != predicted)
        failing_step = DialogueUnderstandingTestStep(
            actor=ACTOR_USER,
            text="world",
            commands=[SetSlotCommand(name="slot1", value="value1")],
            dialogue_understanding_output=DialogueUnderstandingOutput(
                commands={
                    "dummy_component": [SetSlotCommand(name="slot1", value="value2")]
                }
            ),
        )
        failing_test_case = DialogueUnderstandingTestCase(
            name="test_case_fail",
            steps=[failing_step],
            file="test_file_fail.yml",
        )
        failing_test_result = DialogueUnderstandingTestResult(
            test_case=failing_test_case,
            passed=False,
        )

        # Create command metrics for demonstration
        metrics = {"some_command": CommandMetrics(tp=2, fp=1, fn=1, total_count=5)}

        result = DialogueUnderstandingTestSuiteResult.from_results(
            failing_test_results=[failing_test_result],
            passing_test_results=[passing_test_result],
            command_metrics=metrics,
        )

        assert result.number_of_passed_tests == 1
        assert result.number_of_failed_tests == 1
        assert result.accuracy["test_cases"] == 0.5

        assert result.number_of_passed_user_utterances == 1
        assert result.number_of_failed_user_utterances == 1
        assert result.accuracy["user_utterances"] == 0.5

        # Check command metrics
        assert result.command_metrics["some_command"].total_count == 5

        # Check names of tests
        assert result.names_of_passed_tests == ["test_file_pass.yml::test_case_pass"]
        assert result.names_of_failed_tests == ["test_file_fail.yml::test_case_fail"]

        # Check failed steps
        assert len(result.failed_test_steps) == 1
        failed_step_obj = result.failed_test_steps[0]
        assert failed_step_obj.failed_user_utterance == "world"
        assert failed_step_obj.pass_status is False
        assert failed_step_obj.expected_commands[0].to_dsl() == "SetSlot(slot1, value1)"
        assert (
            failed_step_obj.predicted_commands["dummy_component"][0].to_dsl()
            == "SetSlot(slot1, value2)"
        )

    def test_create_failed_steps_from_results(self):
        """Test the static method create_failed_steps_from_results() directly."""
        fail_step_1 = DialogueUnderstandingTestStep(
            actor=ACTOR_USER,
            text="user 1",
            commands=[SetSlotCommand(name="slot1", value="value1")],
            dialogue_understanding_output=DialogueUnderstandingOutput(
                commands={"dummy": [SetSlotCommand(name="slot1", value="value2")]}
            ),
        )
        fail_test_case_1 = DialogueUnderstandingTestCase(
            name="fail_case_1",
            file="fail_file_1.yml",
            steps=[fail_step_1],
        )
        fail_result_1 = DialogueUnderstandingTestResult(
            test_case=fail_test_case_1, passed=False
        )

        fail_step_2 = DialogueUnderstandingTestStep(
            actor=ACTOR_USER,
            text="user 2",
            commands=[SetSlotCommand(name="slot1", value="value1")],
            dialogue_understanding_output=DialogueUnderstandingOutput(
                commands={"dummy": [SetSlotCommand(name="slot1", value="value2")]}
            ),
        )
        fail_test_case_2 = DialogueUnderstandingTestCase(
            name="fail_case_2",
            file="fail_file_2.yml",
            steps=[fail_step_2],
        )
        fail_result_2 = DialogueUnderstandingTestResult(
            test_case=fail_test_case_2, passed=False
        )

        failed_steps = (
            DialogueUnderstandingTestSuiteResult._create_failed_steps_from_results(
                [fail_result_1, fail_result_2]
            )
        )

        assert len(failed_steps) == 2
        step1 = failed_steps[0]
        step2 = failed_steps[1]

        assert step1.test_case_name == "fail_case_1"
        assert step1.file == "fail_file_1.yml"
        assert step1.failed_user_utterance == "user 1"
        assert step1.expected_commands[0].to_dsl() == "SetSlot(slot1, value1)"
        assert step1.predicted_commands["dummy"][0].to_dsl() == "SetSlot(slot1, value2)"

        assert step2.test_case_name == "fail_case_2"
        assert step2.file == "fail_file_2.yml"
        assert step2.failed_user_utterance == "user 2"
        assert step2.expected_commands[0].to_dsl() == "SetSlot(slot1, value1)"
        assert step2.predicted_commands["dummy"][0].to_dsl() == "SetSlot(slot1, value2)"

    def test_to_dict_output(self):
        """Test that to_dict() returns the correct structure."""
        dsr = DialogueUnderstandingTestSuiteResult()
        dsr.number_of_passed_tests = 2
        dsr.number_of_failed_tests = 1
        dsr.number_of_passed_user_utterances = 5
        dsr.number_of_failed_user_utterances = 2
        dsr.accuracy["test_cases"] = 0.66
        dsr.accuracy["user_utterances"] = 0.71

        # Add command metrics
        dsr.command_metrics = {
            "some_command": CommandMetrics(tp=2, fp=1, fn=1, total_count=5),
        }

        # Add lists of test names
        dsr.names_of_passed_tests = ["file_pass:test_name_pass"]
        dsr.names_of_failed_tests = ["file_fail:test_name_fail"]

        # Add a failed step for demonstration
        failed_step = FailedTestStep(
            file="fail_file_1.yml",
            test_case_name="fail_tc_1",
            failed_user_utterance="Hello",
            error_line=10,
            pass_status=False,
            command_generators=["SingleStepLLMCommandGenerator"],
            prompt=None,
            expected_commands=[SetSlotCommand(name="slot1", value="value1")],
            predicted_commands={
                "dummy_comp": [SetSlotCommand(name="slot1", value="value2")]
            },
            conversation_with_diff=["user: Hello", "bot: Hi!"],
        )
        dsr.failed_test_steps = [failed_step]

        # Convert to dict
        result = dsr.to_dict(output_prompt=False)

        assert result["accuracy"]["test_cases"] == 0.66
        assert result["accuracy"]["user_utterances"] == 0.71
        assert result["number_of_passed_tests"] == 2
        assert result["number_of_failed_tests"] == 1
        assert result["number_of_passed_user_utterances"] == 5
        assert result["number_of_failed_user_utterances"] == 2
        assert "command_metrics" in result
        assert result["command_metrics"]["some_command"]["total_count"] == 5
        assert result["names_of_passed_tests"] == ["file_pass:test_name_pass"]
        assert result["names_of_failed_tests"] == ["file_fail:test_name_fail"]

        # Check failed_test_steps
        assert len(result["failed_test_steps"]) == 1
        step_info = result["failed_test_steps"][0]
        assert step_info["file"] == "fail_file_1.yml"
        assert step_info["test_case"] == "fail_tc_1"
        assert step_info["failed_user_utterance"] == "Hello"
        assert step_info["error_line"] == 10
        assert step_info["pass_status"] is False
        assert step_info["expected_commands"] == ["SetSlot(slot1, value1)"]
        predicted_cmds = step_info["predicted_commands"]
        assert len(predicted_cmds) == 1
        assert predicted_cmds[0]["component"] == "dummy_comp"
        assert predicted_cmds[0]["commands"] == ["SetSlot(slot1, value2)"]
        assert "prompts" not in step_info

    def test_get_command_comparison(self):
        sample_test_step = DialogueUnderstandingTestStep(
            actor="user",
            text="Hello",
            template=None,
            line=42,
            metadata_name=None,
            commands=[StartFlowCommand("bar")],
            dialogue_understanding_output=DialogueUnderstandingOutput(
                commands={
                    "component1": [
                        SetSlotCommand("bar", "baz"),
                        SetSlotCommand("foo", "bar"),
                    ],
                    "component2": [StartFlowCommand("foo")],
                },
                prompts={
                    "component1": [("system1", {"user_prompt": "user1"})],
                },
            ),
        )

        """Test get_command_comparison() returns the formatted difference lines."""
        diff_lines = get_command_comparison(sample_test_step)
        assert any("---EXPECTED---" in line.split("|")[0] for line in diff_lines)
        assert any("---PREDICTED---" in line.split("|")[1] for line in diff_lines)
        assert any("StartFlow(bar)" in line.split("|")[0] for line in diff_lines)
        assert any("SetSlot(bar, baz)" in line.split("|")[1] for line in diff_lines)
        assert any("SetSlot(foo, bar)" in line.split("|")[1] for line in diff_lines)
        assert any("StartFlow(foo)" in line.split("|")[1] for line in diff_lines)
