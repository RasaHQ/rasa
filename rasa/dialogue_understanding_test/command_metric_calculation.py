import typing
from collections import defaultdict
from typing import Dict, List

from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding_test.command_comparison import (
    is_command_present_in_list,
)
from rasa.dialogue_understanding_test.command_metrics import CommandMetrics

if typing.TYPE_CHECKING:
    from rasa.dialogue_understanding_test.du_test_result import (
        DialogueUnderstandingTestResult,
    )


def calculate_command_metrics(
    test_results: List["DialogueUnderstandingTestResult"],
) -> Dict[str, CommandMetrics]:
    """Calculate the command metrics for the test result."""
    metrics: Dict[str, CommandMetrics] = defaultdict(
        lambda: CommandMetrics(tp=0, fp=0, fn=0, total_count=0)
    )

    for test_result in test_results:
        _increase_total_count(test_result.get_expected_commands(), metrics)

        # if the test case passed, count all commands as tp
        if test_result.passed:
            _increase_tp(test_result.get_expected_commands(), metrics)
            continue

        # in case the test case failed, we need to compare
        # the expected and actual commands for each step
        for step in test_result.test_case.iterate_over_user_steps():
            expected_commands = step.commands
            predicted_commands = step.get_predicted_commands()

            _update_metrics_true_positive_and_false_negative(
                expected_commands, predicted_commands, metrics
            )
            _update_metrics_false_positive(
                expected_commands, predicted_commands, metrics
            )

    return metrics


def _get_command_name(command: Command) -> str:
    return command.command().replace(" ", "_")


def _increase_total_count(
    commands: List[Command],
    metrics: Dict[str, CommandMetrics],
) -> None:
    for command in commands:
        metrics[_get_command_name(command)].total_count += 1


def _increase_tp(
    commands: List[Command],
    metrics: Dict[str, CommandMetrics],
) -> None:
    for command in commands:
        metrics[_get_command_name(command)].tp += 1


def _update_metrics_true_positive_and_false_negative(
    expected_commands: List[Command],
    predicted_commands: List[Command],
    metrics: Dict[str, CommandMetrics],
) -> None:
    for expected_command in expected_commands:
        command_name = _get_command_name(expected_command)
        if is_command_present_in_list(expected_command, predicted_commands):
            metrics[command_name].tp += 1
        else:
            metrics[command_name].fn += 1


def _update_metrics_false_positive(
    expected_commands: List[Command],
    predicted_commands: List[Command],
    metrics: Dict[str, CommandMetrics],
) -> None:
    for predicted_command in predicted_commands:
        if not is_command_present_in_list(predicted_command, expected_commands):
            metrics[_get_command_name(predicted_command)].fp += 1
