from typing import Dict, List, Tuple

from rasa.dialogue_understanding_test.du_test_case import (
    DialogueUnderstandingTestCase,
    DialogueUnderstandingTestStep,
)
from rasa.e2e_test.e2e_test_case import Metadata
from rasa.e2e_test.e2e_test_runner import E2ETestRunner


def filter_metadata(
    test_case: DialogueUnderstandingTestCase,
    user_step: DialogueUnderstandingTestStep,
    metadata: List[Metadata],
    sender_id: str,
) -> Dict[str, str]:
    """Filter metadata for a test case and a step."""
    # test case metadata
    test_case_metadata = E2ETestRunner.filter_metadata_for_input(
        test_case.metadata_name, metadata
    )
    test_case_metadata_dict = test_case_metadata.metadata if test_case_metadata else {}

    # step metadata
    step_metadata = E2ETestRunner.filter_metadata_for_input(
        user_step.metadata_name, metadata
    )
    step_metadata_dict = step_metadata.metadata if step_metadata else {}

    # merge metadata
    return E2ETestRunner.merge_metadata(
        sender_id, user_step.text, test_case_metadata_dict, step_metadata_dict
    )


def get_command_comparison(step: DialogueUnderstandingTestStep) -> List[str]:
    expected_commands = (
        [command.to_dsl() for command in step.commands] if step.commands else []
    )
    predicted_commands = [command.to_dsl() for command in step.get_predicted_commands()]

    expected_commands.insert(0, "---EXPECTED---")
    predicted_commands.insert(0, "---PREDICTED---")

    max_line_length = max(len(line) for line in expected_commands)
    expected_commands, predicted_commands = make_lists_equal(
        expected_commands, predicted_commands
    )

    command_comparison = []
    for i, (expected_line, actual_line) in enumerate(
        zip(expected_commands, predicted_commands)
    ):
        expected_line += " " * (max_line_length - len(expected_line))
        if i == 0:
            # make the first line red:
            expected_line = "[red3]" + expected_line + "[/red3]"
            actual_line = "[red3]" + actual_line + "[/red3]"
        command_comparison.append(f"{expected_line} | {actual_line}")

    return command_comparison


def make_lists_equal(list1: List[str], list2: List[str]) -> Tuple[List[str], List[str]]:
    if len(list1) < len(list2):
        list1.extend([""] * (len(list2) - len(list1)))
        return list1, list2

    list2.extend([""] * (len(list1) - len(list2)))
    return list1, list2
