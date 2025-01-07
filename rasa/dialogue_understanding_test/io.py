from typing import List, Dict

from rasa.dialogue_understanding_test.du_test_result import (
    DialogueUnderstandingTestResult,
)
from rasa.e2e_test.e2e_test_case import TestSuite


def read_test_suite(test_case_path: str) -> TestSuite:
    """Read the test cases from the given test case path.

    Args:
        test_case_path: Path to the test cases.

    Returns:
        TestSuite: Test suite containing the dialogue understanding test cases.
    """
    return TestSuite([], [], [], {})


def write_test_results_to_file(
    failed_tests: List[DialogueUnderstandingTestResult],
    passed_tests: List[DialogueUnderstandingTestResult],
    command_metrics: Dict[str, Dict[str, float]],
    output_file: str,
    output_prompt: bool,
) -> None:
    """Write the test results to the given output file.

    Args:
        failed_tests: Failed test cases.
        passed_tests: Passed test cases.
        command_metrics: Metrics for the commands.
        output_file: Path to the output file.
        output_prompt: Whether to log the prompt or not.
    """
    pass


def print_test_results(
    failed_tests: List[DialogueUnderstandingTestResult],
    passed_tests: List[DialogueUnderstandingTestResult],
    command_metrics: Dict[str, Dict[str, float]],
    output_prompt: bool,
) -> None:
    """Print the test results to console.

    Args:
        failed_tests: Failed test cases.
        passed_tests: Passed test cases.
        command_metrics: Metrics for the commands.
        output_prompt: Whether to log the prompt or not.
    """
    pass
