from typing import List, Dict

from rasa.dialogue_understanding_test.du_test_result import (
    DialogueUnderstandingTestResult,
)


def calculate_command_metrics(
    test_results: List[DialogueUnderstandingTestResult],
) -> Dict[str, Dict[str, float]]:
    """Calculate the metrics for the commands."""
    return {}
