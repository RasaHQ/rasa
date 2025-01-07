import argparse
from typing import List

from rasa.dialogue_understanding_test.du_test_case import DialogueUnderstandingTestCase


def validate_cli_arguments(args: argparse.Namespace) -> None:
    """Validate the CLI arguments for the dialogue understanding test.

    Args:
        args: Commandline arguments.
    """
    pass


def validate_test_cases(test_cases: List[DialogueUnderstandingTestCase]) -> None:
    """Validate the dialogue understanding test cases.

    Args:
        test_cases: Test cases to validate.
    """
    pass
