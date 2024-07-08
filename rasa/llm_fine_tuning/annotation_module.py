from typing import List

from rasa.e2e_test.e2e_test_case import TestSuite
from rasa.e2e_test.e2e_test_runner import E2ETestRunner
from rasa.llm_fine_tuning.conversations import Conversation


def annotate_e2e_tests(
    e2e_test_runner: E2ETestRunner, test_suite: TestSuite, output_dir: str
) -> List[Conversation]:
    # TODO placeholder

    return []
