from contextlib import contextmanager
from typing import List, Generator

from rasa.e2e_test.e2e_test_case import TestSuite
from rasa.e2e_test.e2e_test_runner import E2ETestRunner
from rasa.llm_fine_tuning.conversation_storage import StorageContext
from rasa.llm_fine_tuning.conversations import Conversation

preparing_fine_tuning_data = False


@contextmanager
def set_preparing_fine_tuning_data() -> Generator:
    global preparing_fine_tuning_data
    preparing_fine_tuning_data = True
    try:
        yield
    finally:
        preparing_fine_tuning_data = False


def annotate_e2e_tests(
    e2e_test_runner: E2ETestRunner,
    test_suite: TestSuite,
    storage_context: StorageContext,
) -> List[Conversation]:
    with set_preparing_fine_tuning_data():
        # e2e_test_runner.run_tests()
        pass
    # TODO placeholder

    return []
