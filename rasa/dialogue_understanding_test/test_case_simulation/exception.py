from typing import Any, Dict, Optional

from rasa.shared.exceptions import RasaException


class TestCaseTrackerSimulatorException(RasaException):
    def __init__(
        self,
        test_case_name: str,
        user_message: Optional[str] = None,
        failure_reason: Optional[str] = None,
        original_exception: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("An error occurred while simulating a conversation.")
        self.test_case_name = test_case_name
        self.user_message = user_message
        self.original_exception = original_exception
        self.failure_reason = failure_reason

    def __str__(self) -> str:
        s = f"{self.__class__.__name__}:"
        s += f"\nTest case: {self.test_case_name}"
        if self.user_message is not None:
            s += f"\nUser message: {self.user_message}"
        if self.failure_reason is not None:
            s += f"\nFailure reason: {self.failure_reason}"
        s += f"\nOriginal error: {self.original_exception}\n"
        return s
