from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text

from rasa.e2e_test.e2e_test_case import TestCase

NO_RESPONSE = "* No Bot Response *"
NO_SLOT = "* No Slot Set *"


@dataclass
class TestResult:
    """Class for storing results of every test case run."""

    test_case: TestCase
    pass_status: bool
    difference: List[Text]
    error_line: Optional[int] = None

    def as_dict(self) -> Dict[Text, Any]:
        """Returns the test result as a dictionary."""
        return {
            "name": self.test_case.name,
            "pass_status": self.pass_status,
            "expected_steps": [s.as_dict() for s in self.test_case.steps],
            "difference": self.difference,
        }


@dataclass
class TestFailure:
    """Class for storing the test case failure."""

    test_case: TestCase
    error_line: Optional[int]
