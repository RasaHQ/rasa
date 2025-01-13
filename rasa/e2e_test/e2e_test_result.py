from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text

if TYPE_CHECKING:
    from rasa.e2e_test.assertions import AssertionFailure
    from rasa.e2e_test.e2e_test_case import TestCase
    from rasa.shared.core.flows.flow_path import FlowPath

NO_RESPONSE = "* No Bot Response *"
NO_SLOT = "* No Slot Set *"


@dataclass
class TestResult:
    """Class for storing results of every test case run."""

    test_case: "TestCase"
    pass_status: bool
    difference: List[Text]
    error_line: Optional[int] = None
    assertion_failure: Optional["AssertionFailure"] = None
    tested_paths: Optional[List["FlowPath"]] = None
    tested_commands: Optional[Dict[str, Dict[str, int]]] = None

    def as_dict(self) -> Dict[Text, Any]:
        """Returns the test result as a dictionary."""
        serialized_test_result = {
            "name": self.test_case.name,
            "pass_status": self.pass_status,
            "expected_steps": [s.as_dict() for s in self.test_case.steps],
        }

        if self.error_line is not None:
            serialized_test_result["error_path"] = (
                f"{self.test_case.file}:{self.error_line}"
            )

        if self.difference:
            serialized_test_result["difference"] = self.difference

        if self.assertion_failure is not None:
            serialized_test_result["assertion_failure"] = (
                self.assertion_failure.as_dict()
            )

        return serialized_test_result


@dataclass
class TestFailure:
    """Class for storing the test case failure."""

    test_case: "TestCase"
    error_line: Optional[int]
