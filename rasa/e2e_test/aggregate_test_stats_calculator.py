from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Set

import structlog

from rasa.e2e_test.assertions import _get_all_assertion_subclasses

if TYPE_CHECKING:
    from rasa.e2e_test.assertions import Assertion
    from rasa.e2e_test.e2e_test_case import TestCase
    from rasa.e2e_test.e2e_test_result import TestResult

structlogger = structlog.get_logger()


@dataclass
class AccuracyCalculation:
    """Data class for storing the accuracy calculation for an assertion type."""

    assertion_type: str
    accuracy: float


class AggregateTestStatsCalculator:
    """Class for calculating the aggregate test statistics for assertion types."""

    def __init__(
        self,
        passed_results: List["TestResult"],
        failed_results: List["TestResult"],
        test_cases: List["TestCase"],
    ) -> None:
        self.passed_results = passed_results
        self.failed_results = failed_results
        self.test_cases = test_cases

        self.failed_assertion_set: Set["Assertion"] = set()
        self.failed_test_cases_without_assertion_failure: Set[str] = set()
        self.passed_count_mapping = {
            subclass_type: 0
            for subclass_type in _get_all_assertion_subclasses().keys()
            if subclass_type != ""
        }
        self.failed_count_mapping = {
            subclass_type: 0
            for subclass_type in _get_all_assertion_subclasses().keys()
            if subclass_type != ""
        }

    def calculate(self) -> List[AccuracyCalculation]:
        """Calculates the aggregate test statistics for assertion types."""
        self._update_failed_count_mapping()
        self._update_passed_count_mapping()

        accuracy_calculations = []

        for assertion_type in self.passed_count_mapping.keys():
            accuracy = self._calculate_accuracy(assertion_type)
            if accuracy is None:
                continue

            accuracy_calculations.append(AccuracyCalculation(assertion_type, accuracy))

        return accuracy_calculations

    def _calculate_accuracy(self, assertion_type: str) -> Optional[float]:
        """Calculates the accuracy for the given assertion type."""
        passed_count = self.passed_count_mapping[assertion_type]
        failed_count = self.failed_count_mapping[assertion_type]

        total_count = passed_count + failed_count

        if total_count == 0:
            structlogger.debug(
                "aggregate_test_stats.calculate_accuracy."
                "no_test_cases_for_assertion_type",
                assertion_type=assertion_type,
            )
            return None

        return passed_count / total_count

    def _update_passed_count_mapping(self) -> None:
        """Updates the passed count mapping based on the passed results.

        We only count the assertions of passed tests and those
        that are not in the failed assertion set.
        We also do not count the assertions that follow a failed assertion.
        """
        passed_test_case_names = [
            passed.test_case.name for passed in self.passed_results
        ]
        # We filter out test cases that failed without an assertion failure
        filtered_test_cases = [
            test_case
            for test_case in self.test_cases
            if test_case.name not in self.failed_test_cases_without_assertion_failure
        ]

        for test_case in filtered_test_cases:
            if test_case.name in passed_test_case_names:
                for step in test_case.steps:
                    if step.assertions is None:
                        continue

                    for assertion in step.assertions:
                        self.passed_count_mapping[assertion.type()] += 1
            else:
                for step in test_case.steps:
                    if step.assertions is None:
                        continue

                    for assertion in step.assertions:
                        if assertion not in self.failed_assertion_set:
                            self.passed_count_mapping[assertion.type()] += 1
                        else:
                            break

    def _update_failed_count_mapping(self) -> None:
        """Updates the failed count mapping based on the failed results."""
        for failed in self.failed_results:
            if failed.assertion_failure is None:
                structlogger.debug(
                    "aggregate_test_stats.calculate."
                    "no_assertion_failure_in_failed_result",
                    test_case=failed.test_case.name,
                )
                self.failed_test_cases_without_assertion_failure.add(
                    failed.test_case.name
                )
                continue

            self.failed_assertion_set.add(failed.assertion_failure.assertion)
            self.failed_count_mapping[failed.assertion_failure.assertion.type()] += 1
