from typing import List

import pytest

from rasa.e2e_test.aggregate_test_stats_calculator import (
    AccuracyCalculation,
    AggregateTestStatsCalculator,
)
from rasa.e2e_test.assertions import (
    _get_all_assertion_subclasses,
    PatternClarificationContainsAssertion,
)
from rasa.e2e_test.e2e_test_case import TestStep, TestCase
from rasa.e2e_test.e2e_test_result import TestResult


@pytest.fixture
def aggregate_test_stats_calculator(
    test_cases: List["TestCase"],
    passed_assertion_results: List["TestResult"],
    failed_assertion_results: List["TestResult"],
) -> AggregateTestStatsCalculator:
    return AggregateTestStatsCalculator(
        passed_results=passed_assertion_results,
        failed_results=failed_assertion_results,
        test_cases=test_cases,
    )


@pytest.fixture
def calculator_with_pattern_clarification_assertion() -> AggregateTestStatsCalculator:
    return AggregateTestStatsCalculator(
        passed_results=[],
        failed_results=[],
        test_cases=[
            TestCase(
                "case_1",
                [
                    TestStep(
                        "actor",
                        assertions=[
                            PatternClarificationContainsAssertion(
                                flow_names={"add a card", "add a contact"}, line=12
                            ),
                        ],
                    )
                ],
            )
        ],
    )


def test_aggregate_stats_calculator_init(
    aggregate_test_stats_calculator: AggregateTestStatsCalculator,
) -> None:
    assert aggregate_test_stats_calculator.passed_results
    assert aggregate_test_stats_calculator.failed_results
    assert aggregate_test_stats_calculator.test_cases
    assert aggregate_test_stats_calculator.failed_assertion_set == set()
    assert aggregate_test_stats_calculator.passed_count_mapping == {
        subclass_type: 0
        for subclass_type in _get_all_assertion_subclasses().keys()
        if subclass_type != ""
    }
    assert aggregate_test_stats_calculator.failed_count_mapping == {
        subclass_type: 0
        for subclass_type in _get_all_assertion_subclasses().keys()
        if subclass_type != ""
    }


def test_aggregate_stats_calculator_update_failed_count_mapping(
    aggregate_test_stats_calculator: AggregateTestStatsCalculator,
) -> None:
    aggregate_test_stats_calculator._update_failed_count_mapping()

    expected_failed_count_mapping = {
        "flow_started": 0,
        "flow_completed": 0,
        "flow_cancelled": 0,
        "action_executed": 1,
        "bot_uttered": 0,
        "slot_was_set": 0,
        "slot_was_not_set": 0,
        "pattern_clarification_contains": 0,
        "generative_response_is_grounded": 0,
        "generative_response_is_relevant": 0,
    }

    assert (
        aggregate_test_stats_calculator.failed_count_mapping
        == expected_failed_count_mapping
    )


def test_aggregate_stats_calculator_update_passed_count_mapping(
    aggregate_test_stats_calculator: AggregateTestStatsCalculator,
) -> None:
    # we have to run this method first to update the failed count mapping
    # and the failed assertion set
    aggregate_test_stats_calculator._update_failed_count_mapping()
    aggregate_test_stats_calculator._update_passed_count_mapping()

    expected_passed_count_mapping = {
        "flow_started": 3,
        "flow_completed": 1,
        "flow_cancelled": 1,
        "action_executed": 1,
        "bot_uttered": 6,
        "slot_was_set": 4,
        "slot_was_not_set": 1,
        "pattern_clarification_contains": 1,
        "generative_response_is_grounded": 0,
        "generative_response_is_relevant": 0,
    }

    assert (
        aggregate_test_stats_calculator.passed_count_mapping
        == expected_passed_count_mapping
    )


def test_aggregate_stats_calculator_calculate(
    aggregate_test_stats_calculator: AggregateTestStatsCalculator,
) -> None:
    # the fixtures do not include the generative response assertions
    # so we expect the accuracy calculations to not include these assertions
    expected_accuracy_calculations = [
        AccuracyCalculation("flow_started", accuracy=1.0),
        AccuracyCalculation("flow_completed", accuracy=1.0),
        AccuracyCalculation("flow_cancelled", accuracy=1.0),
        AccuracyCalculation("action_executed", accuracy=0.5),
        AccuracyCalculation("bot_uttered", accuracy=1.0),
        AccuracyCalculation("slot_was_set", accuracy=1.0),
        AccuracyCalculation("slot_was_not_set", accuracy=1.0),
        AccuracyCalculation("pattern_clarification_contains", accuracy=1.0),
    ]

    actual_accuracy_calculations = aggregate_test_stats_calculator.calculate()

    assert len(actual_accuracy_calculations) == len(expected_accuracy_calculations)

    for expected_accuracy_calculation in expected_accuracy_calculations:
        assert expected_accuracy_calculation in actual_accuracy_calculations


def test_aggregate_stats_pattern_clarification_contains_assertion(
    calculator_with_pattern_clarification_assertion: AggregateTestStatsCalculator,
) -> None:
    try:
        calculator_with_pattern_clarification_assertion.calculate()
    except TypeError:
        pytest.fail("Unexpected TypeError")
