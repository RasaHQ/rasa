from unittest.mock import Mock

import pytest

from rasa.dialogue_understanding.commands import SetSlotCommand, StartFlowCommand
from rasa.dialogue_understanding_test.command_comparison import (
    is_command_present_in_list,
)
from rasa.dialogue_understanding_test.command_metric_calculation import (
    CommandMetrics,
    calculate_command_metrics,
)
from rasa.dialogue_understanding_test.du_test_result import (
    DialogueUnderstandingTestResult,
)


class TestCommandMetrics:
    def test_get_precision_zero_predictions(self):
        metrics = CommandMetrics(tp=0, fp=0, fn=1, total_count=1)
        assert metrics.get_precision() == 0.0

    def test_get_precision_perfect(self):
        metrics = CommandMetrics(tp=10, fp=0, fn=0, total_count=10)
        assert metrics.get_precision() == 1.0

    def test_get_precision_mixed(self):
        metrics = CommandMetrics(tp=8, fp=2, fn=1, total_count=9)
        assert metrics.get_precision() == 0.8

    # Edge case: Very large numbers
    def test_get_precision_large_numbers(self):
        metrics = CommandMetrics(
            tp=1000000, fp=1000000, fn=1000000, total_count=2000000
        )
        assert pytest.approx(metrics.get_precision(), 0.01) == 0.5

    # Edge case: Very small numbers
    def test_get_precision_small_numbers(self):
        metrics = CommandMetrics(tp=1, fp=999999, fn=1, total_count=2)
        assert pytest.approx(metrics.get_precision(), 0.00001) == 0.000001

    def test_get_recall_zero_actual(self):
        metrics = CommandMetrics(tp=0, fp=1, fn=0, total_count=0)
        assert metrics.get_recall() == 0.0

    def test_get_recall_perfect(self):
        metrics = CommandMetrics(tp=10, fp=0, fn=0, total_count=10)
        assert metrics.get_recall() == 1.0

    def test_get_recall_mixed(self):
        metrics = CommandMetrics(tp=8, fp=1, fn=2, total_count=10)
        assert metrics.get_recall() == 0.8

    # Edge case: All false negatives
    def test_get_recall_all_false_negatives(self):
        metrics = CommandMetrics(tp=0, fp=0, fn=100, total_count=100)
        assert metrics.get_recall() == 0.0

    # Edge case: Single true positive with many false negatives
    def test_get_recall_sparse_true_positives(self):
        metrics = CommandMetrics(tp=1, fp=0, fn=999, total_count=1000)
        assert pytest.approx(metrics.get_recall(), 0.001) == 0.001

    def test_get_f1_score_zero(self):
        metrics = CommandMetrics(tp=0, fp=0, fn=0, total_count=0)
        assert metrics.get_f1_score() == 0.0

    def test_get_f1_score_perfect(self):
        metrics = CommandMetrics(tp=10, fp=0, fn=0, total_count=10)
        assert metrics.get_f1_score() == 1.0

    def test_get_f1_score_mixed(self):
        metrics = CommandMetrics(tp=8, fp=2, fn=2, total_count=10)
        assert pytest.approx(metrics.get_f1_score(), 0.01) == 0.8

    # Edge case: Perfect precision but poor recall
    def test_get_f1_score_perfect_precision_poor_recall(self):
        metrics = CommandMetrics(tp=1, fp=0, fn=99, total_count=100)
        # F1 = 2 * (1 * 0.01) / (1 + 0.01) ≈ 0.0198
        assert pytest.approx(metrics.get_f1_score(), 0.001) == 0.0198

    # Edge case: Perfect recall but poor precision
    def test_get_f1_score_perfect_recall_poor_precision(self):
        metrics = CommandMetrics(tp=1, fp=99, fn=0, total_count=1)
        # F1 = 2 * (0.01 * 1) / (0.01 + 1) ≈ 0.0198
        assert pytest.approx(metrics.get_f1_score(), 0.001) == 0.0198

    def test_as_dict(self):
        metrics = CommandMetrics(tp=8, fp=2, fn=2, total_count=10)
        result = metrics.as_dict()
        assert result["tp"] == 8
        assert result["fp"] == 2
        assert result["fn"] == 2
        assert pytest.approx(result["precision"], 0.01) == 0.8
        assert pytest.approx(result["recall"], 0.01) == 0.8
        assert pytest.approx(result["f1_score"], 0.01) == 0.8
        assert result["total_count"] == 10


class TestCalculateCommandMetrics:
    def test_calculate_metrics_all_passed(self):
        # Create test data
        cmd1 = StartFlowCommand("flow1")
        cmd2 = StartFlowCommand("flow2")

        test_result = Mock(spec=DialogueUnderstandingTestResult)
        test_result.passed = True
        test_result.get_expected_commands.return_value = [cmd1, cmd2]

        results = calculate_command_metrics([test_result])

        command_name = StartFlowCommand.command()

        assert command_name in results
        assert results[command_name].tp == 2
        assert results[command_name].fp == 0
        assert results[command_name].fn == 0
        assert results[command_name].total_count == 2

    def test_calculate_metrics_failed_case(self):
        # Create commands
        expected_cmd = StartFlowCommand("flow1")
        predicted_cmd_1 = StartFlowCommand("flow2")
        predicted_cmd_2 = SetSlotCommand("value", "name")

        # Mock step
        step = Mock()
        step.commands = [expected_cmd]
        step.get_predicted_commands.return_value = [predicted_cmd_1, predicted_cmd_2]

        # Mock test case
        test_case = Mock()
        test_case.iterate_over_user_steps.return_value = [step]

        # Mock test result
        test_result = Mock(spec=DialogueUnderstandingTestResult)
        test_result.passed = False
        test_result.test_case = test_case
        test_result.get_expected_commands.return_value = [expected_cmd]

        # Mock command comparison
        is_command_present_in_list.return_value = False

        results = calculate_command_metrics([test_result])

        # start flow should have fn=1 (missed prediction) and
        # fp=1 (false prediction)
        command_name = StartFlowCommand.command()
        assert results[command_name].tp == 0
        assert results[command_name].fp == 1
        assert results[command_name].fn == 1
        assert results[command_name].total_count == 1
        # set slot should have fp=1 (false prediction)
        command_name = SetSlotCommand.command()
        assert results[command_name].tp == 0
        assert results[command_name].fp == 1
        assert results[command_name].fn == 0
        assert results[command_name].total_count == 0

    def test_empty_test_results(self):
        results = calculate_command_metrics([])
        assert len(results) == 0

    def test_multiple_test_results_combined(self):
        # First test result - passed
        cmd1 = StartFlowCommand("flow1")
        test_result1 = Mock(spec=DialogueUnderstandingTestResult)
        test_result1.passed = True
        test_result1.get_expected_commands.return_value = [cmd1]

        # Second test result - failed
        expected_cmd = StartFlowCommand("flow2")
        predicted_cmd = StartFlowCommand("flow3")

        step = Mock()
        step.commands = [expected_cmd]
        step.get_predicted_commands.return_value = [predicted_cmd]

        test_case = Mock()
        test_case.iterate_over_user_steps.return_value = [step]

        test_result2 = Mock(spec=DialogueUnderstandingTestResult)
        test_result2.passed = False
        test_result2.test_case = test_case
        test_result2.get_expected_commands.return_value = [expected_cmd]

        is_command_present_in_list.return_value = False

        results = calculate_command_metrics([test_result1, test_result2])

        # start flow should have tp=1 from first test
        # start flow should have fp=1 and fn=1 from second test
        command_name = StartFlowCommand.command()
        assert results[command_name].tp == 1
        assert results[command_name].fp == 1
        assert results[command_name].fn == 1
        assert results[command_name].total_count == 2
