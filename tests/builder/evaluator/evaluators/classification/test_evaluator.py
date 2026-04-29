"""Tests for ClassificationEvaluator — thorough coverage."""

from types import SimpleNamespace

import pytest

from rasa.builder.copilot.models import ResponseCategory
from rasa.builder.evaluator.evaluators.classification.evaluator import (
    ClassificationEvaluator,
)
from rasa.builder.evaluator.evaluators.classification.models import (
    ClassificationResult,
    MetricsSummary,
    OverallClassificationMetrics,
    PerClassMetrics,
)
from rasa.builder.evaluator.tasks.base import ClassifierTaskResult

COPILOT = ResponseCategory.COPILOT
ERROR = ResponseCategory.ERROR_FALLBACK


class TestExtractResults:
    def test_happy_path(self, make_item_result):
        item = make_item_result(
            output=ClassifierTaskResult(predicted_category=COPILOT),
            expected_output={"response_category": "copilot"},
            item_input={"message": "hello"},
        )
        evaluator = ClassificationEvaluator()
        results, skip_count = evaluator.extract_results([item])

        assert len(results) == 1
        assert skip_count == 0
        assert results[0].prediction == COPILOT
        assert results[0].expected == COPILOT
        assert results[0].input_text == "hello"

    def test_multiple_valid_items(self, make_item_result):
        items = [
            make_item_result(
                output=ClassifierTaskResult(predicted_category=COPILOT),
                expected_output={"response_category": "copilot"},
                item_id=f"item-{i}",
            )
            for i in range(3)
        ]
        evaluator = ClassificationEvaluator()
        results, skip_count = evaluator.extract_results(items)

        assert len(results) == 3
        assert skip_count == 0

    def test_skip_none_output(self, make_item_result):
        item = make_item_result(
            output=None,
            expected_output={"response_category": "copilot"},
        )
        evaluator = ClassificationEvaluator()
        results, skip_count = evaluator.extract_results([item])

        assert len(results) == 0
        assert skip_count == 1

    def test_skip_non_task_result_output(self, make_item_result):
        item = make_item_result(
            output="plain string",
            expected_output={"response_category": "copilot"},
        )
        evaluator = ClassificationEvaluator()
        results, skip_count = evaluator.extract_results([item])

        assert len(results) == 0
        assert skip_count == 1

    def test_skip_none_predicted_category(self, make_item_result):
        item = make_item_result(
            output=ClassifierTaskResult(predicted_category=None),
            expected_output={"response_category": "copilot"},
        )
        evaluator = ClassificationEvaluator()
        results, skip_count = evaluator.extract_results([item])

        assert len(results) == 0
        assert skip_count == 1

    def test_skip_none_expected(self):
        """When item is None, expected becomes None → skipped."""
        item_result = SimpleNamespace(
            output=ClassifierTaskResult(predicted_category=COPILOT), item=None
        )
        evaluator = ClassificationEvaluator()
        results, skip_count = evaluator.extract_results([item_result])

        assert len(results) == 0
        assert skip_count == 1

    def test_skip_expected_not_dict(self, make_item_result):
        item = make_item_result(
            output=ClassifierTaskResult(predicted_category=COPILOT),
            expected_output="not a dict",
        )
        evaluator = ClassificationEvaluator()
        results, skip_count = evaluator.extract_results([item])

        assert len(results) == 0
        assert skip_count == 1

    def test_skip_invalid_response_category(self, make_item_result):
        item = make_item_result(
            output=ClassifierTaskResult(predicted_category=COPILOT),
            expected_output={"response_category": "TOTALLY_INVALID"},
        )
        evaluator = ClassificationEvaluator()
        results, skip_count = evaluator.extract_results([item])

        assert len(results) == 0
        assert skip_count == 1

    def test_input_text_from_dict(self, make_item_result):
        item = make_item_result(
            output=ClassifierTaskResult(predicted_category=COPILOT),
            expected_output={"response_category": "copilot"},
            item_input={"message": "test message"},
        )
        evaluator = ClassificationEvaluator()
        results, _ = evaluator.extract_results([item])

        assert results[0].input_text == "test message"

    def test_input_text_none_fallback(self, make_item_result):
        item = make_item_result(
            output=ClassifierTaskResult(predicted_category=COPILOT),
            expected_output={"response_category": "copilot"},
            item_input=None,
        )
        evaluator = ClassificationEvaluator()
        results, _ = evaluator.extract_results([item])

        assert results[0].input_text is None

    def test_mixed_valid_and_invalid(self, make_item_result):
        """Skipped items are logged; valid items still extracted."""
        items = [
            make_item_result(
                output=ClassifierTaskResult(predicted_category=COPILOT),
                expected_output={"response_category": "copilot"},
            ),
            make_item_result(output=None, expected_output=None),
            make_item_result(
                output=ClassifierTaskResult(predicted_category=ERROR),
                expected_output={"response_category": "error_fallback"},
            ),
        ]
        evaluator = ClassificationEvaluator()
        results, skip_count = evaluator.extract_results(items)

        assert len(results) == 2
        assert skip_count == 1


class TestEvaluate:
    def test_returns_metrics_summary(self):
        results = [
            ClassificationResult(prediction=COPILOT, expected=COPILOT),
            ClassificationResult(prediction=ERROR, expected=ERROR),
        ]
        evaluator = ClassificationEvaluator()
        summary = evaluator.evaluate(results)

        assert isinstance(summary, MetricsSummary)
        assert summary.overall.accuracy == pytest.approx(1.0)


class TestToEvaluations:
    def test_structure_and_count(self):
        summary = MetricsSummary(
            per_class={
                COPILOT: PerClassMetrics(
                    precision=1.0,
                    recall=0.5,
                    f1=0.667,
                    support=2,
                    true_positives=1,
                    false_positives=0,
                    false_negatives=1,
                ),
            },
            overall=OverallClassificationMetrics(
                accuracy=0.75,
                micro_precision=0.75,
                macro_precision=0.75,
                weighted_avg_precision=0.75,
                micro_recall=0.75,
                macro_recall=0.75,
                weighted_avg_recall=0.75,
                micro_f1=0.75,
                macro_f1=0.75,
                weighted_avg_f1=0.75,
                support=4,
                true_positives=3,
                false_positives=1,
                false_negatives=1,
            ),
        )
        evaluator = ClassificationEvaluator()
        evaluations = evaluator.to_evaluations(summary, skip_count=2)

        # 10 overall + 4 per class (1 class) + 1 skipped_items
        assert len(evaluations) == 15

        names = [e.name for e in evaluations]
        assert "accuracy" in names
        assert "micro_f1" in names
        assert "copilot_precision" in names
        assert "copilot_recall" in names
        assert "copilot_f1" in names
        assert "copilot_support" in names
        assert "skipped_items" in names

        skipped = next(e for e in evaluations if e.name == "skipped_items")
        assert skipped.value == 2


class TestBuildArtifacts:
    def test_returns_misclassifications_csv(self):
        evaluator = ClassificationEvaluator()
        evaluator.results = [
            ClassificationResult(
                prediction=COPILOT, expected=ERROR, input_text="wrong"
            ),
            ClassificationResult(
                prediction=COPILOT, expected=COPILOT, input_text="right"
            ),
        ]

        artifacts = evaluator.build_artifacts(timestamp="20260408_120000")

        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert artifact.filename == "20260408_120000_misclassifications.csv"
        assert artifact.fieldnames == ["input_text", "predicted", "expected"]
        assert artifact.rows == [
            {
                "input_text": "wrong",
                "predicted": "copilot",
                "expected": "error_fallback",
            }
        ]

    def test_no_misclassifications_returns_empty(self):
        evaluator = ClassificationEvaluator()
        evaluator.results = [
            ClassificationResult(prediction=COPILOT, expected=COPILOT),
        ]

        artifacts = evaluator.build_artifacts(timestamp="20260408_120000")

        assert artifacts == []

    def test_non_list_results_returns_empty(self):
        evaluator = ClassificationEvaluator()
        evaluator.results = None

        artifacts = evaluator.build_artifacts(timestamp="20260408_120000")

        assert artifacts == []

    def test_none_input_text_becomes_empty_string(self):
        evaluator = ClassificationEvaluator()
        evaluator.results = [
            ClassificationResult(prediction=COPILOT, expected=ERROR, input_text=None)
        ]

        artifacts = evaluator.build_artifacts(timestamp="20260408_120000")

        assert artifacts[0].rows[0]["input_text"] == ""
