from typing import List

import pytest

from rasa.builder.copilot.models import ResponseCategory
from rasa.builder.evaluator.response_classification.evaluator import (
    ResponseClassificationEvaluator,
)
from rasa.builder.evaluator.response_classification.models import (
    ClassificationResult,
    MetricsSummary,
    OverallClassificationMetrics,
    PerClassMetrics,
)


class TestResponseClassificationEvaluator:
    """Test suite for ResponseClassificationEvaluator."""

    def test_classification_metrics_calculations(self):
        """Test comprehensive metrics calculations for different scenarios."""
        # Given
        results: List[ClassificationResult] = [
            # Correct predictions
            ClassificationResult(
                prediction=ResponseCategory.COPILOT, expected=ResponseCategory.COPILOT
            ),
            ClassificationResult(
                prediction=ResponseCategory.COPILOT, expected=ResponseCategory.COPILOT
            ),
            ClassificationResult(
                prediction=ResponseCategory.COPILOT, expected=ResponseCategory.COPILOT
            ),
            ClassificationResult(
                prediction=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                expected=ResponseCategory.OUT_OF_SCOPE_DETECTION,
            ),
            ClassificationResult(
                prediction=ResponseCategory.ROLEPLAY_DETECTION,
                expected=ResponseCategory.ROLEPLAY_DETECTION,
            ),
            # False predictions
            ClassificationResult(
                prediction=ResponseCategory.COPILOT,
                expected=ResponseCategory.OUT_OF_SCOPE_DETECTION,
            ),
            ClassificationResult(
                prediction=ResponseCategory.COPILOT,
                expected=ResponseCategory.ROLEPLAY_DETECTION,
            ),
            ClassificationResult(
                prediction=ResponseCategory.ROLEPLAY_DETECTION,
                expected=ResponseCategory.COPILOT,
            ),
            ClassificationResult(
                prediction=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                expected=ResponseCategory.COPILOT,
            ),
            ClassificationResult(
                prediction=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                expected=ResponseCategory.ROLEPLAY_DETECTION,
            ),
        ]

        expected_metrics_summary: MetricsSummary = MetricsSummary(
            overall=OverallClassificationMetrics(
                # Precision
                micro_precision=0.5,  # tp/(tp+fp) = 0.5
                macro_precision=0.2867,  # precision per class / num classes = 1.4333 / 5 = 0.2867  # noqa: E501
                weighted_avg_precision=0.5167,  # (0.6*5 + 0.3333*2 + 0.5*3) / 10 = 0.5167  # noqa: E501
                # Recall
                micro_recall=0.5,  # tp/(tp+fn) = 0.5
                macro_recall=0.2867,  # recall per class / num classes = 1.4333 / 5 = 0.2867  # noqa: E501
                weighted_avg_recall=0.5,  # (0.6*5 + 0.5*2 + 0.3333*3) / 10 = 0.5
                # F1
                micro_f1=0.5,  # 5/10 = 0.5
                macro_f1=0.28,  # f1 per class / num classes = 1.4 / 5 = 0.28  # noqa: E501
                weighted_avg_f1=0.5,  # (0.6*5 + 0.4*2 + 0.4*3) / 10 = 0.5
                # Other
                support=10,
                true_positives=5,
                false_positives=5,
                false_negatives=5,
            ),
            per_class={
                ResponseCategory.COPILOT: PerClassMetrics(
                    precision=0.6,
                    recall=0.6,
                    f1=0.6,
                    support=5,
                    true_positives=3,
                    false_positives=2,
                    false_negatives=2,
                ),
                ResponseCategory.OUT_OF_SCOPE_DETECTION: PerClassMetrics(
                    precision=0.3333,
                    recall=0.5,
                    f1=0.4,
                    support=2,
                    true_positives=1,
                    false_positives=2,
                    false_negatives=1,
                ),
                ResponseCategory.ROLEPLAY_DETECTION: PerClassMetrics(
                    precision=0.5,
                    recall=0.3333,
                    f1=0.4,
                    support=3,
                    true_positives=1,
                    false_positives=1,
                    false_negatives=2,
                ),
                ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED: PerClassMetrics(
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    support=0,
                    true_positives=0,
                    false_positives=0,
                    false_negatives=0,
                ),
                ResponseCategory.ERROR_FALLBACK: PerClassMetrics(
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    support=0,
                    true_positives=0,
                    false_positives=0,
                    false_negatives=0,
                ),
            },
        )

        evaluator = ResponseClassificationEvaluator()

        # When
        evaluator.evaluate(results)

        # Then
        actual_metrics = evaluator.metrics_summary
        assert actual_metrics is not None

        # Check overall metrics with tolerance for floating point precision
        assert actual_metrics.overall.micro_precision == pytest.approx(
            expected_metrics_summary.overall.micro_precision, abs=1e-4
        )
        assert actual_metrics.overall.macro_precision == pytest.approx(
            expected_metrics_summary.overall.macro_precision, abs=1e-4
        )
        assert actual_metrics.overall.weighted_avg_precision == pytest.approx(
            expected_metrics_summary.overall.weighted_avg_precision, abs=1e-4
        )
        assert actual_metrics.overall.micro_recall == pytest.approx(
            expected_metrics_summary.overall.micro_recall, abs=1e-4
        )
        assert actual_metrics.overall.macro_recall == pytest.approx(
            expected_metrics_summary.overall.macro_recall, abs=1e-4
        )
        assert actual_metrics.overall.weighted_avg_recall == pytest.approx(
            expected_metrics_summary.overall.weighted_avg_recall, abs=1e-4
        )
        assert actual_metrics.overall.micro_f1 == pytest.approx(
            expected_metrics_summary.overall.micro_f1, abs=1e-4
        )
        assert actual_metrics.overall.macro_f1 == pytest.approx(
            expected_metrics_summary.overall.macro_f1, abs=1e-4
        )
        assert actual_metrics.overall.weighted_avg_f1 == pytest.approx(
            expected_metrics_summary.overall.weighted_avg_f1, abs=1e-4
        )

        # Check exact values for counts
        assert (
            actual_metrics.overall.support == expected_metrics_summary.overall.support
        )
        assert (
            actual_metrics.overall.true_positives
            == expected_metrics_summary.overall.true_positives
        )
        assert (
            actual_metrics.overall.false_positives
            == expected_metrics_summary.overall.false_positives
        )
        assert (
            actual_metrics.overall.false_negatives
            == expected_metrics_summary.overall.false_negatives
        )

        # Check per-class metrics with tolerance
        for category in expected_metrics_summary.per_class:
            actual_per_class = actual_metrics.per_class[category]
            expected_per_class = expected_metrics_summary.per_class[category]

            assert actual_per_class.precision == pytest.approx(
                expected_per_class.precision, abs=1e-4
            )
            assert actual_per_class.recall == pytest.approx(
                expected_per_class.recall, abs=1e-4
            )
            assert actual_per_class.f1 == pytest.approx(expected_per_class.f1, abs=1e-4)

            # Check exact values for counts
            assert actual_per_class.support == expected_per_class.support
            assert actual_per_class.true_positives == expected_per_class.true_positives
            assert (
                actual_per_class.false_positives == expected_per_class.false_positives
            )
            assert (
                actual_per_class.false_negatives == expected_per_class.false_negatives
            )

    @pytest.mark.parametrize(
        "results,expected_micro,expected_macro,expected_weighted",
        [
            (
                [],
                0.0,
                0.0,
                0.0,
            ),
            (
                [
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.COPILOT,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.COPILOT,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                        expected=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                    ),
                ],
                1.0,
                pytest.approx(0.4, abs=1e-4),  # (1.0 + 1.0 + 0.0 + 0.0 + 0.0) / 5
                1.0,
            ),
            (
                [
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.ROLEPLAY_DETECTION,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                        expected=ResponseCategory.COPILOT,
                    ),
                ],
                0.0,
                0.0,
                0.0,
            ),
            (
                [
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.COPILOT,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.COPILOT,
                    ),
                ],
                1.0,
                pytest.approx(
                    0.2, abs=1e-4
                ),  # Only COPILOT has precision, others are 0
                1.0,
            ),
        ],
    )
    def test_precision_calculations_edge_cases(
        self,
        results: List[ClassificationResult],
        expected_micro: float,
        expected_macro: float,
        expected_weighted: float,
    ):
        """Test precision calculations with edge cases."""
        evaluator = ResponseClassificationEvaluator()
        evaluator.evaluate(results)

        assert evaluator.calculate_precision("micro") == expected_micro
        assert evaluator.calculate_precision("macro") == expected_macro
        assert evaluator.calculate_precision("weighted") == expected_weighted

    @pytest.mark.parametrize(
        "results,expected_micro,expected_macro,expected_weighted",
        [
            (
                [],
                0.0,
                0.0,
                0.0,
            ),
            (
                [
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.COPILOT,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.COPILOT,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                        expected=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                    ),
                ],
                1.0,
                pytest.approx(0.4, abs=1e-4),  # (1.0 + 1.0 + 0.0 + 0.0 + 0.0) / 5
                1.0,
            ),
            (
                [
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.ROLEPLAY_DETECTION,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                        expected=ResponseCategory.COPILOT,
                    ),
                ],
                0.0,
                0.0,
                0.0,
            ),
            (
                [
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.COPILOT,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.COPILOT,
                    ),
                ],
                1.0,
                pytest.approx(0.2, abs=1e-4),  # Only COPILOT has recall, others are 0
                1.0,
            ),
        ],
    )
    def test_recall_calculations_edge_cases(
        self,
        results: List[ClassificationResult],
        expected_micro: float,
        expected_macro: float,
        expected_weighted: float,
    ):
        """Test recall calculations with edge cases."""
        evaluator = ResponseClassificationEvaluator()
        evaluator.evaluate(results)

        assert evaluator.calculate_recall("micro") == expected_micro
        assert evaluator.calculate_recall("macro") == expected_macro
        assert evaluator.calculate_recall("weighted") == expected_weighted

    @pytest.mark.parametrize(
        "results,expected_micro,expected_macro,expected_weighted",
        [
            (
                [],
                0.0,
                0.0,
                0.0,
            ),
            (
                [
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.COPILOT,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.COPILOT,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                        expected=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                    ),
                ],
                1.0,
                pytest.approx(0.4, abs=1e-4),  # (1.0 + 1.0 + 0.0 + 0.0 + 0.0) / 5
                1.0,
            ),
            (
                [
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.ROLEPLAY_DETECTION,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                        expected=ResponseCategory.COPILOT,
                    ),
                ],
                0.0,
                0.0,
                0.0,
            ),
            (
                [
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.COPILOT,
                    ),
                    ClassificationResult(
                        prediction=ResponseCategory.COPILOT,
                        expected=ResponseCategory.COPILOT,
                    ),
                ],
                1.0,
                pytest.approx(0.2, abs=1e-4),  # Only COPILOT has f1, others are 0
                1.0,
            ),
        ],
    )
    def test_f1_calculations_edge_cases(
        self,
        results: List[ClassificationResult],
        expected_micro: float,
        expected_macro: float,
        expected_weighted: float,
    ):
        """Test F1 calculations with edge cases."""
        evaluator = ResponseClassificationEvaluator()
        evaluator.evaluate(results)

        assert evaluator.calculate_f1("micro") == expected_micro
        assert evaluator.calculate_f1("macro") == expected_macro
        assert evaluator.calculate_f1("weighted") == expected_weighted

    @pytest.mark.parametrize(
        "method_name,invalid_method",
        [
            ("calculate_precision", "invalid"),
            ("calculate_recall", "invalid"),
            ("calculate_f1", "invalid"),
        ],
    )
    def test_invalid_averaging_methods(
        self,
        method_name: str,
        invalid_method: str,
    ):
        """Test that invalid averaging methods raise ValueError."""
        evaluator = ResponseClassificationEvaluator()
        results = [
            ClassificationResult(
                prediction=ResponseCategory.COPILOT,
                expected=ResponseCategory.COPILOT,
            ),
        ]
        evaluator.evaluate(results)

        method = getattr(evaluator, method_name)
        with pytest.raises(ValueError, match="Invalid averaging method"):
            method(invalid_method)
