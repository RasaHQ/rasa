"""Tests for classification data models and MetricsSummary.compute."""

import pytest

from rasa.builder.copilot.models import ResponseCategory
from rasa.builder.evaluator.evaluators.classification.models import (
    ClassificationResult,
    ConfusionMatrix,
    MetricsSummary,
)

COPILOT = ResponseCategory.COPILOT
ERROR = ResponseCategory.ERROR_FALLBACK


def _result(
    prediction: ResponseCategory, expected: ResponseCategory
) -> ClassificationResult:
    return ClassificationResult(prediction=prediction, expected=expected)


class TestConfusionMatrix:
    def test_all_correct(self):
        results = [
            _result(COPILOT, COPILOT),
            _result(ERROR, ERROR),
            _result(COPILOT, COPILOT),
        ]
        cm = ConfusionMatrix.from_results(results)

        assert cm.tp[COPILOT] == 2
        assert cm.tp[ERROR] == 1
        assert cm.fp[COPILOT] == 0
        assert cm.fp[ERROR] == 0
        assert cm.fn[COPILOT] == 0
        assert cm.fn[ERROR] == 0
        assert cm.support[COPILOT] == 2
        assert cm.support[ERROR] == 1

    def test_mixed_predictions(self):
        results = [
            _result(COPILOT, COPILOT),  # TP for COPILOT
            _result(ERROR, COPILOT),  # FN for COPILOT, FP for ERROR
            _result(ERROR, ERROR),  # TP for ERROR
            _result(COPILOT, ERROR),  # FN for ERROR, FP for COPILOT
        ]
        cm = ConfusionMatrix.from_results(results)

        assert cm.tp[COPILOT] == 1
        assert cm.fp[COPILOT] == 1
        assert cm.fn[COPILOT] == 1
        assert cm.tp[ERROR] == 1
        assert cm.fp[ERROR] == 1
        assert cm.fn[ERROR] == 1


class TestMetricsSummaryCompute:
    def test_perfect_predictions(self):
        results = [
            _result(COPILOT, COPILOT),
            _result(COPILOT, COPILOT),
            _result(ERROR, ERROR),
        ]
        summary = MetricsSummary.compute(results)

        assert summary.overall.accuracy == pytest.approx(1.0)
        assert summary.overall.micro_f1 == pytest.approx(1.0)
        assert summary.overall.macro_f1 == pytest.approx(1.0)
        assert summary.per_class[COPILOT].precision == pytest.approx(1.0)
        assert summary.per_class[ERROR].recall == pytest.approx(1.0)
