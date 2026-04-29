"""Tests for classification metric helper functions."""

import pytest

from rasa.builder.copilot.models import ResponseCategory
from rasa.builder.evaluator.evaluators.classification.utils import (
    compute_aggregate,
    compute_f1,
    compute_precision,
    compute_recall,
)


class TestComputePrecision:
    @pytest.mark.parametrize(
        "tp, fp, expected",
        [
            (3, 1, 0.75),
            (5, 0, 1.0),
            (0, 0, 0.0),
            (0, 5, 0.0),
        ],
    )
    def test_compute_precision(self, tp, fp, expected):
        assert compute_precision(tp, fp) == pytest.approx(expected)


class TestComputeRecall:
    @pytest.mark.parametrize(
        "tp, fn, expected",
        [
            (4, 1, 0.8),
            (5, 0, 1.0),
            (0, 0, 0.0),
            (0, 3, 0.0),
        ],
    )
    def test_compute_recall(self, tp, fn, expected):
        assert compute_recall(tp, fn) == pytest.approx(expected)


class TestComputeF1:
    @pytest.mark.parametrize(
        "precision, recall, expected",
        [
            (0.8, 0.6, 2 * 0.8 * 0.6 / (0.8 + 0.6)),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ],
    )
    def test_compute_f1(self, precision, recall, expected):
        assert compute_f1(precision, recall) == pytest.approx(expected)


class TestComputeAggregate:
    def test_macro_averaging(self):
        values = {ResponseCategory.COPILOT: 0.8, ResponseCategory.ERROR_FALLBACK: 0.6}
        support = {ResponseCategory.COPILOT: 10, ResponseCategory.ERROR_FALLBACK: 5}
        result = compute_aggregate(values, support, "macro")
        assert result == pytest.approx(0.7)

    def test_weighted_averaging(self):
        values = {ResponseCategory.COPILOT: 0.8, ResponseCategory.ERROR_FALLBACK: 0.6}
        support = {ResponseCategory.COPILOT: 10, ResponseCategory.ERROR_FALLBACK: 5}
        result = compute_aggregate(values, support, "weighted")
        expected = (0.8 * 10 + 0.6 * 5) / 15
        assert result == pytest.approx(expected)

    def test_macro_empty(self):
        assert compute_aggregate({}, {}, "macro") == 0.0

    def test_weighted_zero_support(self):
        values = {ResponseCategory.COPILOT: 0.5}
        support = {ResponseCategory.COPILOT: 0}
        assert compute_aggregate(values, support, "weighted") == 0.0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid averaging method"):
            compute_aggregate({}, {}, "invalid")  # type: ignore[arg-type]
