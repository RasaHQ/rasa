"""Tests for tool-call metric helpers."""

import pytest

from rasa.builder.evaluator.evaluators.tool_call.utils import (
    efficiency_score,
    precision_score,
)


class TestPrecisionScore:
    def test_both_empty_returns_one(self):
        assert precision_score([], []) == 1.0

    def test_called_empty_expected_non_empty_returns_zero(self):
        assert precision_score([], ["a"]) == 0.0

    def test_perfect_match_returns_one(self):
        assert precision_score(["a", "b"], ["a", "b"]) == 1.0

    def test_partial_overlap(self):
        # 1 of 2 called tools is in the expected set
        assert precision_score(["a", "x"], ["a", "b"]) == 0.5

    def test_duplicates_in_called_count_toward_denominator(self):
        # ["a", "a"] vs {"a"} → both hits, denominator=2
        assert precision_score(["a", "a"], ["a"]) == 1.0
        # ["a", "a", "x"] vs {"a"} → 2 hits / 3 called
        assert precision_score(["a", "a", "x"], ["a"]) == pytest.approx(2 / 3)


class TestEfficiencyScore:
    def test_both_empty_returns_one(self):
        assert efficiency_score([], []) == 1.0

    def test_called_empty_expected_non_empty_returns_zero(self):
        assert efficiency_score([], ["a"]) == 0.0

    def test_over_call_returns_less_than_one(self):
        # 2 called, 1 expected → efficiency 0.5
        assert efficiency_score(["a", "b"], ["a"]) == 0.5

    def test_under_call_returns_greater_than_one(self):
        # 1 called, 2 expected → efficiency 2.0
        assert efficiency_score(["a"], ["a", "b"]) == 2.0
