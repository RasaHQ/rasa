"""Tests for ToolCallEvaluator."""

from types import SimpleNamespace
from typing import List, Optional

import pytest

from rasa.builder.evaluator.evaluators.tool_call.evaluator import ToolCallEvaluator
from rasa.builder.evaluator.evaluators.tool_call.models import ToolCallResult
from rasa.builder.evaluator.tasks.base import ToolCallTaskResult


def _make_item_result(
    output,
    expected_output,
    metadata: Optional[dict] = None,
    item_id: str = "item-1",
) -> SimpleNamespace:
    item = SimpleNamespace(
        id=item_id,
        input={},
        expected_output=expected_output,
        metadata=metadata if metadata is not None else {"category": "default"},
    )
    return SimpleNamespace(output=output, item=item)


def _make_task_output(
    query: str = "q",
    called_tools: Optional[List[str]] = None,
    latency_ms: float = 100.0,
    error: Optional[str] = None,
) -> ToolCallTaskResult:
    return ToolCallTaskResult(
        query=query,
        called_tools=called_tools if called_tools is not None else [],
        latency_ms=latency_ms,
        error=error,
    )


class TestExtractResults:
    def test_happy_path(self):
        output = _make_task_output(
            query="add a slot",
            called_tools=["read_file", "write_file"],
        )
        item_results = [
            _make_item_result(
                output=output,
                expected_output={"expected_tools": ["read_file", "write_file"]},
                metadata={"category": "flow-edit"},
            )
        ]
        evaluator = ToolCallEvaluator()
        results, skip_count = evaluator.extract_results(item_results)

        assert skip_count == 0
        assert len(results) == 1
        r = results[0]
        assert r.query == "add a slot"
        assert r.category == "flow-edit"
        assert r.called_tools == ["read_file", "write_file"]
        assert r.expected_tools == ["read_file", "write_file"]
        assert r.precision == 1.0
        assert r.efficiency == 1.0
        assert r.had_error is False

    def test_skips_none_output(self):
        item_results = [
            _make_item_result(
                output=None,
                expected_output={"expected_tools": ["a"]},
            )
        ]
        evaluator = ToolCallEvaluator()
        results, skip_count = evaluator.extract_results(item_results)
        assert results == []
        assert skip_count == 1

    def test_skips_non_tool_call_task_result(self):
        item_results = [
            _make_item_result(
                output="not a ToolCallTaskResult",
                expected_output={"expected_tools": ["a"]},
            )
        ]
        evaluator = ToolCallEvaluator()
        results, skip_count = evaluator.extract_results(item_results)
        assert results == []
        assert skip_count == 1

    def test_skips_when_expected_output_not_dict(self):
        item_results = [
            _make_item_result(
                output=_make_task_output(called_tools=["a"]),
                expected_output="not a dict",
            )
        ]
        evaluator = ToolCallEvaluator()
        results, skip_count = evaluator.extract_results(item_results)
        assert results == []
        assert skip_count == 1

    def test_filters_non_string_expected_tools(self):
        output = _make_task_output(called_tools=["a"])
        item_results = [
            _make_item_result(
                output=output,
                expected_output={"expected_tools": ["a", 42, None, "b"]},
            )
        ]
        evaluator = ToolCallEvaluator()
        results, _ = evaluator.extract_results(item_results)
        assert results[0].expected_tools == ["a", "b"]

    def test_propagates_had_error_when_output_error_set(self):
        output = _make_task_output(
            called_tools=[],
            error="copilot exploded",
        )
        item_results = [
            _make_item_result(
                output=output,
                expected_output={"expected_tools": ["a"]},
            )
        ]
        evaluator = ToolCallEvaluator()
        results, _ = evaluator.extract_results(item_results)
        assert results[0].had_error is True


class TestEvaluate:
    def _make_result(
        self,
        called: List[str],
        expected: List[str],
        category: str = "flow-edit",
        latency_ms: float = 100.0,
        had_error: bool = False,
        query: str = "q",
    ) -> ToolCallResult:
        # Compute precision/efficiency the same way the production code does
        from rasa.builder.evaluator.evaluators.tool_call.utils import (
            efficiency_score,
            precision_score,
        )

        return ToolCallResult(
            query=query,
            category=category,
            called_tools=called,
            expected_tools=expected,
            precision=precision_score(called, expected),
            efficiency=efficiency_score(called, expected),
            latency_ms=latency_ms,
            had_error=had_error,
        )

    def test_aggregates_mean_precision_and_efficiency(self):
        # r1: precision 1.0, efficiency 1.0
        # r2: precision 0.5 (1 of 2 called in expected), efficiency 0.5
        results = [
            self._make_result(called=["a"], expected=["a"]),
            self._make_result(called=["a", "x"], expected=["a"]),
        ]
        evaluator = ToolCallEvaluator()
        summary = evaluator.evaluate(results)

        assert summary.mean_precision == pytest.approx(0.75)
        assert summary.mean_efficiency == pytest.approx(0.75)

    def test_excludes_error_queries_from_scoring(self):
        # Errored result has wildly off precision/efficiency that would skew
        # the means if not excluded.
        results = [
            self._make_result(called=["a"], expected=["a"]),
            self._make_result(
                called=["x", "y", "z"],
                expected=["a"],
                had_error=True,
            ),
        ]
        evaluator = ToolCallEvaluator()
        summary = evaluator.evaluate(results)

        assert summary.mean_precision == 1.0
        assert summary.mean_efficiency == 1.0
        assert summary.error_rate == 0.5

    def test_excludes_no_expected_tools_from_scoring(self):
        results = [
            self._make_result(called=["a"], expected=["a"]),
            # No expected tools — excluded from precision/efficiency
            self._make_result(called=["b"], expected=[]),
        ]
        evaluator = ToolCallEvaluator()
        summary = evaluator.evaluate(results)

        assert summary.mean_precision == 1.0
        assert summary.mean_efficiency == 1.0
        assert summary.no_expected_tools_count == 1

    def test_no_tools_called_rate_counts_non_error_with_empty_called(self):
        # Total = 3; one non-error empty-called, one errored, one normal.
        # no_tools_called_rate = 1 / 3 (errored is excluded).
        results = [
            self._make_result(called=[], expected=["a"]),
            self._make_result(called=[], expected=["a"], had_error=True),
            self._make_result(called=["a"], expected=["a"]),
        ]
        evaluator = ToolCallEvaluator()
        summary = evaluator.evaluate(results)

        assert summary.no_tools_called_rate == pytest.approx(1 / 3)

    def test_per_category_breakdown(self):
        results = [
            self._make_result(called=["a"], expected=["a"], category="flow-edit"),
            self._make_result(called=["a", "x"], expected=["a"], category="flow-edit"),
            self._make_result(called=["b"], expected=["b"], category="debugging"),
        ]
        evaluator = ToolCallEvaluator()
        summary = evaluator.evaluate(results)

        assert "flow-edit" in summary.per_category
        assert "debugging" in summary.per_category
        assert summary.per_category["flow-edit"].query_count == 2
        assert summary.per_category["debugging"].query_count == 1
        assert summary.per_category["flow-edit"].mean_precision == pytest.approx(0.75)
        assert summary.per_category["debugging"].mean_precision == 1.0

    def test_mean_efficiency_abs_error_is_symmetric(self):
        # Over-call: efficiency 0.5 → |1 - 0.5| = 0.5
        # Under-call: efficiency 2.0 → |1 - 2.0| = 1.0
        # Mean abs error = 0.75; but mean efficiency = 1.25 (asymmetric).
        results = [
            self._make_result(called=["a", "b"], expected=["a"]),  # over-call
            self._make_result(called=["a"], expected=["a", "b"]),  # under-call
        ]
        evaluator = ToolCallEvaluator()
        summary = evaluator.evaluate(results)

        assert summary.mean_efficiency == pytest.approx(1.25)
        assert summary.mean_efficiency_abs_error == pytest.approx(0.75)

    def test_latency_stats_only_over_non_error_results(self):
        results = [
            self._make_result(called=["a"], expected=["a"], latency_ms=100.0),
            self._make_result(called=["b"], expected=["b"], latency_ms=200.0),
            # Errored result has latency 0; should be excluded.
            self._make_result(
                called=[],
                expected=["c"],
                latency_ms=0.0,
                had_error=True,
            ),
        ]
        evaluator = ToolCallEvaluator()
        summary = evaluator.evaluate(results)

        assert summary.latency.mean_ms == 150.0
        assert summary.latency.p50_ms == 150.0

    def test_empty_results_returns_zeros(self):
        evaluator = ToolCallEvaluator()
        summary = evaluator.evaluate([])

        assert summary.mean_precision == 0.0
        assert summary.mean_efficiency == 0.0
        assert summary.mean_efficiency_abs_error == 0.0
        assert summary.error_rate == 0.0
        assert summary.no_tools_called_rate == 0.0
        assert summary.no_expected_tools_count == 0
        assert summary.latency.mean_ms == 0.0
        assert summary.per_category == {}
