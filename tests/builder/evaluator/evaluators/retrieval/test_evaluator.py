"""Tests for RetrievalEvaluator."""

from types import SimpleNamespace

import pytest

from rasa.builder.evaluator.evaluators.retrieval.evaluator import RetrievalEvaluator
from rasa.builder.evaluator.evaluators.retrieval.models import (
    BiasEntry,
    RetrievalResult,
)
from rasa.builder.evaluator.tasks.base import RetrievalTaskResult


def _make_item(
    expected_output: dict,
    metadata: dict,
    item_id: str = "item-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        input={},
        expected_output=expected_output,
        metadata=metadata,
    )


def _make_item_result(
    output, expected_output: dict, metadata: dict, item_id: str = "item-1"
) -> SimpleNamespace:
    item = _make_item(expected_output, metadata, item_id)
    return SimpleNamespace(output=output, item=item)


def _make_expected(urls: list) -> dict:
    return {
        "relevant_pages": [
            {"doc_id": "d", "url": u, "confidence": "high"} for u in urls
        ]
    }


class TestExtractResults:
    def test_happy_path(self):
        output = RetrievalTaskResult(
            query="how to add a slot",
            retrieved_urls=["https://rasa.com/docs/slots"],
            retrieved_titles=["Slots"],
            latency_ms=100.0,
        )
        item_results = [
            _make_item_result(
                output=output,
                expected_output=_make_expected(["https://rasa.com/docs/slots"]),
                metadata={"category": "how-to"},
            )
        ]
        evaluator = RetrievalEvaluator()
        results, skip_count = evaluator.extract_results(item_results)

        assert skip_count == 0
        assert len(results) == 1
        assert results[0].query == "how to add a slot"
        assert results[0].category == "how-to"

    def test_skips_non_retrieval_task_result(self):
        item_results = [
            _make_item_result(
                output="not a RetrievalTaskResult",
                expected_output=_make_expected(["https://rasa.com/docs/slots"]),
                metadata={"category": "how-to"},
            )
        ]
        evaluator = RetrievalEvaluator()
        results, skip_count = evaluator.extract_results(item_results)
        assert results == []
        assert skip_count == 1

    def test_skips_none_output(self):
        item_results = [
            _make_item_result(
                output=None,
                expected_output=_make_expected(["https://rasa.com/docs/slots"]),
                metadata={"category": "how-to"},
            )
        ]
        evaluator = RetrievalEvaluator()
        results, skip_count = evaluator.extract_results(item_results)
        assert results == []
        assert skip_count == 1

    def test_normalizes_urls(self):
        output = RetrievalTaskResult(
            query="q",
            retrieved_urls=["https://rasa.com/docs/slots/"],  # trailing slash
            latency_ms=50.0,
        )
        item_results = [
            _make_item_result(
                output=output,
                expected_output=_make_expected(
                    ["https://rasa.com/docs/slots#section"]  # fragment
                ),
                metadata={"category": "how-to"},
            )
        ]
        evaluator = RetrievalEvaluator()
        results, _ = evaluator.extract_results(item_results)

        # Both should normalize to the same URL
        assert results[0].retrieved_urls == ["https://rasa.com/docs/slots"]
        assert results[0].relevant_urls == ["https://rasa.com/docs/slots"]

    def test_defaults_category_when_missing(self):
        output = RetrievalTaskResult(
            query="q", retrieved_urls=["https://a.com"], latency_ms=10.0
        )
        item_results = [
            _make_item_result(
                output=output,
                expected_output=_make_expected(["https://a.com"]),
                metadata={},  # no category
            )
        ]
        evaluator = RetrievalEvaluator()
        results, _ = evaluator.extract_results(item_results)
        assert results[0].category == "uncategorized"


class TestEvaluate:
    def _make_result(
        self,
        retrieved: list,
        relevant: list,
        category: str = "how-to",
        latency_ms: float = 100.0,
        had_error: bool = False,
        query: str = "q",
    ) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            category=category,
            retrieved_urls=retrieved,
            relevant_urls=relevant,
            latency_ms=latency_ms,
            had_error=had_error,
        )

    def test_aggregates_recall_and_mrr(self):
        results = [
            self._make_result(
                retrieved=["https://a.com", "https://b.com"],
                relevant=["https://a.com"],
            ),
            self._make_result(
                retrieved=["https://x.com", "https://c.com"],
                relevant=["https://c.com"],
            ),
        ]
        evaluator = RetrievalEvaluator()
        summary = evaluator.evaluate(results)

        # Both queries hit the relevant URL within top-3
        assert summary.recall_at_3 == 1.0
        # First hit at position 1 for q1, position 2 for q2
        assert summary.mrr == pytest.approx((1.0 + 0.5) / 2)

    def test_excludes_error_queries_from_scoring(self):
        results = [
            self._make_result(
                retrieved=["https://a.com"],
                relevant=["https://a.com"],
            ),
            self._make_result(
                retrieved=[],
                relevant=["https://b.com"],
                had_error=True,
            ),
        ]
        evaluator = RetrievalEvaluator()
        summary = evaluator.evaluate(results)

        # Only 1 query scorable: recall = 1.0, not 0.5
        assert summary.recall_at_3 == 1.0
        assert summary.error_rate == 0.5

    def test_excludes_no_ground_truth_queries(self):
        results = [
            self._make_result(
                retrieved=["https://a.com"],
                relevant=["https://a.com"],
            ),
            # This query has no ground truth — excluded from recall
            self._make_result(
                retrieved=["https://x.com"],
                relevant=[],
            ),
        ]
        evaluator = RetrievalEvaluator()
        summary = evaluator.evaluate(results)

        assert summary.recall_at_3 == 1.0  # Only the scorable query counts
        assert summary.no_ground_truth_count == 1

    def test_per_category_breakdown(self):
        results = [
            self._make_result(
                retrieved=["https://a.com"],
                relevant=["https://a.com"],
                category="how-to",
            ),
            self._make_result(
                retrieved=["https://x.com"],  # miss
                relevant=["https://b.com"],
                category="how-to",
            ),
            self._make_result(
                retrieved=["https://c.com"],
                relevant=["https://c.com"],
                category="concept",
            ),
        ]
        evaluator = RetrievalEvaluator()
        summary = evaluator.evaluate(results)

        assert "how-to" in summary.per_category
        assert "concept" in summary.per_category
        assert summary.per_category["how-to"].recall_at_3 == 0.5
        assert summary.per_category["concept"].recall_at_3 == 1.0
        assert summary.per_category["how-to"].query_count == 2
        assert summary.per_category["concept"].query_count == 1

    def test_computes_latency_stats(self):
        results = [
            self._make_result(
                retrieved=["https://a.com"],
                relevant=["https://a.com"],
                latency_ms=100.0,
            ),
            self._make_result(
                retrieved=["https://b.com"],
                relevant=["https://b.com"],
                latency_ms=200.0,
            ),
        ]
        evaluator = RetrievalEvaluator()
        summary = evaluator.evaluate(results)

        assert summary.latency.mean_ms == 150.0
        assert summary.latency.p50_ms == 150.0

    def test_tracks_empty_result_rate(self):
        results = [
            self._make_result(retrieved=[], relevant=["https://a.com"]),
            self._make_result(retrieved=["https://b.com"], relevant=["https://b.com"]),
        ]
        evaluator = RetrievalEvaluator()
        summary = evaluator.evaluate(results)
        assert summary.empty_result_rate == 0.5


class TestToEvaluations:
    def test_emits_all_overall_metrics(self):
        results = [
            RetrievalResult(
                query="q",
                category="how-to",
                retrieved_urls=["https://a.com"],
                relevant_urls=["https://a.com"],
                latency_ms=100.0,
                had_error=False,
            )
        ]
        evaluator = RetrievalEvaluator()
        summary = evaluator.evaluate(results)
        evaluations = evaluator.to_evaluations(summary, skip_count=0)

        names = {e.name for e in evaluations}
        # All overall metrics should be present
        for required in [
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "empty_result_rate",
            "error_rate",
            "latency_mean_ms",
            "latency_p50_ms",
            "latency_p95_ms",
            "skipped_items",
            "no_ground_truth_count",
        ]:
            assert required in names, f"missing metric: {required}"

    def test_emits_per_category_metrics(self):
        results = [
            RetrievalResult(
                query="q",
                category="debugging",
                retrieved_urls=["https://a.com"],
                relevant_urls=["https://a.com"],
                latency_ms=100.0,
                had_error=False,
            )
        ]
        evaluator = RetrievalEvaluator()
        summary = evaluator.evaluate(results)
        evaluations = evaluator.to_evaluations(summary, skip_count=0)

        names = {e.name for e in evaluations}
        assert "debugging_recall_at_3" in names
        assert "debugging_recall_at_5" in names
        assert "debugging_mrr" in names
        assert "debugging_count" in names


class TestBuildArtifacts:
    def _bias_entry(self, url: str, bias_score: float) -> BiasEntry:
        return BiasEntry(
            url=url,
            retrieved_in=10,
            relevant_in=1,
            frequency=0.5,
            relevance_rate=0.1,
            bias_score=bias_score,
        )

    def test_returns_bias_report(self):
        evaluator = RetrievalEvaluator()
        evaluator.summary = SimpleNamespace(
            top_biased_urls=[
                self._bias_entry("https://rasa.com/docs/intro", 0.45),
                self._bias_entry("https://rasa.com/docs/calm", 0.25),
            ]
        )

        artifacts = evaluator.build_artifacts(timestamp="20260408_120000")

        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert artifact.filename == "20260408_120000_bias_report.csv"
        assert artifact.fieldnames == [
            "url",
            "retrieved_in",
            "relevant_in",
            "frequency",
            "relevance_rate",
            "bias_score",
        ]
        assert len(artifact.rows) == 2
        assert artifact.rows[0]["url"] == "https://rasa.com/docs/intro"

    def test_no_summary_returns_empty(self):
        evaluator = RetrievalEvaluator()
        evaluator.summary = None

        artifacts = evaluator.build_artifacts(timestamp="20260408_120000")

        assert artifacts == []

    def test_empty_biased_urls_returns_empty(self):
        evaluator = RetrievalEvaluator()
        evaluator.summary = SimpleNamespace(top_biased_urls=[])

        artifacts = evaluator.build_artifacts(timestamp="20260408_120000")

        assert artifacts == []

    def test_summary_missing_attribute_returns_empty(self):
        evaluator = RetrievalEvaluator()
        evaluator.summary = SimpleNamespace(other_field="x")

        artifacts = evaluator.build_artifacts(timestamp="20260408_120000")

        assert artifacts == []
