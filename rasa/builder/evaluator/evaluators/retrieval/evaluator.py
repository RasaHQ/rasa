"""Retrieval evaluator — implements BaseEvaluator for retrieval tasks.

Computes page-level retrieval quality metrics (Recall@K, MRR), operational
stats (latency, empty/error rates), per-category breakdowns, and a bias
report identifying URLs that are over-retrieved.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import structlog

from rasa.builder.evaluator.artifacts import Artifact, CSVArtifact
from rasa.builder.evaluator.evaluators.base import BaseEvaluator
from rasa.builder.evaluator.evaluators.retrieval.models import (
    LatencyStats,
    PerCategoryMetrics,
    RetrievalMetricsSummary,
    RetrievalResult,
)
from rasa.builder.evaluator.evaluators.retrieval.utils import (
    compute_bias,
    mean,
    mrr_single,
    normalize_url,
    percentile,
    recall_at_k,
)
from rasa.builder.evaluator.tasks.base import RetrievalTaskResult
from rasa.builder.telemetry.langfuse_integration.langfuse_compat import require_langfuse

require_langfuse()

from langfuse import Evaluation  # noqa: E402, TID251
from langfuse.experiment import ExperimentItemResult  # noqa: E402, TID251

structlogger = structlog.get_logger()

RECALL_KS = (3, 5, 10)
TOP_BIASED_URLS = 10


class RetrievalEvaluator(BaseEvaluator):
    """Run-level evaluator that computes retrieval metrics across all items."""

    def extract_results(
        self, item_results: List[ExperimentItemResult]
    ) -> Tuple[List[RetrievalResult], int]:
        """Parse ExperimentItemResult objects into RetrievalResult objects.

        Normalizes all URLs (retrieved and expected) for consistent comparison.
        """
        results: List[RetrievalResult] = []
        skipped_items: List = []

        for item_result in item_results:
            output = item_result.output
            item = item_result.item

            if output is None or not isinstance(output, RetrievalTaskResult):
                skipped_items.append(getattr(item, "id", None))
                continue

            # Extract expected relevant pages
            expected = getattr(item, "expected_output", None)
            if not isinstance(expected, dict):
                skipped_items.append(getattr(item, "id", None))
                continue

            relevant_pages = expected.get("relevant_pages") or []
            relevant_urls = [
                normalize_url(p["url"])
                for p in relevant_pages
                if isinstance(p, dict) and p.get("url")
            ]

            # Extract category from metadata
            metadata = getattr(item, "metadata", None) or {}
            category = (
                metadata.get("category") if isinstance(metadata, dict) else None
            ) or "uncategorized"

            # Normalize retrieved URLs
            retrieved_urls = [normalize_url(u) for u in output.retrieved_urls if u]

            results.append(
                RetrievalResult(
                    query=output.query,
                    category=category,
                    retrieved_urls=retrieved_urls,
                    relevant_urls=relevant_urls,
                    latency_ms=output.latency_ms,
                    had_error=output.error is not None,
                )
            )

        skip_count = len(skipped_items)
        if skip_count > 0:
            structlogger.warning(
                "evaluators.retrieval.skipped_items",
                skipped=skip_count,
                total=len(item_results),
                skipped_items=skipped_items,
            )

        return results, skip_count

    def evaluate(self, results: List[RetrievalResult]) -> RetrievalMetricsSummary:
        """Compute aggregate retrieval metrics."""
        total = len(results)

        # Error and empty-result rates computed over all queries
        error_results = [r for r in results if r.had_error]
        non_error_results = [r for r in results if not r.had_error]
        empty_results = [r for r in non_error_results if not r.retrieved_urls]

        error_rate = len(error_results) / total if total else 0.0
        empty_result_rate = len(empty_results) / total if total else 0.0

        # Recall/MRR only on queries with ground truth and no error
        scorable = [r for r in non_error_results if r.relevant_urls]
        no_ground_truth_count = len(non_error_results) - len(scorable)

        overall_recall: Dict[int, float] = {}
        for k in RECALL_KS:
            per_query = [
                recall_at_k(r.retrieved_urls, set(r.relevant_urls), k) for r in scorable
            ]
            overall_recall[k] = mean(per_query)

        overall_mrr = mean(
            [mrr_single(r.retrieved_urls, set(r.relevant_urls)) for r in scorable]
        )

        # Latency over successful (non-error) queries
        latency_values = [r.latency_ms for r in non_error_results]
        latency_stats = LatencyStats(
            mean_ms=mean(latency_values),
            p50_ms=percentile(latency_values, 50.0),
            p95_ms=percentile(latency_values, 95.0),
        )

        # Per-category metrics
        by_category: Dict[str, List[RetrievalResult]] = defaultdict(list)
        for r in scorable:
            by_category[r.category].append(r)

        per_category: Dict[str, PerCategoryMetrics] = {}
        for category, category_results in by_category.items():
            per_category[category] = PerCategoryMetrics(
                recall_at_3=mean(
                    [
                        recall_at_k(r.retrieved_urls, set(r.relevant_urls), 3)
                        for r in category_results
                    ]
                ),
                recall_at_5=mean(
                    [
                        recall_at_k(r.retrieved_urls, set(r.relevant_urls), 5)
                        for r in category_results
                    ]
                ),
                recall_at_10=mean(
                    [
                        recall_at_k(r.retrieved_urls, set(r.relevant_urls), 10)
                        for r in category_results
                    ]
                ),
                mrr=mean(
                    [
                        mrr_single(r.retrieved_urls, set(r.relevant_urls))
                        for r in category_results
                    ]
                ),
                query_count=len(category_results),
            )

        # Bias report across non-error queries (ground truth needed to judge
        # relevance rate, so we use only scorable queries)
        top_biased = compute_bias(scorable, top_n=TOP_BIASED_URLS)

        return RetrievalMetricsSummary(
            recall_at_3=overall_recall[3],
            recall_at_5=overall_recall[5],
            recall_at_10=overall_recall[10],
            mrr=overall_mrr,
            empty_result_rate=empty_result_rate,
            error_rate=error_rate,
            no_ground_truth_count=no_ground_truth_count,
            latency=latency_stats,
            per_category=per_category,
            top_biased_urls=top_biased,
        )

    def to_evaluations(
        self, summary: RetrievalMetricsSummary, skip_count: int
    ) -> List[Evaluation]:
        """Convert the summary into Langfuse Evaluation objects.

        The bias report is not exported here — it's saved as a CSV by the
        runner (too many URLs to be useful in the Langfuse UI).
        """
        evaluations: List[Evaluation] = []

        # Overall metrics
        for name, value in [
            ("recall_at_3", summary.recall_at_3),
            ("recall_at_5", summary.recall_at_5),
            ("recall_at_10", summary.recall_at_10),
            ("mrr", summary.mrr),
            ("empty_result_rate", summary.empty_result_rate),
            ("error_rate", summary.error_rate),
            ("latency_mean_ms", summary.latency.mean_ms),
            ("latency_p50_ms", summary.latency.p50_ms),
            ("latency_p95_ms", summary.latency.p95_ms),
        ]:
            evaluations.append(
                Evaluation(
                    name=name,
                    value=value,
                    comment=f"{name}: {value:.4f}",
                )
            )

        # Per-category metrics
        for category, metrics in summary.per_category.items():
            cat = category.lower()
            for metric_name, metric_value in [
                ("recall_at_3", metrics.recall_at_3),
                ("recall_at_5", metrics.recall_at_5),
                ("recall_at_10", metrics.recall_at_10),
                ("mrr", metrics.mrr),
                ("count", float(metrics.query_count)),
            ]:
                evaluations.append(
                    Evaluation(
                        name=f"{cat}_{metric_name}",
                        value=metric_value,
                        comment=f"[{category}] {metric_name}: {metric_value}",
                    )
                )

        evaluations.append(
            Evaluation(
                name="no_ground_truth_count",
                value=summary.no_ground_truth_count,
                comment=(
                    f"{summary.no_ground_truth_count} queries had no ground-truth "
                    "relevant pages and were excluded from recall/MRR."
                ),
            )
        )

        evaluations.append(
            Evaluation(
                name="skipped_items",
                value=skip_count,
                comment=f"Skipped {skip_count} items due to invalid data.",
            )
        )

        return evaluations

    def build_artifacts(self, timestamp: str) -> List[Artifact]:
        """Build the retrieval bias report CSV artifact."""
        if self.summary is None or not hasattr(self.summary, "top_biased_urls"):
            structlogger.warning("evaluators.retrieval.export.no_summary")
            return []

        biased = self.summary.top_biased_urls
        if not biased:
            structlogger.info("evaluators.retrieval.export.empty")
            return []

        fieldnames = [
            "url",
            "retrieved_in",
            "relevant_in",
            "frequency",
            "relevance_rate",
            "bias_score",
        ]
        rows = [entry.model_dump() for entry in biased]
        return [
            CSVArtifact(
                filename=f"{timestamp}_bias_report.csv",
                fieldnames=fieldnames,
                rows=rows,
            )
        ]
