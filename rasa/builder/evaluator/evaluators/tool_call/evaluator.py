"""Tool call evaluator — implements BaseEvaluator for tool-call tasks.

Computes set-based tool-call precision and unclamped efficiency
(``expected/called``), aggregates per-category, and exports a per-example
JSONL artifact with query, expected_tools, and called_tools.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import structlog

from rasa.builder.evaluator.artifacts import Artifact, JSONLArtifact
from rasa.builder.evaluator.evaluators.base import BaseEvaluator
from rasa.builder.evaluator.evaluators.tool_call.models import (
    LatencyStats,
    PerCategoryToolCallMetrics,
    ToolCallExportRecord,
    ToolCallMetricsSummary,
    ToolCallResult,
)
from rasa.builder.evaluator.evaluators.tool_call.utils import (
    efficiency_score,
    precision_score,
)
from rasa.builder.evaluator.tasks.base import ToolCallTaskResult
from rasa.builder.telemetry.langfuse_integration.langfuse_compat import require_langfuse

require_langfuse()

from langfuse import Evaluation  # noqa: E402, TID251
from langfuse.experiment import ExperimentItemResult  # noqa: E402, TID251

structlogger = structlog.get_logger()


class ToolCallEvaluator(BaseEvaluator):
    """Run-level evaluator that computes tool-call metrics across all items."""

    def extract_results(
        self, item_results: List[ExperimentItemResult]
    ) -> Tuple[List[ToolCallResult], int]:
        """Parse ExperimentItemResult objects into ToolCallResult objects."""
        results: List[ToolCallResult] = []
        skipped_items: List = []

        for item_result in item_results:
            output = item_result.output
            item = item_result.item

            if output is None or not isinstance(output, ToolCallTaskResult):
                skipped_items.append(getattr(item, "id", None))
                continue

            expected = getattr(item, "expected_output", None)
            if not isinstance(expected, dict):
                skipped_items.append(getattr(item, "id", None))
                continue

            expected_tools = [
                t for t in (expected.get("expected_tools") or []) if isinstance(t, str)
            ]

            metadata = getattr(item, "metadata", None) or {}
            category = (
                metadata.get("category") if isinstance(metadata, dict) else None
            ) or "uncategorized"

            called_tools = list(output.called_tools)

            results.append(
                ToolCallResult(
                    query=output.query,
                    category=category,
                    called_tools=called_tools,
                    expected_tools=expected_tools,
                    precision=precision_score(called_tools, expected_tools),
                    efficiency=efficiency_score(called_tools, expected_tools),
                    latency_ms=output.latency_ms,
                    had_error=output.error is not None,
                )
            )

        skip_count = len(skipped_items)
        if skip_count > 0:
            structlogger.warning(
                "evaluators.tool_call.skipped_items",
                skipped=skip_count,
                total=len(item_results),
                skipped_items=skipped_items,
            )

        return results, skip_count

    def evaluate(self, results: List[ToolCallResult]) -> ToolCallMetricsSummary:
        """Compute aggregate tool-call metrics."""
        total = len(results)

        error_results = [r for r in results if r.had_error]
        non_error_results = [r for r in results if not r.had_error]

        error_rate = (len(error_results) / total) if total else 0.0
        no_tools_called = [r for r in non_error_results if not r.called_tools]
        no_tools_called_rate = (len(no_tools_called) / total) if total else 0.0

        # Precision/efficiency only on items with ground truth and no error.
        scorable = [r for r in non_error_results if r.expected_tools]
        no_expected_tools_count = len(non_error_results) - len(scorable)

        precisions = [r.precision for r in scorable]
        efficiencies = [r.efficiency for r in scorable]
        efficiency_abs_errors = [abs(1.0 - e) for e in efficiencies]

        latency_values = [r.latency_ms for r in non_error_results]
        latency_stats = LatencyStats(
            mean_ms=float(np.mean(latency_values)) if latency_values else 0.0,
            p50_ms=(
                float(np.percentile(latency_values, 50.0)) if latency_values else 0.0
            ),
            p95_ms=(
                float(np.percentile(latency_values, 95.0)) if latency_values else 0.0
            ),
        )

        by_category: Dict[str, List[ToolCallResult]] = defaultdict(list)
        for r in scorable:
            by_category[r.category].append(r)

        per_category: Dict[str, PerCategoryToolCallMetrics] = {}
        for category, category_results in by_category.items():
            cat_efficiencies = [r.efficiency for r in category_results]
            per_category[category] = PerCategoryToolCallMetrics(
                mean_precision=float(np.mean([r.precision for r in category_results])),
                mean_efficiency=float(np.mean(cat_efficiencies)),
                mean_efficiency_abs_error=float(
                    np.mean([abs(1.0 - e) for e in cat_efficiencies])
                ),
                query_count=len(category_results),
            )

        return ToolCallMetricsSummary(
            mean_precision=float(np.mean(precisions)) if precisions else 0.0,
            mean_efficiency=float(np.mean(efficiencies)) if efficiencies else 0.0,
            mean_efficiency_abs_error=(
                float(np.mean(efficiency_abs_errors)) if efficiency_abs_errors else 0.0
            ),
            error_rate=error_rate,
            no_tools_called_rate=no_tools_called_rate,
            no_expected_tools_count=no_expected_tools_count,
            latency=latency_stats,
            per_category=per_category,
        )

    def to_evaluations(
        self, summary: ToolCallMetricsSummary, skip_count: int
    ) -> List[Evaluation]:
        """Convert the summary into Langfuse Evaluation objects."""
        evaluations: List[Evaluation] = []

        for name, value in [
            ("mean_precision", summary.mean_precision),
            ("mean_efficiency", summary.mean_efficiency),
            ("mean_efficiency_abs_error", summary.mean_efficiency_abs_error),
            ("error_rate", summary.error_rate),
            ("no_tools_called_rate", summary.no_tools_called_rate),
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

        for category, metrics in summary.per_category.items():
            cat = category.lower()
            for metric_name, metric_value in [
                ("mean_precision", metrics.mean_precision),
                ("mean_efficiency", metrics.mean_efficiency),
                ("mean_efficiency_abs_error", metrics.mean_efficiency_abs_error),
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
                name="no_expected_tools_count",
                value=summary.no_expected_tools_count,
                comment=(
                    f"{summary.no_expected_tools_count} queries had no expected tools "
                    "and were excluded from precision/efficiency."
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
        """Export per-example JSONL with query, expected_tools, called_tools."""
        if not isinstance(self.results, list) or not self.results:
            structlogger.info("evaluators.tool_call.export.empty")
            return []

        records: List[ToolCallExportRecord] = [
            ToolCallExportRecord(
                query=r.query,
                expected_tools=r.expected_tools,
                called_tools=r.called_tools,
                precision=r.precision,
                efficiency=r.efficiency,
            )
            for r in self.results
            if isinstance(r, ToolCallResult)
        ]

        return [
            JSONLArtifact(
                filename=f"{timestamp}_tool_call_examples.jsonl",
                records=records,
            )
        ]
