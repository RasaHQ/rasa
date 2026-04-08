"""Classification evaluator — implements BaseEvaluator for classification tasks."""

from typing import List, Tuple

import structlog

from rasa.builder.copilot.models import ResponseCategory
from rasa.builder.evaluator.evaluators.base import BaseEvaluator
from rasa.builder.evaluator.evaluators.classification.models import (
    ClassificationResult,
    MetricsSummary,
)
from rasa.builder.evaluator.tasks.base import TaskResult
from rasa.builder.telemetry.langfuse.langfuse_compat import require_langfuse

require_langfuse()

from langfuse import Evaluation  # noqa: E402, TID251
from langfuse.experiment import ExperimentItemResult  # noqa: E402, TID251

structlogger = structlog.get_logger()


class ClassificationEvaluator(BaseEvaluator):
    """Run-level evaluator that computes precision/recall/F1 across all items."""

    def extract_results(
        self, item_results: List[ExperimentItemResult]
    ) -> Tuple[List[ClassificationResult], int]:
        """Extract ClassificationResult pairs from experiment item results."""
        results: List[ClassificationResult] = []
        skipped_items = []

        for item_result in item_results:
            output = item_result.output
            expected = item_result.item.expected_output if item_result.item else None  # type: ignore[union-attr]

            if (
                output is None
                or not isinstance(output, TaskResult)
                or output.predicted_category is None
                or expected is None
                or not isinstance(expected, dict)
                or expected.get("response_category") is None
            ):
                item_id = getattr(item_result.item, "id", None)
                skipped_items.append(item_id)
                continue

            try:
                expected_category = ResponseCategory(expected["response_category"])
            except ValueError:
                item_id = getattr(item_result.item, "id", None)
                structlogger.warning(
                    "evaluators.classification.invalid_response_category",
                    item_id=item_id,
                    response_category=expected["response_category"],
                )
                skipped_items.append(item_id)
                continue

            item_input = getattr(item_result.item, "input", None) or {}
            input_text = (
                item_input.get("message") if isinstance(item_input, dict) else None
            )

            results.append(
                ClassificationResult(
                    input_text=input_text,
                    prediction=output.predicted_category,
                    expected=expected_category,
                )
            )

        skip_count = len(skipped_items)
        if skip_count > 0:
            structlogger.warning(
                "evaluators.classification.skipped_items",
                skipped=skip_count,
                total=len(item_results),
                skipped_items=skipped_items,
            )

        return results, skip_count

    def evaluate(self, results: List[ClassificationResult]) -> MetricsSummary:
        return MetricsSummary.compute(results)

    def to_evaluations(
        self, summary: MetricsSummary, skip_count: int
    ) -> List[Evaluation]:
        """Convert MetricsSummary into Langfuse Evaluation objects."""
        evaluations: List[Evaluation] = []

        # Overall metrics
        overall = summary.overall
        for name, value in [
            ("accuracy", overall.accuracy),
            ("micro_precision", overall.micro_precision),
            ("macro_precision", overall.macro_precision),
            ("weighted_precision", overall.weighted_avg_precision),
            ("micro_recall", overall.micro_recall),
            ("macro_recall", overall.macro_recall),
            ("weighted_recall", overall.weighted_avg_recall),
            ("micro_f1", overall.micro_f1),
            ("macro_f1", overall.macro_f1),
            ("weighted_f1", overall.weighted_avg_f1),
        ]:
            evaluations.append(
                Evaluation(name=name, value=value, comment=f"{name}: {value:.3f}")
            )

        # Per-class metrics
        for category, per_class in summary.per_class.items():
            cat = category.value.lower()
            for metric_name, metric_value in [
                ("precision", per_class.precision),
                ("recall", per_class.recall),
                ("f1", per_class.f1),
                ("support", float(per_class.support)),
            ]:
                evaluations.append(
                    Evaluation(
                        name=f"{cat}_{metric_name}",
                        value=metric_value,
                        comment=f"[{category.value}] {metric_name}: {metric_value}",
                    )
                )

        evaluations.append(
            Evaluation(
                name="skipped_items",
                value=skip_count,
                comment=f"Skipped {skip_count} items due to invalid data",
            )
        )

        return evaluations
