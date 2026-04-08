"""Data models and metric computation for classification evaluation."""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from rasa.builder.copilot.models import ResponseCategory
from rasa.builder.evaluator.evaluators.classification.utils import (
    MACRO_AVERAGING_METHOD,
    WEIGHTED_AVERAGING_METHOD,
    compute_aggregate,
    compute_f1,
    compute_precision,
    compute_recall,
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    input_text: Optional[str] = None
    prediction: ResponseCategory
    expected: ResponseCategory


class PerClassMetrics(BaseModel):
    """Metrics for a single response category."""

    precision: float = Field(ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(ge=0.0, le=1.0, description="Recall score")
    f1: float = Field(ge=0.0, le=1.0, description="F1 score")

    support: int = Field(ge=0, description="Number of actual occurrences.")

    true_positives: int = Field(ge=0, description="Number of true positives.")
    false_positives: int = Field(ge=0, description="Number of false positives.")
    false_negatives: int = Field(ge=0, description="Number of false negatives.")


class OverallClassificationMetrics(BaseModel):
    """Overall evaluation metrics."""

    accuracy: float = Field(ge=0.0, le=1.0, description="Overall accuracy")

    micro_precision: float = Field(
        ge=0.0, le=1.0, description="Micro-averaged Precision"
    )
    macro_precision: float = Field(
        ge=0.0, le=1.0, description="Macro-averaged Precision"
    )
    weighted_avg_precision: float = Field(
        ge=0.0, le=1.0, description="Weighted Precision"
    )

    micro_recall: float = Field(ge=0.0, le=1.0, description="Micro-averaged Recall")
    macro_recall: float = Field(ge=0.0, le=1.0, description="Macro-averaged Recall")
    weighted_avg_recall: float = Field(ge=0.0, le=1.0, description="Weighted Recall")

    micro_f1: float = Field(ge=0.0, le=1.0, description="Micro-averaged F1 score")
    macro_f1: float = Field(ge=0.0, le=1.0, description="Macro-averaged F1 score")
    weighted_avg_f1: float = Field(ge=0.0, le=1.0, description="Weighted F1 score")

    support: int = Field(ge=0, description="Total number of occurrences.")

    true_positives: int = Field(ge=0, description="Total number of true positives.")
    false_positives: int = Field(ge=0, description="Total number of false positives.")
    false_negatives: int = Field(ge=0, description="Total number of false negatives.")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


class ConfusionMatrix(BaseModel):
    """Immutable per-class TP/FP/FN counts built from a list of results."""

    model_config = ConfigDict(frozen=True)

    classes: List[ResponseCategory]
    tp: Dict[ResponseCategory, int]
    fp: Dict[ResponseCategory, int]
    fn: Dict[ResponseCategory, int]
    support: Dict[ResponseCategory, int]

    @classmethod
    def from_results(cls, results: List[ClassificationResult]) -> "ConfusionMatrix":
        # Derive classes from the dataset's expected labels
        known_classes = sorted({r.expected for r in results}, key=lambda c: c.value)
        tp: Dict[ResponseCategory, int] = {c: 0 for c in known_classes}
        fp: Dict[ResponseCategory, int] = {c: 0 for c in known_classes}
        fn: Dict[ResponseCategory, int] = {c: 0 for c in known_classes}
        support: Dict[ResponseCategory, int] = {c: 0 for c in known_classes}

        for result in results:
            support[result.expected] += 1

            for clazz in known_classes:
                if result.prediction == clazz and result.expected == clazz:
                    tp[clazz] += 1
                elif result.prediction == clazz and result.expected != clazz:
                    fp[clazz] += 1
                elif result.prediction != clazz and result.expected == clazz:
                    fn[clazz] += 1

        return cls(classes=known_classes, tp=tp, fp=fp, fn=fn, support=support)


class MetricsSummary(BaseModel):
    """Complete metrics summary with per-class and overall metrics."""

    per_class: Dict[ResponseCategory, PerClassMetrics] = Field(
        description="Per-class metrics"
    )
    overall: OverallClassificationMetrics = Field(description="Overall metrics")

    @classmethod
    def compute(cls, results: List[ClassificationResult]) -> "MetricsSummary":
        """Compute all classification metrics from a list of results."""
        cm = ConfusionMatrix.from_results(results)

        # Per-class metrics
        per_class_precision = {
            c: compute_precision(cm.tp[c], cm.fp[c]) for c in cm.classes
        }
        per_class_recall = {c: compute_recall(cm.tp[c], cm.fn[c]) for c in cm.classes}
        per_class_f1 = {
            c: compute_f1(per_class_precision[c], per_class_recall[c])
            for c in cm.classes
        }

        per_class: Dict[ResponseCategory, PerClassMetrics] = {
            c: PerClassMetrics(
                precision=per_class_precision[c],
                recall=per_class_recall[c],
                f1=per_class_f1[c],
                support=cm.support[c],
                true_positives=cm.tp[c],
                false_positives=cm.fp[c],
                false_negatives=cm.fn[c],
            )
            for c in cm.classes
        }

        # Micro overall counts
        total_tp = sum(cm.tp.values())
        total_fp = sum(cm.fp.values())
        total_fn = sum(cm.fn.values())
        total_support = sum(cm.support.values())

        micro_precision = compute_precision(total_tp, total_fp)
        micro_recall = compute_recall(total_tp, total_fn)

        overall = OverallClassificationMetrics(
            accuracy=total_tp / total_support if total_support > 0 else 0.0,
            micro_precision=micro_precision,
            macro_precision=compute_aggregate(
                per_class_precision, cm.support, MACRO_AVERAGING_METHOD
            ),
            weighted_avg_precision=compute_aggregate(
                per_class_precision, cm.support, WEIGHTED_AVERAGING_METHOD
            ),
            micro_recall=micro_recall,
            macro_recall=compute_aggregate(
                per_class_recall, cm.support, MACRO_AVERAGING_METHOD
            ),
            weighted_avg_recall=compute_aggregate(
                per_class_recall, cm.support, WEIGHTED_AVERAGING_METHOD
            ),
            micro_f1=compute_f1(micro_precision, micro_recall),
            macro_f1=compute_aggregate(
                per_class_f1, cm.support, MACRO_AVERAGING_METHOD
            ),
            weighted_avg_f1=compute_aggregate(
                per_class_f1, cm.support, WEIGHTED_AVERAGING_METHOD
            ),
            support=total_support,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
        )

        return cls(per_class=per_class, overall=overall)
