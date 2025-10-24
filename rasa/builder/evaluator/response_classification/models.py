from typing import Dict

from pydantic import BaseModel, Field

from rasa.builder.copilot.models import ResponseCategory


class ClassificationResult(BaseModel):
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


class MetricsSummary(BaseModel):
    """Complete metrics summary with per-class and overall metrics."""

    per_class: Dict[ResponseCategory, PerClassMetrics] = Field(
        description="Per-class metrics"
    )
    overall: OverallClassificationMetrics = Field(description="Overall metrics")
