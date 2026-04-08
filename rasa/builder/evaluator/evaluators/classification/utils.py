"""Metric computation helpers for classification evaluation."""

from typing import Dict, Literal

from rasa.builder.copilot.models import ResponseCategory

MACRO_AVERAGING_METHOD: Literal["macro"] = "macro"
WEIGHTED_AVERAGING_METHOD: Literal["weighted"] = "weighted"


def compute_precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if tp + fp > 0 else 0.0


def compute_recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if tp + fn > 0 else 0.0


def compute_f1(precision: float, recall: float) -> float:
    return (
        2 * (precision * recall) / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def compute_aggregate(
    per_class_values: Dict[ResponseCategory, float],
    support: Dict[ResponseCategory, int],
    method: Literal["macro", "weighted"],
) -> float:
    if method == MACRO_AVERAGING_METHOD:
        values = list(per_class_values.values())
        return sum(values) / len(values) if values else 0.0
    if method == WEIGHTED_AVERAGING_METHOD:
        total = sum(support.values())
        if total == 0:
            return 0.0
        return sum(v * support[c] for c, v in per_class_values.items()) / total
    raise ValueError(f"Invalid averaging method: {method}")
