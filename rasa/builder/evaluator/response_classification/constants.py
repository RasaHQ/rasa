"""Constants for the response classification evaluator."""

from typing import List, Literal

# Options for averaging methods for Response Classification Evaluator
MICRO_AVERAGING_METHOD: Literal["micro"] = "micro"
MACRO_AVERAGING_METHOD: Literal["macro"] = "macro"
WEIGHTED_AVERAGING_METHOD: Literal["weighted"] = "weighted"

AVERAGING_METHODS: List[Literal["micro", "macro", "weighted"]] = [
    MICRO_AVERAGING_METHOD,
    MACRO_AVERAGING_METHOD,
    WEIGHTED_AVERAGING_METHOD,
]

# Overall evaluation metric names
MICRO_PRECISION_METRIC = "micro_precision"
MACRO_PRECISION_METRIC = "macro_precision"
WEIGHTED_PRECISION_METRIC = "weighted_precision"

MICRO_RECALL_METRIC = "micro_recall"
MACRO_RECALL_METRIC = "macro_recall"
WEIGHTED_RECALL_METRIC = "weighted_recall"

MICRO_F1_METRIC = "micro_f1"
MACRO_F1_METRIC = "macro_f1"
WEIGHTED_F1_METRIC = "weighted_f1"

# Skip count metric name due to invalid data
SKIP_COUNT_METRIC = "skipped_items"

# Per-class evaluation metric name templates
PER_CLASS_PRECISION_METRIC_TEMPLATE = "{category}_precision"
PER_CLASS_RECALL_METRIC_TEMPLATE = "{category}_recall"
PER_CLASS_F1_METRIC_TEMPLATE = "{category}_f1"
PER_CLASS_SUPPORT_METRIC_TEMPLATE = "{category}_support"

# Description templates for evaluation metrics
MICRO_PRECISION_DESCRIPTION = "Micro Precision: {value:.3f}"
MACRO_PRECISION_DESCRIPTION = "Macro Precision: {value:.3f}"
WEIGHTED_PRECISION_DESCRIPTION = "Weighted Precision: {value:.3f}"

MICRO_RECALL_DESCRIPTION = "Micro Recall: {value:.3f}"
MACRO_RECALL_DESCRIPTION = "Macro Recall: {value:.3f}"
WEIGHTED_RECALL_DESCRIPTION = "Weighted Recall: {value:.3f}"

MICRO_F1_DESCRIPTION = "Micro F1: {value:.3f}"
MACRO_F1_DESCRIPTION = "Macro F1: {value:.3f}"
WEIGHTED_F1_DESCRIPTION = "Weighted F1: {value:.3f}"

# Skip count metric description
SKIP_COUNT_DESCRIPTION = "Skipped {value} items due to invalid data"

# Per-class description templates
PER_CLASS_PRECISION_DESCRIPTION = "[{category}] Precision: {value:.3f}"
PER_CLASS_RECALL_DESCRIPTION = "[{category}] Recall: {value:.3f}"
PER_CLASS_F1_DESCRIPTION = "[{category}] F1: {value:.3f}"
PER_CLASS_SUPPORT_DESCRIPTION = "[{category}] Support: {value}"

# Experiment configuration
EXPERIMENT_NAME = "Copilot Response Classification Evaluation"
EXPERIMENT_DESCRIPTION = (
    "Evaluating Copilot response classification performance with per-class metrics "
    "and overall averages (micro, macro, weighted). The metric that are reported are: "
    "precision, recall, F1, support."
)
