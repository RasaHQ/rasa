from typing import Dict, List, Literal, Optional

import structlog

from rasa.builder.copilot.models import ResponseCategory
from rasa.builder.evaluator.response_classification.constants import (
    MACRO_AVERAGING_METHOD,
    MICRO_AVERAGING_METHOD,
    WEIGHTED_AVERAGING_METHOD,
)
from rasa.builder.evaluator.response_classification.models import (
    ClassificationResult,
    MetricsSummary,
    OverallClassificationMetrics,
    PerClassMetrics,
)

structlogger = structlog.get_logger()


class ResponseClassificationEvaluator:
    def __init__(self):  # type: ignore[no-untyped-def]
        self._classes: List[ResponseCategory] = [
            ResponseCategory.COPILOT,
            ResponseCategory.OUT_OF_SCOPE_DETECTION,
            ResponseCategory.ROLEPLAY_DETECTION,
            ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED,
            ResponseCategory.ERROR_FALLBACK,
            # TODO: Add the greetings and goodbyes as support once the orchestrator
            #       aproach is implemented
        ]
        self._true_positives_per_class: Dict[ResponseCategory, int] = {
            clazz: 0 for clazz in self._classes
        }
        self._false_positives_per_class: Dict[ResponseCategory, int] = {
            clazz: 0 for clazz in self._classes
        }
        self._false_negatives_per_class: Dict[ResponseCategory, int] = {
            clazz: 0 for clazz in self._classes
        }
        self._support_per_class: Dict[ResponseCategory, int] = {
            clazz: 0 for clazz in self._classes
        }

        self._evaluated = False

    @property
    def metrics_summary(self) -> Optional[MetricsSummary]:
        """Get the metrics summary.

        Returns:
            MetricsSummary with structured per-class and overall metrics if
            the evaluator has been run on the data, otherwise None.
        """
        if not self._evaluated:
            structlogger.warning(
                "evaluator.response_classification_evaluator"
                ".metrics_summary.not_evaluated",
                event_info="Evaluator not evaluated. Returning empty metrics summary.",
            )
            return None

        return self._get_metrics_summary()

    def reset(self) -> None:
        self._true_positives_per_class = {clazz: 0 for clazz in self._classes}
        self._false_positives_per_class = {clazz: 0 for clazz in self._classes}
        self._false_negatives_per_class = {clazz: 0 for clazz in self._classes}
        self._support_per_class = {clazz: 0 for clazz in self._classes}
        self._evaluated = False

    def evaluate(self, item_results: List[ClassificationResult]) -> MetricsSummary:
        """Evaluate the classifier on the given item results."""
        if self._evaluated:
            structlogger.warning(
                "evaluator.response_classification_evaluator.evaluate.already_evaluated",
                event_info="Evaluator already evaluated. Resetting evaluator.",
            )
            self.reset()

        for result in item_results:
            # Skip and raise a warning if the class is not in the list of classes
            if result.expected not in self._classes:
                structlogger.warning(
                    "evaluator.response_classification_evaluator"
                    ".evaluate.class_not_recognized",
                    event_info=(
                        f"Class '{result.expected}' is not recognized. "
                        f"Skipping evaluation for this class."
                    ),
                    expected_class=result.expected,
                    classes=self._classes,
                )
                continue

            # Update support for the expected class
            if result.expected in self._support_per_class:
                self._support_per_class[result.expected] += 1

            # Calculate TP, FP, FN per class
            for clazz in self._classes:
                if result.prediction == clazz and result.expected == clazz:
                    self._true_positives_per_class[clazz] += 1

                elif result.prediction == clazz and result.expected != clazz:
                    self._false_positives_per_class[clazz] += 1

                elif result.prediction != clazz and result.expected == clazz:
                    self._false_negatives_per_class[clazz] += 1

        self._evaluated = True
        return self._get_metrics_summary()

    def calculate_precision_per_class(self, clazz: ResponseCategory) -> float:
        """Calculate precision for a specific response category."""
        tp = self._true_positives_per_class.get(clazz, 0)
        fp = self._false_positives_per_class.get(clazz, 0)

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    def calculate_recall_per_class(self, clazz: ResponseCategory) -> float:
        """Calculate recall for a specific response category."""
        tp = self._true_positives_per_class.get(clazz, 0)
        fn = self._false_negatives_per_class.get(clazz, 0)

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    def calculate_f1_per_class(self, clazz: ResponseCategory) -> float:
        """Calculate F1 score for a specific response category."""
        precision = self.calculate_precision_per_class(clazz)
        recall = self.calculate_recall_per_class(clazz)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def calculate_precision(
        self, average: Literal["micro", "macro", "weighted"] = MICRO_AVERAGING_METHOD
    ) -> float:
        """Calculate precision with specified averaging method."""
        if average == MICRO_AVERAGING_METHOD:
            return self._calculate_micro_precision()
        elif average == MACRO_AVERAGING_METHOD:
            return self._calculate_macro_precision()
        elif average == WEIGHTED_AVERAGING_METHOD:
            return self._calculate_weighted_avg_precision()
        else:
            raise ValueError(f"Invalid averaging method: {average}")

    def _calculate_micro_precision(self) -> float:
        """Calculate overall precision with specified averaging method.

        Calculates the metric globally by aggregating the total true positives, false
        positives, across all classes. Each sample contributes equally to the final
        score.
        """
        total_tp = sum(self._true_positives_per_class.values())
        total_fp = sum(self._false_positives_per_class.values())

        if total_tp + total_fp == 0:
            return 0.0

        return total_tp / (total_tp + total_fp)

    def _calculate_macro_precision(self) -> float:
        """Calculate macro-averaged precision.

        Calculates the metric independently for each class and then takes the
        unweighted average. Each class contributes equally.
        """
        precisions = [
            self.calculate_precision_per_class(clazz) for clazz in self._classes
        ]
        return sum(precisions) / len(precisions) if precisions else 0.0

    def _calculate_weighted_avg_precision(self) -> float:
        """Calculate weighted-averaged precision.

        Calculates the metric independently for each class and then takes the average
        weighted by the class support (number of true samples per class).
        """
        total_support = sum(self._support_per_class.values())
        if total_support == 0:
            return 0.0

        weighted_sum = 0.0
        for clazz in self._classes:
            precision = self.calculate_precision_per_class(clazz)
            support = self._support_per_class.get(clazz, 0)
            weighted_sum += precision * support

        return weighted_sum / total_support

    def calculate_recall(
        self, average: Literal["micro", "macro", "weighted"] = MICRO_AVERAGING_METHOD
    ) -> float:
        """Calculate recall with specified averaging method."""
        if average == MICRO_AVERAGING_METHOD:
            return self._calculate_micro_recall()
        elif average == MACRO_AVERAGING_METHOD:
            return self._calculate_macro_recall()
        elif average == WEIGHTED_AVERAGING_METHOD:
            return self._calculate_weighted_avg_recall()
        else:
            raise ValueError(f"Invalid averaging method: {average}")

    def _calculate_micro_recall(self) -> float:
        """Calculate micro-averaged recall.

        Calculates the metric globally by aggregating the total true positives, false
        negatives, across all classes. Each sample contributes equally to the final
        score.
        """
        total_tp = sum(self._true_positives_per_class.values())
        total_fn = sum(self._false_negatives_per_class.values())

        if total_tp + total_fn == 0:
            return 0.0

        return total_tp / (total_tp + total_fn)

    def _calculate_macro_recall(self) -> float:
        """Calculate macro-averaged recall.

        Calculates the metric independently for each class and then takes the
        unweighted average. Each class contributes equally.
        """
        recalls = [self.calculate_recall_per_class(clazz) for clazz in self._classes]
        return sum(recalls) / len(recalls) if recalls else 0.0

    def _calculate_weighted_avg_recall(self) -> float:
        """Calculate weighted-averaged recall.

        Calculates the metric independently for each class and then takes the average
        weighted by the class support (number of true samples per class).
        """
        total_support = sum(self._support_per_class.values())
        if total_support == 0:
            return 0.0

        weighted_sum = 0.0
        for clazz in self._classes:
            recall = self.calculate_recall_per_class(clazz)
            support = self._support_per_class.get(clazz, 0)
            weighted_sum += recall * support

        return weighted_sum / total_support

    def calculate_f1(
        self, average: Literal["micro", "macro", "weighted"] = MICRO_AVERAGING_METHOD
    ) -> float:
        """Calculate F1 score with specified averaging method."""
        if average == MICRO_AVERAGING_METHOD:
            return self._calculate_micro_f1()
        elif average == MACRO_AVERAGING_METHOD:
            return self._calculate_macro_f1()
        elif average == WEIGHTED_AVERAGING_METHOD:
            return self._calculate_weighted_avg_f1()
        else:
            raise ValueError(f"Invalid averaging method: {average}")

    def _calculate_micro_f1(self) -> float:
        """Calculate micro-averaged F1 score.

        Calculates the metric globally by aggregating the total true positives, false
        positives, and false negatives across all classes. Each sample contributes
        equally to the final score.
        """
        micro_precision = self._calculate_micro_precision()
        micro_recall = self._calculate_micro_recall()

        if micro_precision + micro_recall == 0:
            return 0.0

        return 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    def _calculate_macro_f1(self) -> float:
        """Calculate macro-averaged F1 score.

        Calculates the metric independently for each class and then takes the
        unweighted average. Each class contributes equally.
        """
        f1_scores = [self.calculate_f1_per_class(clazz) for clazz in self._classes]
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    def _calculate_weighted_avg_f1(self) -> float:
        """Calculate weighted F1 score.

        Calculates the metric independently for each class and then takes the average
        weighted by the class support (number of true samples per class).
        """
        total_support = sum(self._support_per_class.values())
        if total_support == 0:
            return 0.0

        weighted_sum = 0.0
        for clazz in self._classes:
            f1 = self.calculate_f1_per_class(clazz)
            support = self._support_per_class.get(clazz, 0)
            weighted_sum += f1 * support

        return weighted_sum / total_support

    def _get_metrics_summary(self) -> MetricsSummary:
        """Get the metrics summary without Optional wrapper.

        This method assumes the evaluator has been evaluated and will always
        return a MetricsSummary.
        """
        # Build per-class metrics
        per_class_metrics: Dict[ResponseCategory, PerClassMetrics] = {}
        for clazz in self._classes:
            per_class_metrics[clazz] = PerClassMetrics(
                precision=self.calculate_precision_per_class(clazz),
                recall=self.calculate_recall_per_class(clazz),
                f1=self.calculate_f1_per_class(clazz),
                support=self._support_per_class.get(clazz, 0),
                true_positives=self._true_positives_per_class.get(clazz, 0),
                false_positives=self._false_positives_per_class.get(clazz, 0),
                false_negatives=self._false_negatives_per_class.get(clazz, 0),
            )

        # Build overall metrics
        overall_metrics = OverallClassificationMetrics(
            micro_precision=self.calculate_precision(MICRO_AVERAGING_METHOD),
            macro_precision=self.calculate_precision(MACRO_AVERAGING_METHOD),
            weighted_avg_precision=self.calculate_precision(WEIGHTED_AVERAGING_METHOD),
            micro_recall=self.calculate_recall(MICRO_AVERAGING_METHOD),
            macro_recall=self.calculate_recall(MACRO_AVERAGING_METHOD),
            weighted_avg_recall=self.calculate_recall(WEIGHTED_AVERAGING_METHOD),
            micro_f1=self.calculate_f1(MICRO_AVERAGING_METHOD),
            macro_f1=self.calculate_f1(MACRO_AVERAGING_METHOD),
            weighted_avg_f1=self.calculate_f1(WEIGHTED_AVERAGING_METHOD),
            support=sum(self._support_per_class.values()),
            true_positives=sum(self._true_positives_per_class.values()),
            false_positives=sum(self._false_positives_per_class.values()),
            false_negatives=sum(self._false_negatives_per_class.values()),
        )
        return MetricsSummary(per_class=per_class_metrics, overall=overall_metrics)
