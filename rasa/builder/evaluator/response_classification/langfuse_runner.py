from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
import yaml  # type: ignore[import-untyped]

from rasa.builder.copilot.models import (
    ResponseCategory,
)
from rasa.builder.evaluator.response_classification.constants import (
    EXPERIMENT_DESCRIPTION,
    EXPERIMENT_NAME,
    MACRO_F1_DESCRIPTION,
    MACRO_F1_METRIC,
    MACRO_PRECISION_DESCRIPTION,
    MACRO_PRECISION_METRIC,
    MACRO_RECALL_DESCRIPTION,
    MACRO_RECALL_METRIC,
    MICRO_F1_DESCRIPTION,
    MICRO_F1_METRIC,
    MICRO_PRECISION_DESCRIPTION,
    MICRO_PRECISION_METRIC,
    MICRO_RECALL_DESCRIPTION,
    MICRO_RECALL_METRIC,
    PER_CLASS_F1_DESCRIPTION,
    PER_CLASS_F1_METRIC_TEMPLATE,
    PER_CLASS_PRECISION_DESCRIPTION,
    PER_CLASS_PRECISION_METRIC_TEMPLATE,
    PER_CLASS_RECALL_DESCRIPTION,
    PER_CLASS_RECALL_METRIC_TEMPLATE,
    PER_CLASS_SUPPORT_DESCRIPTION,
    PER_CLASS_SUPPORT_METRIC_TEMPLATE,
    SKIP_COUNT_DESCRIPTION,
    SKIP_COUNT_METRIC,
    WEIGHTED_F1_DESCRIPTION,
    WEIGHTED_F1_METRIC,
    WEIGHTED_PRECISION_DESCRIPTION,
    WEIGHTED_PRECISION_METRIC,
    WEIGHTED_RECALL_DESCRIPTION,
    WEIGHTED_RECALL_METRIC,
)
from rasa.builder.evaluator.response_classification.evaluator import (
    ResponseClassificationEvaluator,
)
from rasa.builder.evaluator.response_classification.models import (
    ClassificationResult,
    MetricsSummary,
)
from rasa.builder.evaluator.shared.constants import (
    DEFAULT_RESPONSE_CLASSIFICATION_EVALUATION_TEXT_OUTPUT_FILENAME,
    RESPONSE_CLASSIFICATION_EVALUATION_RESULTS_DIR,
    RESPONSE_CLASSIFICATION_EVALUATION_YAML_OUTPUT_FILENAME,
)
from rasa.builder.evaluator.shared.copilot_executor import (
    CopilotRunResult,
)
from rasa.builder.evaluator.shared.langfuse_utils import (
    report_langfuse_run_results_to_txt_file,
    run_copilot_task,
)
from rasa.builder.telemetry.langfuse_compat import langfuse, require_langfuse

# Ensure langfuse is available - raises ImportError if not
require_langfuse()

# These imports are safe now that we've verified langfuse is available
from langfuse import Evaluation  # noqa: E402, TID251
from langfuse._client.datasets import DatasetClient  # noqa: E402, TID251
from langfuse.experiment import (  # noqa: E402, TID251
    ExperimentItemResult,
    ExperimentResult,
)

structlogger = structlog.get_logger()


class ResponseClassificationLangfuseRunner:
    """Main class for running Langfuse evaluations on the classification evaluator."""

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        self._langfuse = langfuse.get_client()
        self._dataset = self._retrieve_dataset(dataset_name)
        self._output_dir = (
            Path(output_dir)
            if output_dir
            else RESPONSE_CLASSIFICATION_EVALUATION_RESULTS_DIR
        )

    def _retrieve_dataset(self, dataset_name: str) -> DatasetClient:
        """Get the dataset."""
        try:
            return self._langfuse.get_dataset(dataset_name)
        except Exception as e:
            structlogger.error(
                "langfuse_runner.init.dataset_not_found",
                event_info=f"Failed to get dataset '{dataset_name}'",
                dataset_name=dataset_name,
                error=str(e),
            )
            raise

    def run_experiment(self) -> ExperimentResult:
        """Run the experiment."""
        result = self._dataset.run_experiment(
            name=EXPERIMENT_NAME,
            description=EXPERIMENT_DESCRIPTION,
            task=run_copilot_task,
            run_evaluators=[self._run_classification_metrics_evaluator],
        )
        self._report_run_results_to_txt_file(result)
        self._langfuse.flush()
        return result

    def _run_classification_metrics_evaluator(
        self, *, item_results: List[ExperimentItemResult], **kwargs: Dict[str, Any]
    ) -> List[Evaluation]:
        """Main evaluator function that calculates classification metrics.

        This function follows the languse.experiment.RunEvaluatorFunction protocol.
        It will be called after the langfuse.experiment.TaskFunction has been called
        for each item in the dataset.

        Args:
            item_results: The item results to evaluate.
            kwargs: Additional keyword arguments.

        Returns:
            A list of Langfuse Evaluation objects.
        """
        # Create a list of ClassificationResult objects from item_results
        classification_results, skip_count = (
            self._create_classification_results_from_dataset_items(item_results)
        )

        # Log a warning if any items were skipped due to invalid data, as this will
        # affect the overall metrics.
        if skip_count > 0:
            structlogger.warning(
                "langfuse_runner._run_classification_metrics_evaluator.skipped_items",
                event_info=(
                    f"Skipped {skip_count} items due to invalid data. This will affect "
                    f"the overall metrics."
                ),
                skipped_count=skip_count,
                total_items=len(item_results),
            )

        # Run the response classification evaluator on the classification results and
        # get the metrics summary
        evaluator = ResponseClassificationEvaluator()  # type: ignore[no-untyped-call]
        metrics_summary = evaluator.evaluate(classification_results)

        # Record the metrics in Langfuse.
        evaluations = self._create_langfuse_evaluation_objects(
            metrics_summary, skip_count
        )
        self._report_yaml_structured_results(evaluations)
        return evaluations

    def _create_classification_results_from_dataset_items(
        self, item_results: List[ExperimentItemResult]
    ) -> Tuple[List[ClassificationResult], int]:
        """Create a list of ClassificationResult objects from item results.

        Args:
            item_results: The item results to create ClassificationResult objects from.

        Returns:
            A tuple containing the list of ClassificationResult objects and the number
            of items that were skipped due to missing predicted or expected categories.
        """
        classification_results: List[ClassificationResult] = []
        skip_count = 0

        # Try to create a ClassificationResult from the each item result. if either
        # predicted or expected category is missing, skip the item.
        for item_result in item_results:
            # If the output is None, the task function resulted in an error, skip the
            # item.
            if (
                item_result.output is None
                or not isinstance(item_result.output, CopilotRunResult)
                or item_result.output.response_category is None
                or item_result.item.expected_output is None  # type: ignore[union-attr]
                or not isinstance(item_result.item.expected_output, dict)  # type: ignore[union-attr]
                or item_result.item.expected_output.get("response_category") is None  # type: ignore[union-attr]
            ):
                structlogger.error(
                    "langfuse_runner._create_classification_results_from_dataset_items"
                    ".invalid_item_result",
                    event_info=(
                        f"Cannot create a ClassificationResult from item result with "
                        f"id: {item_result.item.id}. This item will not be used for "  # type: ignore[union-attr]
                        f"evaluation."
                    ),
                    item_id=item_result.item.id,  # type: ignore[union-attr]
                    item_output=item_result.output,
                    item_expected_output=item_result.item.expected_output,  # type: ignore[union-attr]
                )
                skip_count += 1
                continue

            predicted_category = item_result.output.response_category.value
            expected_category = item_result.item.expected_output["response_category"]  # type: ignore[union-attr]
            classification_result = ClassificationResult(
                prediction=ResponseCategory(predicted_category),
                expected=ResponseCategory(expected_category),
            )
            classification_results.append(classification_result)

        return classification_results, skip_count

    def _create_langfuse_evaluation_objects(
        self, metrics_summary: MetricsSummary, skip_count: int
    ) -> List[Evaluation]:
        """Create Langfuse Evaluation objects from metrics summary."""
        evaluations: List[Evaluation] = []

        # Overall metrics
        evaluations.extend(
            [
                Evaluation(
                    name=MICRO_PRECISION_METRIC,
                    value=metrics_summary.overall.micro_precision,
                    comment=MICRO_PRECISION_DESCRIPTION.format(
                        value=metrics_summary.overall.micro_precision
                    ),
                ),
                Evaluation(
                    name=MACRO_PRECISION_METRIC,
                    value=metrics_summary.overall.macro_precision,
                    comment=MACRO_PRECISION_DESCRIPTION.format(
                        value=metrics_summary.overall.macro_precision
                    ),
                ),
                Evaluation(
                    name=WEIGHTED_PRECISION_METRIC,
                    value=metrics_summary.overall.weighted_avg_precision,
                    comment=WEIGHTED_PRECISION_DESCRIPTION.format(
                        value=metrics_summary.overall.weighted_avg_precision
                    ),
                ),
                Evaluation(
                    name=MICRO_RECALL_METRIC,
                    value=metrics_summary.overall.micro_recall,
                    comment=MICRO_RECALL_DESCRIPTION.format(
                        value=metrics_summary.overall.micro_recall
                    ),
                ),
                Evaluation(
                    name=MACRO_RECALL_METRIC,
                    value=metrics_summary.overall.macro_recall,
                    comment=MACRO_RECALL_DESCRIPTION.format(
                        value=metrics_summary.overall.macro_recall
                    ),
                ),
                Evaluation(
                    name=WEIGHTED_RECALL_METRIC,
                    value=metrics_summary.overall.weighted_avg_recall,
                    comment=WEIGHTED_RECALL_DESCRIPTION.format(
                        value=metrics_summary.overall.weighted_avg_recall
                    ),
                ),
                Evaluation(
                    name=MICRO_F1_METRIC,
                    value=metrics_summary.overall.micro_f1,
                    comment=MICRO_F1_DESCRIPTION.format(
                        value=metrics_summary.overall.micro_f1
                    ),
                ),
                Evaluation(
                    name=MACRO_F1_METRIC,
                    value=metrics_summary.overall.macro_f1,
                    comment=MACRO_F1_DESCRIPTION.format(
                        value=metrics_summary.overall.macro_f1
                    ),
                ),
                Evaluation(
                    name=WEIGHTED_F1_METRIC,
                    value=metrics_summary.overall.weighted_avg_f1,
                    comment=WEIGHTED_F1_DESCRIPTION.format(
                        value=metrics_summary.overall.weighted_avg_f1
                    ),
                ),
            ]
        )

        # Per-class metrics
        for category, per_class_metrics in metrics_summary.per_class.items():
            category_name = category.value.lower()
            evaluations.extend(
                [
                    Evaluation(
                        name=PER_CLASS_PRECISION_METRIC_TEMPLATE.format(
                            category=category_name
                        ),
                        value=per_class_metrics.precision,
                        comment=PER_CLASS_PRECISION_DESCRIPTION.format(
                            category=category.value,
                            value=per_class_metrics.precision,
                        ),
                    ),
                    Evaluation(
                        name=PER_CLASS_RECALL_METRIC_TEMPLATE.format(
                            category=category_name
                        ),
                        value=per_class_metrics.recall,
                        comment=PER_CLASS_RECALL_DESCRIPTION.format(
                            category=category.value,
                            value=per_class_metrics.recall,
                        ),
                    ),
                    Evaluation(
                        name=PER_CLASS_F1_METRIC_TEMPLATE.format(
                            category=category_name
                        ),
                        value=per_class_metrics.f1,
                        comment=PER_CLASS_F1_DESCRIPTION.format(
                            category=category.value,
                            value=per_class_metrics.f1,
                        ),
                    ),
                    Evaluation(
                        name=PER_CLASS_SUPPORT_METRIC_TEMPLATE.format(
                            category=category_name
                        ),
                        value=float(per_class_metrics.support),
                        comment=PER_CLASS_SUPPORT_DESCRIPTION.format(
                            category=category.value,
                            value=per_class_metrics.support,
                        ),
                    ),
                ]
            )

        # Record the number of items that were skipped due to invalid data
        evaluations.append(
            Evaluation(
                name=SKIP_COUNT_METRIC,
                value=skip_count,
                comment=SKIP_COUNT_DESCRIPTION.format(value=skip_count),
            )
        )

        return evaluations

    def _report_run_results_to_txt_file(self, result: ExperimentResult) -> None:
        """Export evaluation results to a text file."""
        # Add timestamp prefix to filename
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_filename = (
            f"{current_date}_"
            f"{DEFAULT_RESPONSE_CLASSIFICATION_EVALUATION_TEXT_OUTPUT_FILENAME}"
        )
        output_path = self._output_dir / timestamped_filename
        report_langfuse_run_results_to_txt_file(result, output_path)

    def _report_yaml_structured_results(self, evaluations: list[Evaluation]) -> None:
        """Export evaluation results to a YAML file with structured data."""
        # Ensure results directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Add timestamp prefix to filename
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_filename = (
            f"{current_date}_"
            f"{RESPONSE_CLASSIFICATION_EVALUATION_YAML_OUTPUT_FILENAME}"
        )
        output_path = self._output_dir / timestamped_filename
        # Convert evaluations to structured data
        structured_data: Dict[str, Any] = {
            "experiment": {
                "name": EXPERIMENT_NAME,
                "description": EXPERIMENT_DESCRIPTION,
                "timestamp": datetime.now().isoformat(),
            },
            "metrics": [],
        }

        # Add each evaluation as a metric
        for evaluation in evaluations:
            metric_data: Dict[str, Any] = {
                "name": evaluation.name,
                "value": evaluation.value,
                "description": evaluation.comment,
            }
            structured_data["metrics"].append(metric_data)

        # Write to YAML file
        with open(str(output_path), "w") as f:
            yaml.dump(structured_data, f, default_flow_style=False, sort_keys=False)

        structlogger.info(
            "langfuse_runner._report_yaml_structured_results.exported",
            event_info="Evaluation results exported to YAML file",
            yaml_file=output_path,
            metrics_count=len(evaluations),
        )
