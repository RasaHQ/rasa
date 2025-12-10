"""Langfuse Runner for Copilot Response Evaluator.

This module provides functionality to:
- Run evaluations on datasets using Langfuse experiments
- Track evaluation metrics in Langfuse (item-level evaluations)
- Generate evaluation reports
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import structlog
import yaml  # type: ignore[import-untyped]

from rasa.builder.telemetry.langfuse_compat import langfuse, require_langfuse

# Ensure langfuse is available - raises ImportError if not
require_langfuse()

from langfuse import Evaluation  # noqa: E402, TID251
from langfuse._client.datasets import DatasetClient  # noqa: E402, TID251
from langfuse.experiment import ExperimentResult  # noqa: E402, TID251

from rasa.builder.evaluator.copilot_response_evaluator.constants import (  # noqa: E402
    COMPLETENESS_CONFIDENCE_DESCRIPTION,
    COMPLETENESS_CONFIDENCE_METRIC,
    COMPLETENESS_COVERAGE_RATE_DESCRIPTION,
    COMPLETENESS_COVERAGE_RATE_METRIC,
    COMPLETENESS_COVERED_PARTS_DESCRIPTION,
    COMPLETENESS_COVERED_PARTS_METRIC,
    COMPLETENESS_REASONING_METADATA_DESCRIPTION,
    COMPLETENESS_REASONING_METADATA_METRIC,
    COMPLETENESS_TOTAL_PARTS_DESCRIPTION,
    COMPLETENESS_TOTAL_PARTS_METRIC,
    COMPLETENESS_UNCOVERED_PARTS_DESCRIPTION,
    COMPLETENESS_UNCOVERED_PARTS_METRIC,
    EXPERIMENT_DESCRIPTION,
    EXPERIMENT_NAME,
    FAITHFULNESS_AVG_VERDICT_CONFIDENCE_DESCRIPTION,
    FAITHFULNESS_AVG_VERDICT_CONFIDENCE_METRIC,
    FAITHFULNESS_CONTRADICTED_CLAIMS_COUNT_DESCRIPTION,
    FAITHFULNESS_CONTRADICTED_CLAIMS_COUNT_METRIC,
    FAITHFULNESS_EXTRACTED_CLAIMS_COUNT_DESCRIPTION,
    FAITHFULNESS_EXTRACTED_CLAIMS_COUNT_METRIC,
    FAITHFULNESS_NOT_ENOUGH_INFO_CLAIMS_COUNT_DESCRIPTION,
    FAITHFULNESS_NOT_ENOUGH_INFO_CLAIMS_COUNT_METRIC,
    FAITHFULNESS_REASONING_METADATA_DESCRIPTION,
    FAITHFULNESS_REASONING_METADATA_METRIC,
    FAITHFULNESS_SUPPORT_RATE_DESCRIPTION,
    FAITHFULNESS_SUPPORT_RATE_METRIC,
    FAITHFULNESS_SUPPORTED_CLAIMS_COUNT_DESCRIPTION,
    FAITHFULNESS_SUPPORTED_CLAIMS_COUNT_METRIC,
)
from rasa.builder.evaluator.copilot_response_evaluator.evaluator import (  # noqa: E402
    CopilotResponseEvaluator,
)
from rasa.builder.evaluator.copilot_response_evaluator.models import (  # noqa: E402
    CopilotResponseCompletenessEvaluationResult,
    CopilotResponseFaithfulnessEvaluationResult,
)
from rasa.builder.evaluator.dataset.models import (  # noqa: E402
    DatasetEntry,
    DatasetExpectedOutput,
    DatasetInput,
    DatasetMetadata,
)
from rasa.builder.evaluator.shared.constants import (  # noqa: E402
    COPILOT_RESPONSE_EVALUATION_RESULTS_DIR,
    COPILOT_RESPONSE_EVALUATION_YAML_OUTPUT_FILENAME,
    DEFAULT_COPILOT_RESPONSE_EVALUATION_TEXT_OUTPUT_FILENAME,
)
from rasa.builder.evaluator.shared.langfuse_utils import (  # noqa: E402
    report_langfuse_run_results_to_txt_file,
    run_copilot_task,
)

structlogger = structlog.get_logger()


class CopilotResponseEvaluatorLangfuseRunner:
    """Main class for running Langfuse evaluations on the Copilot responses.

    This runner uses Langfuse's experiment framework with item-level evaluators to
    track faithfulness and completeness metrics for each evaluated response.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        self._langfuse = langfuse.get_client()
        self._dataset = self._retrieve_dataset(dataset_name)
        self._output_dir: Path = (
            Path(output_dir) if output_dir else COPILOT_RESPONSE_EVALUATION_RESULTS_DIR
        )
        self._evaluator = CopilotResponseEvaluator()

    def _retrieve_dataset(self, dataset_name: str) -> DatasetClient:
        """Get the dataset from Langfuse."""
        try:
            return self._langfuse.get_dataset(dataset_name)
        except Exception as e:
            structlogger.error(
                "copilot_response_evaluator.langfuse_runner.init.dataset_not_found",
                event_info=f"Failed to get dataset '{dataset_name}'",
                dataset_name=dataset_name,
                error=str(e),
            )
            raise

    def run_experiment(self) -> ExperimentResult:
        """Run the experiment on the dataset.

        Returns:
            ExperimentResult containing all evaluation results
        """
        result = self._dataset.run_experiment(
            name=EXPERIMENT_NAME,
            description=EXPERIMENT_DESCRIPTION,
            task=run_copilot_task,
            evaluators=[self._response_quality_evaluator],
        )
        self._langfuse.flush()
        try:
            self._report_run_results_to_txt_file(result)
            self._report_yaml_structured_results(result)
        except Exception as e:
            structlogger.error(
                "copilot_response_evaluator.langfuse_runner"
                ".run_experiment.failed_to_report_results_to_files",
                event_info="Failed to report results to text file and YAML file",
                error=str(e),
            )
        return result

    async def _response_quality_evaluator(
        self,
        *,
        input: Any,
        output: Any,
        expected_output: Any,
        metadata: Optional[Dict[str, Any]],
        **kwargs: Dict[str, Any],
    ) -> List[Evaluation]:
        """Item-level evaluator that evaluates copilot response quality.

        This function follows the langfuse.experiment.EvaluatorFunction protocol. It is
        called for each item in the dataset after the copilot task has generated a
        response. It evaluates the response on two quality dimensions:
        - Faithfulness: Whether claims in the response are grounded in evidence.
        - Completeness: Whether the response fully addresses all parts of the user's
          request.

        Args:
            input: The input data for the dataset item.
            output: (CopilotRunResult) The generated output from the copilot task.
            expected_output: The expected output for the dataset item (unused).
            metadata: Optional metadata associated with the dataset item.
            **kwargs: Additional keyword arguments passed by Langfuse.
        """
        # Create a dataset entry from the given input, metadata and the generated
        # output.
        try:
            dataset_input = DatasetInput.model_validate(input)
            dataset_metadata = DatasetMetadata.model_validate(metadata)
            dataset_output = DatasetExpectedOutput(
                answer=output.complete_response,
                response_category=output.response_category,
                references=(
                    output.reference_section.references
                    if output.reference_section
                    else []
                ),
            )
            dataset_entry = DatasetEntry(
                # ID is only for internal DatasetEntry model structure. Langfuse
                # automatically links evaluations to dataset items when this evaluator
                # is called per item. Becuse of this, it's not passed in the call
                # arguments of the evaluator
                id=str(uuid.uuid4()),
                input=dataset_input,
                expected_output=dataset_output,
                metadata=dataset_metadata,
            )
        except Exception as e:
            structlogger.error(
                "copilot_response_evaluator.langfuse_runner.response_quality_evaluator"
                ".failed_to_create_dataset_entry",
                event_info=(
                    "Failed to create DatasetEntry from input, metadata and "
                    "the generated output. Skipping evaluation."
                ),
                error=str(e),
            )
            return []

        # Run the evaluator and parse the results into Langfuse Evaluation objects.
        try:
            faithfulness_results, completeness_results = await self._evaluator.evaluate(
                [dataset_entry]
            )
            faithfulness_evaluations = self._parse_faithfulness_evaluation_results(
                faithfulness_results[0]
            )
            completeness_evaluations = self._parse_completeness_evaluation_results(
                completeness_results[0]
            )
            return [*faithfulness_evaluations, *completeness_evaluations]
        except Exception as e:
            structlogger.error(
                "copilot_response_evaluator.langfuse_runner.response_quality_evaluator"
                ".evaluation_failed",
                event_info="Failed to evaluate dataset item. Skipping evaluation.",
                error=str(e),
            )
            return []

    def _parse_faithfulness_evaluation_results(
        self, result: CopilotResponseFaithfulnessEvaluationResult
    ) -> List[Evaluation]:
        """Parse faithfulness evaluation results into Langfuse Evaluation objects.

        Args:
            result: The faithfulness evaluation result.

        Returns:
            A list of Langfuse Evaluation objects for faithfulness metrics.
        """
        # Use the metrics property which handles all calculations
        metrics = result.metrics

        return [
            # Aggregation metrics for faithfulness.
            Evaluation(
                name=FAITHFULNESS_SUPPORT_RATE_METRIC,
                value=metrics.support_rate,
                comment=FAITHFULNESS_SUPPORT_RATE_DESCRIPTION.format(
                    value=metrics.support_rate
                ),
            ),
            Evaluation(
                name=FAITHFULNESS_AVG_VERDICT_CONFIDENCE_METRIC,
                value=metrics.avg_verdict_confidence,
                comment=FAITHFULNESS_AVG_VERDICT_CONFIDENCE_DESCRIPTION.format(
                    value=metrics.avg_verdict_confidence
                ),
            ),
            # Evaluation reasoning data for faithfulness.
            Evaluation(
                name=FAITHFULNESS_REASONING_METADATA_METRIC,
                value="faithfulness_reasoning_data",
                comment=FAITHFULNESS_REASONING_METADATA_DESCRIPTION,
                metadata=result.evaluation_reasoning_data.model_dump(mode="json"),
            ),
            # Informational metrics for faithfulness.
            Evaluation(
                name=FAITHFULNESS_SUPPORTED_CLAIMS_COUNT_METRIC,
                value=float(metrics.supported_claims_count),
                comment=FAITHFULNESS_SUPPORTED_CLAIMS_COUNT_DESCRIPTION.format(
                    value=metrics.supported_claims_count
                ),
            ),
            Evaluation(
                name=FAITHFULNESS_CONTRADICTED_CLAIMS_COUNT_METRIC,
                value=float(metrics.contradicted_claims_count),
                comment=FAITHFULNESS_CONTRADICTED_CLAIMS_COUNT_DESCRIPTION.format(
                    value=metrics.contradicted_claims_count
                ),
            ),
            Evaluation(
                name=FAITHFULNESS_NOT_ENOUGH_INFO_CLAIMS_COUNT_METRIC,
                value=float(metrics.not_enough_info_claims_count),
                comment=FAITHFULNESS_NOT_ENOUGH_INFO_CLAIMS_COUNT_DESCRIPTION.format(
                    value=metrics.not_enough_info_claims_count
                ),
            ),
            Evaluation(
                name=FAITHFULNESS_EXTRACTED_CLAIMS_COUNT_METRIC,
                value=float(metrics.extracted_claims_count),
                comment=FAITHFULNESS_EXTRACTED_CLAIMS_COUNT_DESCRIPTION.format(
                    value=metrics.extracted_claims_count
                ),
            ),
        ]

    def _parse_completeness_evaluation_results(
        self, result: CopilotResponseCompletenessEvaluationResult
    ) -> List[Evaluation]:
        """Parse completeness evaluation results into Langfuse Evaluation objects.

        Args:
            result: The completeness evaluation result.

        Returns:
            A list of Langfuse Evaluation objects for completeness metrics.
        """
        metrics = result.metrics

        return [
            # Aggregation metrics for completeness.
            Evaluation(
                name=COMPLETENESS_COVERAGE_RATE_METRIC,
                value=metrics.completeness_rate,
                comment=COMPLETENESS_COVERAGE_RATE_DESCRIPTION.format(
                    value=metrics.completeness_rate
                ),
            ),
            Evaluation(
                name=COMPLETENESS_CONFIDENCE_METRIC,
                value=metrics.confidence,
                comment=COMPLETENESS_CONFIDENCE_DESCRIPTION.format(
                    value=metrics.confidence
                ),
            ),
            # Evaluation reasoning data for faithfulness.
            Evaluation(
                name=COMPLETENESS_REASONING_METADATA_METRIC,
                value="completeness_reasoning_data",
                comment=COMPLETENESS_REASONING_METADATA_DESCRIPTION,
                metadata=result.evaluation_reasoning_data.model_dump(mode="json"),
            ),
            # Informational metrics for completeness.
            Evaluation(
                name=COMPLETENESS_COVERED_PARTS_METRIC,
                value=float(metrics.covered_parts_count),
                comment=COMPLETENESS_COVERED_PARTS_DESCRIPTION.format(
                    value=metrics.covered_parts_count
                ),
            ),
            Evaluation(
                name=COMPLETENESS_UNCOVERED_PARTS_METRIC,
                value=float(metrics.uncovered_parts_count),
                comment=COMPLETENESS_UNCOVERED_PARTS_DESCRIPTION.format(
                    value=metrics.uncovered_parts_count
                ),
            ),
            Evaluation(
                name=COMPLETENESS_TOTAL_PARTS_METRIC,
                value=float(metrics.total_parts_count),
                comment=COMPLETENESS_TOTAL_PARTS_DESCRIPTION.format(
                    value=metrics.total_parts_count
                ),
            ),
        ]

    def _report_run_results_to_txt_file(self, result: ExperimentResult) -> None:
        # Add timestamp prefix to filename
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_filename = (
            f"{current_date}_"
            f"{DEFAULT_COPILOT_RESPONSE_EVALUATION_TEXT_OUTPUT_FILENAME}"
        )
        output_path = self._output_dir / timestamped_filename
        report_langfuse_run_results_to_txt_file(result, output_path)

    def _report_yaml_structured_results(self, result: ExperimentResult) -> None:
        """Export evaluation results to a YAML file with structured data."""
        # Ensure results directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Add timestamp prefix to filename
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_filename = (
            f"{current_date}_" f"{COPILOT_RESPONSE_EVALUATION_YAML_OUTPUT_FILENAME}"
        )
        output_path = self._output_dir / timestamped_filename

        # Convert evaluations to structured data
        structured_data: Dict[str, Any] = {
            "experiment": {
                "name": EXPERIMENT_NAME,
                "description": EXPERIMENT_DESCRIPTION,
                "timestamp": datetime.now().isoformat(),
                "run_url": result.dataset_run_url,
                "run_id": result.dataset_run_id,
            },
            "per_item_metrics": [],
        }

        for item_result in result.item_results:
            item_id = cast(Optional[Union[str, uuid.UUID]], item_result.item.id)  # type: ignore[union-attr]
            item_data: Dict[str, Any] = {
                "id": str(item_id) if item_id is not None else None,
                "input": item_result.item.input,  # type: ignore[union-attr]
                "output": (
                    item_result.output.complete_response
                    if item_result.output is not None
                    else None
                ),
                "evaluations": [],
            }
            for evaluation in item_result.evaluations:
                item_evaluation_data: Dict[str, Any] = {
                    "name": evaluation.name,
                    "value": evaluation.value,
                    "description": evaluation.comment,
                }
                if evaluation.metadata:
                    item_evaluation_data["metadata"] = evaluation.metadata
                item_data["evaluations"].append(item_evaluation_data)
            structured_data["per_item_metrics"].append(item_data)

        # Write to YAML file
        with open(str(output_path), "w") as f:
            yaml.dump(structured_data, f, default_flow_style=False, sort_keys=False)

        structlogger.info(
            "langfuse_runner._report_yaml_structured_results.exported",
            event_info="Evaluation results exported to YAML file",
            yaml_file=output_path,
            items_count=len(structured_data["per_item_metrics"]),
        )
