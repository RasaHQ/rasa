"""Universal Langfuse experiment runner.

A single runner that handles any experiment defined by a YAML config file.
"""

import csv
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, NamedTuple, Type

import structlog
import yaml  # type: ignore[import-untyped]

from rasa.builder.evaluator.configs.models import ExperimentConfig, load_config
from rasa.builder.evaluator.evaluators.base import BaseEvaluator
from rasa.builder.evaluator.tasks.base import BaseTask
from rasa.builder.telemetry.langfuse.langfuse_compat import langfuse, require_langfuse

require_langfuse()

from langfuse._client.datasets import DatasetClient  # noqa: TID251, E402
from langfuse.experiment import ExperimentResult  # noqa: TID251, E402

structlogger = structlog.get_logger()


class AvailableTasks(str, Enum):
    """Available eval experiment task types."""

    CLASSIFICATION = "classification"


class AvailableLevels(str, Enum):
    """Available Langfuse eval levels."""

    RUN = "run"
    ITEM = "item"


class EvaluatorEntry(NamedTuple):
    """A task class, evaluator class, and its Langfuse execution level."""

    task_cls: Type[BaseTask]
    eval_cls: Type[BaseEvaluator]
    level: AvailableLevels


def _build_evaluator_registry() -> Dict[AvailableTasks, EvaluatorEntry]:
    from rasa.builder.evaluator.evaluators.classification.evaluator import (
        ClassificationEvaluator,
    )
    from rasa.builder.evaluator.tasks.classifier_task import ClassifierTask

    return {
        AvailableTasks.CLASSIFICATION: EvaluatorEntry(
            task_cls=ClassifierTask,
            eval_cls=ClassificationEvaluator,
            level=AvailableLevels.RUN,
        ),
    }


class ExperimentRunner:
    """Universal experiment runner driven by YAML config."""

    def __init__(self, config_path: str) -> None:
        self._config: ExperimentConfig = load_config(config_path)
        self._langfuse = langfuse.get_client()

        self._task = self._resolve_task(self._config.task)
        self._registry = _build_evaluator_registry()
        self._dataset = self._retrieve_dataset(self._config.dataset_name)

        self._output_dir = Path(self._config.results_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _retrieve_dataset(self, dataset_name: str) -> DatasetClient:
        structlogger.info("runner.dataset_retrieval.start", dataset_name=dataset_name)
        try:
            return self._langfuse.get_dataset(dataset_name)
        except Exception as e:
            structlogger.error(
                "runner.dataset_not_found",
                dataset_name=dataset_name,
                error=str(e),
            )
            raise

    def _resolve_task(self, task_name: str) -> AvailableTasks:
        try:
            return AvailableTasks(task_name)
        except ValueError:
            raise ValueError(
                f"Unknown task: '{task_name}'. "
                f"Available tasks: {[t.value for t in AvailableTasks]}"
            )

    def run_experiment(self) -> ExperimentResult:
        """Run the experiment as defined in the config."""
        experiment_kwargs = self._build_experiment_kwargs(self._task)

        result = self._dataset.run_experiment(
            name=self._config.name,
            description=self._config.description,
            **experiment_kwargs,
        )

        self._langfuse.flush()
        self._export_results(result)
        return result

    def _build_experiment_kwargs(self, task: AvailableTasks) -> Dict[str, Any]:
        if task not in self._registry:
            raise ValueError(
                f"No evaluator registered for task '{task.value}'. "
                f"Registered tasks: {[t.value for t in self._registry]}"
            )
        entry = self._registry[task]

        task_instance = entry.task_cls()
        self._evaluator = entry.eval_cls()
        eval_key = (
            "run_evaluators" if entry.level == AvailableLevels.RUN else "evaluators"
        )

        return {"task": task_instance.run_task, eval_key: [self._evaluator.run]}

    def _export_results(self, result: ExperimentResult) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if "txt" in self._config.formats:
            self._write_txt(result, timestamp)

        if "yaml" in self._config.formats:
            self._write_yaml(result, timestamp)

        if self._task == AvailableTasks.CLASSIFICATION:
            self._write_misclassifications_csv(timestamp)

    def _write_txt(self, result: ExperimentResult, timestamp: str) -> None:
        output_path = self._output_dir / f"{timestamp}_run_results.txt"
        try:
            result_str = result.format().replace("\\n", "\n")
            with open(str(output_path), "w") as f:
                f.write(result_str)
            structlogger.info(
                "runner.export.txt",
                file=str(output_path),
            )
        except Exception as e:
            structlogger.error("runner.export.txt_failed", error=str(e))

    def _write_yaml(self, result: ExperimentResult, timestamp: str) -> None:
        output_path = self._output_dir / f"{timestamp}_run_results.yaml"
        try:
            structured_data: Dict[str, Any] = {
                "experiment": {
                    "name": self._config.name,
                    "description": self._config.description,
                    "timestamp": datetime.now().isoformat(),
                    "run_url": result.dataset_run_url,
                    "run_id": result.dataset_run_id,
                },
                "metrics": {},
            }

            summary = self._evaluator.summary
            if summary is not None and hasattr(summary, "model_dump"):
                structured_data["metrics"] = summary.model_dump()

            with open(str(output_path), "w") as f:
                yaml.dump(structured_data, f, default_flow_style=False, sort_keys=False)
            structlogger.info(
                "runner.export.yaml",
                file=str(output_path),
            )
        except Exception as e:
            structlogger.error("runner.export.yaml_failed", error=str(e))

    def _write_misclassifications_csv(self, timestamp: str) -> None:
        from rasa.builder.evaluator.evaluators.classification.models import (
            ClassificationResult,
        )

        results = self._evaluator.results
        if not isinstance(results, list):
            structlogger.warning("runner.export.misclassifications_csv_no_results")
            return

        misclassified = [
            r
            for r in results
            if isinstance(r, ClassificationResult) and r.prediction != r.expected
        ]
        if not misclassified:
            structlogger.info("runner.export.misclassifications_csv_empty")
            return

        output_path = self._output_dir / f"{timestamp}_misclassifications.csv"
        try:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["input_text", "predicted", "expected"]
                )
                writer.writeheader()
                for r in misclassified:
                    writer.writerow(
                        {
                            "input_text": r.input_text or "",
                            "predicted": r.prediction.value,
                            "expected": r.expected.value,
                        }
                    )
            structlogger.info(
                "runner.export.misclassifications_csv",
                file=str(output_path),
                count=len(misclassified),
            )
        except Exception as e:
            structlogger.error(
                "runner.export.misclassifications_csv_failed", error=str(e)
            )
