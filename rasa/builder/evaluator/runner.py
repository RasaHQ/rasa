"""Universal Langfuse experiment runner.

A single runner that handles any experiment defined by a YAML config file.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Type

import structlog

from rasa.builder.evaluator.artifacts import Artifact, YAMLArtifact
from rasa.builder.evaluator.configs.models import (
    ConfigType,
    ExperimentConfig,
    load_config,
)
from rasa.builder.evaluator.evaluators.base import BaseEvaluator
from rasa.builder.evaluator.results_export import ResultsExporter
from rasa.builder.evaluator.tasks.base import AvailableTasks, BaseTask
from rasa.builder.telemetry.langfuse_integration.langfuse_compat import (
    langfuse,
    require_langfuse,
)

require_langfuse()

from langfuse._client.datasets import DatasetClient  # noqa: TID251, E402
from langfuse.experiment import ExperimentResult  # noqa: TID251, E402

structlogger = structlog.get_logger()


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
    from rasa.builder.evaluator.evaluators.retrieval.evaluator import (
        RetrievalEvaluator,
    )
    from rasa.builder.evaluator.tasks.classifier_task import ClassifierTask
    from rasa.builder.evaluator.tasks.retrieval_task import RetrievalTask

    return {
        AvailableTasks.CLASSIFICATION: EvaluatorEntry(
            task_cls=ClassifierTask,
            eval_cls=ClassificationEvaluator,
            level=AvailableLevels.RUN,
        ),
        AvailableTasks.RETRIEVAL: EvaluatorEntry(
            task_cls=RetrievalTask,
            eval_cls=RetrievalEvaluator,
            level=AvailableLevels.RUN,
        ),
    }


class ExperimentRunner:
    """Universal experiment runner driven by YAML config."""

    def __init__(self, config_path: str) -> None:
        self._config: ExperimentConfig = load_config(config_path, ConfigType.EXPERIMENT)
        self._langfuse = langfuse.get_client()

        self._task = self._resolve_task(self._config.task)
        self._registry = _build_evaluator_registry()
        self._dataset = self._retrieve_dataset(self._config.dataset_name)

        self._output_dir = Path(self._config.results_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._exporter = ResultsExporter(output_dir=self._output_dir)

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts: List[Artifact] = list(self._evaluator.build_artifacts(timestamp))
        if "yaml" in self._config.formats:
            artifacts.append(self._build_run_results_artifact(result, timestamp))
        self._exporter.export(artifacts=artifacts)

        return result

    def _build_experiment_kwargs(self, task: AvailableTasks) -> Dict[str, Any]:
        if task not in self._registry:
            raise ValueError(
                f"No evaluator registered for task '{task.value}'. "
                f"Registered tasks: {[t.value for t in self._registry]}"
            )
        entry = self._registry[task]

        task_instance = entry.task_cls(config=self._config)
        self._evaluator = entry.eval_cls()
        eval_key = (
            "run_evaluators" if entry.level == AvailableLevels.RUN else "evaluators"
        )

        return {"task": task_instance.run_task, eval_key: [self._evaluator.run]}

    def _build_run_results_artifact(
        self, experiment_result: ExperimentResult, timestamp: str
    ) -> YAMLArtifact:
        """Compose the experiment-metadata + metrics YAML artifact."""
        summary = self._evaluator.summary
        metrics: Dict[str, Any] = {}
        if summary is not None and hasattr(summary, "model_dump"):
            metrics = summary.model_dump()

        data = {
            "experiment": {
                "name": self._config.name,
                "description": self._config.description,
                "timestamp": datetime.now().isoformat(),
                "run_url": experiment_result.dataset_run_url,
                "run_id": experiment_result.dataset_run_id,
            },
            "metrics": metrics,
        }
        return YAMLArtifact(
            filename=f"{timestamp}_run_results.yaml",
            data=data,
        )
