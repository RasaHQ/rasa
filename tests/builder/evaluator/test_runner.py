"""Tests for ExperimentRunner — orchestration coverage.

File-writing is covered separately in ``test_results_export.py``.
"""

from unittest.mock import MagicMock

import pytest

from rasa.builder.evaluator.runner import (
    AvailableLevels,
    AvailableTasks,
    EvaluatorEntry,
)


class TestResolveTask:
    def test_valid_task(self, make_runner):
        runner = make_runner()
        result = runner._resolve_task("classification")
        assert result == AvailableTasks.CLASSIFICATION

    def test_invalid_task(self, make_runner):
        runner = make_runner()
        with pytest.raises(ValueError, match="Unknown task: 'unknown'"):
            runner._resolve_task("unknown")


class TestRetrieveDataset:
    def test_success(self, make_runner):
        mock_dataset = MagicMock()
        mock_langfuse = MagicMock()
        mock_langfuse.get_dataset.return_value = mock_dataset
        runner = make_runner(_langfuse=mock_langfuse)

        result = runner._retrieve_dataset("my-dataset")

        assert result is mock_dataset
        mock_langfuse.get_dataset.assert_called_once_with("my-dataset")

    def test_failure_reraises(self, make_runner):
        mock_langfuse = MagicMock()
        mock_langfuse.get_dataset.side_effect = RuntimeError("not found")
        runner = make_runner(_langfuse=mock_langfuse)

        with pytest.raises(RuntimeError, match="not found"):
            runner._retrieve_dataset("missing-dataset")


class TestBuildExperimentKwargs:
    def test_run_level(self, make_runner):
        mock_task_cls = MagicMock()
        mock_eval_cls = MagicMock()
        registry = {
            AvailableTasks.CLASSIFICATION: EvaluatorEntry(
                task_cls=mock_task_cls,
                eval_cls=mock_eval_cls,
                level=AvailableLevels.RUN,
            ),
        }
        runner = make_runner(_registry=registry)
        kwargs = runner._build_experiment_kwargs(AvailableTasks.CLASSIFICATION)

        assert "run_evaluators" in kwargs
        assert "evaluators" not in kwargs
        assert "task" in kwargs

    def test_item_level(self, make_runner):
        mock_task_cls = MagicMock()
        mock_eval_cls = MagicMock()
        registry = {
            AvailableTasks.CLASSIFICATION: EvaluatorEntry(
                task_cls=mock_task_cls,
                eval_cls=mock_eval_cls,
                level=AvailableLevels.ITEM,
            ),
        }
        runner = make_runner(_registry=registry)
        kwargs = runner._build_experiment_kwargs(AvailableTasks.CLASSIFICATION)

        assert "evaluators" in kwargs
        assert "run_evaluators" not in kwargs

    def test_unregistered_task_raises(self, make_runner):
        runner = make_runner(_registry={})
        with pytest.raises(ValueError, match="No evaluator registered"):
            runner._build_experiment_kwargs(AvailableTasks.CLASSIFICATION)


class TestRunExperiment:
    def test_orchestration(self, make_runner):
        mock_dataset = MagicMock()
        mock_result = MagicMock()
        mock_dataset.run_experiment.return_value = mock_result
        mock_langfuse = MagicMock()
        mock_exporter = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.build_artifacts.return_value = []
        mock_evaluator.summary = None
        runner = make_runner(
            _dataset=mock_dataset,
            _langfuse=mock_langfuse,
            _exporter=mock_exporter,
            _evaluator=mock_evaluator,
        )
        runner._build_experiment_kwargs = MagicMock(return_value={"task": MagicMock()})

        result = runner.run_experiment()

        assert result is mock_result
        mock_dataset.run_experiment.assert_called_once()
        mock_langfuse.flush.assert_called_once()
        mock_evaluator.build_artifacts.assert_called_once()
        mock_exporter.export.assert_called_once()
        # artifacts passed by keyword
        assert "artifacts" in mock_exporter.export.call_args.kwargs

    def test_yaml_format_appends_run_results_artifact(self, make_runner):
        from rasa.builder.evaluator.artifacts import YAMLArtifact
        from rasa.builder.evaluator.configs.models import ExperimentConfig

        config = ExperimentConfig(
            name="t",
            description="d",
            dataset_name="ds",
            task="classification",
            results_dir="r",
            formats=["yaml"],
        )
        mock_dataset = MagicMock()
        mock_dataset.run_experiment.return_value = MagicMock(
            dataset_run_url="https://example.com/run",
            dataset_run_id="run-123",
        )
        mock_exporter = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.build_artifacts.return_value = []
        mock_evaluator.summary = MagicMock()
        mock_evaluator.summary.model_dump.return_value = {"accuracy": 0.9}
        runner = make_runner(
            _config=config,
            _dataset=mock_dataset,
            _exporter=mock_exporter,
            _evaluator=mock_evaluator,
        )
        runner._build_experiment_kwargs = MagicMock(return_value={"task": MagicMock()})

        runner.run_experiment()

        artifacts = mock_exporter.export.call_args.kwargs["artifacts"]
        assert len(artifacts) == 1
        assert isinstance(artifacts[0], YAMLArtifact)
        assert artifacts[0].data["metrics"] == {"accuracy": 0.9}
        assert artifacts[0].data["experiment"]["run_id"] == "run-123"

    def test_non_yaml_format_skips_run_results_artifact(self, make_runner):
        from rasa.builder.evaluator.configs.models import ExperimentConfig

        config = ExperimentConfig(
            name="t",
            description="d",
            dataset_name="ds",
            task="classification",
            results_dir="r",
            formats=["langfuse"],
        )
        mock_dataset = MagicMock()
        mock_dataset.run_experiment.return_value = MagicMock()
        mock_exporter = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.build_artifacts.return_value = []
        runner = make_runner(
            _config=config,
            _dataset=mock_dataset,
            _exporter=mock_exporter,
            _evaluator=mock_evaluator,
        )
        runner._build_experiment_kwargs = MagicMock(return_value={"task": MagicMock()})

        runner.run_experiment()

        artifacts = mock_exporter.export.call_args.kwargs["artifacts"]
        assert artifacts == []
