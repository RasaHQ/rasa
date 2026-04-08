"""Tests for ExperimentRunner — thorough coverage."""

import csv
from unittest.mock import MagicMock

import pytest
import yaml

from rasa.builder.copilot.models import ResponseCategory
from rasa.builder.evaluator.evaluators.classification.models import (
    ClassificationResult,
)
from rasa.builder.evaluator.runner import (
    AvailableLevels,
    AvailableTasks,
    EvaluatorEntry,
)

COPILOT = ResponseCategory.COPILOT
ERROR = ResponseCategory.ERROR_FALLBACK


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


class TestExportResults:
    def test_txt_only(self, make_runner):
        from rasa.builder.evaluator.configs.models import ExperimentConfig

        config = ExperimentConfig(
            name="t",
            description="d",
            dataset_name="ds",
            task="classification",
            results_dir="r",
            formats=["txt"],
        )
        runner = make_runner(_config=config, _task=AvailableTasks.CLASSIFICATION)
        runner._write_txt = MagicMock()
        runner._write_yaml = MagicMock()
        runner._write_misclassifications_csv = MagicMock()

        runner._export_results(MagicMock())

        runner._write_txt.assert_called_once()
        runner._write_yaml.assert_not_called()
        runner._write_misclassifications_csv.assert_called_once()

    def test_yaml_only(self, make_runner):
        from rasa.builder.evaluator.configs.models import ExperimentConfig

        config = ExperimentConfig(
            name="t",
            description="d",
            dataset_name="ds",
            task="classification",
            results_dir="r",
            formats=["yaml"],
        )
        runner = make_runner(_config=config, _task=AvailableTasks.CLASSIFICATION)
        runner._write_txt = MagicMock()
        runner._write_yaml = MagicMock()
        runner._write_misclassifications_csv = MagicMock()

        runner._export_results(MagicMock())

        runner._write_txt.assert_not_called()
        runner._write_yaml.assert_called_once()

    def test_classification_calls_csv(self, make_runner):
        from rasa.builder.evaluator.configs.models import ExperimentConfig

        config = ExperimentConfig(
            name="t",
            description="d",
            dataset_name="ds",
            task="classification",
            results_dir="r",
            formats=["langfuse"],
        )
        runner = make_runner(_config=config, _task=AvailableTasks.CLASSIFICATION)
        runner._write_txt = MagicMock()
        runner._write_yaml = MagicMock()
        runner._write_misclassifications_csv = MagicMock()

        runner._export_results(MagicMock())

        runner._write_misclassifications_csv.assert_called_once()


class TestWriteTxt:
    def test_success(self, make_runner, tmp_path):
        runner = make_runner(_output_dir=tmp_path)
        mock_result = MagicMock()
        mock_result.format.return_value = "line1\\nline2"

        runner._write_txt(mock_result, "20260408_120000")

        output_file = tmp_path / "20260408_120000_run_results.txt"
        assert output_file.exists()
        content = output_file.read_text()
        assert "line1" in content
        assert "line2" in content

    def test_failure_logs_error(self, make_runner, tmp_path):
        runner = make_runner(_output_dir=tmp_path)
        mock_result = MagicMock()
        mock_result.format.side_effect = RuntimeError("format boom")

        # Should not raise
        runner._write_txt(mock_result, "20260408_120000")


class TestWriteYaml:
    def test_with_summary(self, make_runner, tmp_path):
        runner = make_runner(_output_dir=tmp_path)
        mock_evaluator = MagicMock()
        mock_evaluator.summary = MagicMock()
        mock_evaluator.summary.model_dump.return_value = {"accuracy": 0.95}
        runner._evaluator = mock_evaluator

        mock_result = MagicMock()
        mock_result.dataset_run_url = "https://example.com/run"
        mock_result.dataset_run_id = "run-123"

        runner._write_yaml(mock_result, "20260408_120000")

        output_file = tmp_path / "20260408_120000_run_results.yaml"
        assert output_file.exists()
        with open(output_file) as f:
            data = yaml.safe_load(f)
        assert data["metrics"]["accuracy"] == 0.95
        assert data["experiment"]["run_id"] == "run-123"

    def test_no_summary(self, make_runner, tmp_path):
        runner = make_runner(_output_dir=tmp_path)
        mock_evaluator = MagicMock()
        mock_evaluator.summary = None
        runner._evaluator = mock_evaluator

        mock_result = MagicMock()
        mock_result.dataset_run_url = "https://example.com/run"
        mock_result.dataset_run_id = "run-123"

        runner._write_yaml(mock_result, "20260408_120000")

        output_file = tmp_path / "20260408_120000_run_results.yaml"
        with open(output_file) as f:
            data = yaml.safe_load(f)
        assert data["metrics"] == {}


class TestWriteMisclassificationsCsv:
    def test_writes_file(self, make_runner, tmp_path):
        runner = make_runner(_output_dir=tmp_path)
        mock_evaluator = MagicMock()
        mock_evaluator.results = [
            ClassificationResult(
                prediction=COPILOT, expected=ERROR, input_text="wrong one"
            ),
            ClassificationResult(
                prediction=COPILOT, expected=COPILOT, input_text="correct"
            ),
        ]
        runner._evaluator = mock_evaluator

        runner._write_misclassifications_csv("20260408_120000")

        output_file = tmp_path / "20260408_120000_misclassifications.csv"
        assert output_file.exists()
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["input_text"] == "wrong one"
        assert rows[0]["predicted"] == "copilot"
        assert rows[0]["expected"] == "error_fallback"

    def test_no_results_not_list(self, make_runner, tmp_path):
        runner = make_runner(_output_dir=tmp_path)
        mock_evaluator = MagicMock()
        mock_evaluator.results = "not a list"
        runner._evaluator = mock_evaluator

        runner._write_misclassifications_csv("20260408_120000")

        # No file should be written
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 0

    def test_no_misclassifications(self, make_runner, tmp_path):
        runner = make_runner(_output_dir=tmp_path)
        mock_evaluator = MagicMock()
        mock_evaluator.results = [
            ClassificationResult(prediction=COPILOT, expected=COPILOT),
        ]
        runner._evaluator = mock_evaluator

        runner._write_misclassifications_csv("20260408_120000")

        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 0

    def test_csv_write_failure(self, make_runner, tmp_path):
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        runner = make_runner(_output_dir=readonly_dir)
        mock_evaluator = MagicMock()
        mock_evaluator.results = [
            ClassificationResult(prediction=COPILOT, expected=ERROR),
        ]
        runner._evaluator = mock_evaluator

        # Should not raise despite write failure
        runner._write_misclassifications_csv("20260408_120000")

        # Restore permissions for cleanup
        readonly_dir.chmod(0o755)


class TestRunExperiment:
    def test_orchestration(self, make_runner):
        mock_dataset = MagicMock()
        mock_result = MagicMock()
        mock_dataset.run_experiment.return_value = mock_result
        mock_langfuse = MagicMock()
        runner = make_runner(_dataset=mock_dataset, _langfuse=mock_langfuse)
        runner._build_experiment_kwargs = MagicMock(return_value={"task": MagicMock()})
        runner._export_results = MagicMock()

        result = runner.run_experiment()

        assert result is mock_result
        mock_dataset.run_experiment.assert_called_once()
        mock_langfuse.flush.assert_called_once()
        runner._export_results.assert_called_once_with(mock_result)
