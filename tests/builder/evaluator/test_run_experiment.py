"""Tests for run_experiment CLI entry point — thorough coverage."""

from unittest.mock import MagicMock, patch

import pytest

from rasa.builder.evaluator.run_experiment import (
    REQUIRED_ENV_VARS,
    _run,
    _validate_environment,
    main,
)


class TestValidateEnvironment:
    def test_all_present(self, monkeypatch):
        for var in REQUIRED_ENV_VARS:
            monkeypatch.setenv(var, "test-value")

        # Should not raise or exit
        _validate_environment()

    def test_missing_vars_exits(self, monkeypatch):
        for var in REQUIRED_ENV_VARS:
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(SystemExit) as exc_info:
            _validate_environment()

        assert exc_info.value.code == 1


class TestRun:
    @patch("rasa.builder.evaluator.run_experiment.ExperimentRunner")
    def test_success_returns_zero(self, mock_runner_cls):
        mock_runner = MagicMock()
        mock_result = MagicMock()
        mock_result.dataset_run_id = "run-1"
        mock_result.dataset_run_url = "https://example.com"
        mock_runner.run_experiment.return_value = mock_result
        mock_runner_cls.return_value = mock_runner

        exit_code = _run("/path/to/config.yaml")

        assert exit_code == 0
        mock_runner_cls.assert_called_once_with(config_path="/path/to/config.yaml")

    @patch("rasa.builder.evaluator.run_experiment.ExperimentRunner")
    def test_failure_returns_one(self, mock_runner_cls):
        mock_runner_cls.side_effect = RuntimeError("config not found")

        exit_code = _run("/bad/path.yaml")

        assert exit_code == 1


class TestMain:
    @patch("rasa.builder.evaluator.run_experiment._run", return_value=0)
    @patch("rasa.builder.evaluator.run_experiment._validate_environment")
    def test_parses_args_and_runs(self, mock_validate, mock_run, monkeypatch):
        monkeypatch.setattr(
            "sys.argv", ["run_experiment", "--config", "/path/config.yaml"]
        )

        exit_code = main()

        assert exit_code == 0
        mock_validate.assert_called_once()
        mock_run.assert_called_once_with("/path/config.yaml")

    def test_missing_config_arg_exits(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["run_experiment"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 2  # argparse exits with 2
