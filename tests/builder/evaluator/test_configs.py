"""Tests for experiment configuration loading."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rasa.builder.evaluator.configs.models import load_config


class TestLoadConfig:
    def test_success(self, valid_config_yaml: str):
        config = load_config(valid_config_yaml)

        assert config.name == "test-experiment"
        assert config.description == "A test experiment"
        assert config.dataset_name == "test-dataset"
        assert config.task == "classification"
        assert config.results_dir == "results"
        assert set(config.formats) == {"langfuse", "yaml"}

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_invalid_schema(self, tmp_path: Path):
        config_file = tmp_path / "bad.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"name": "test"}, f)  # missing required fields

        with pytest.raises(ValidationError):
            load_config(str(config_file))

    def test_invalid_task_value(self, tmp_path: Path, valid_config_dict: dict):
        valid_config_dict["task"] = "regression"
        config_file = tmp_path / "bad_task.yaml"
        with open(config_file, "w") as f:
            yaml.dump(valid_config_dict, f)

        with pytest.raises(ValidationError):
            load_config(str(config_file))
