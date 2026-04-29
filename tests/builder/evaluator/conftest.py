"""Shared fixtures for evaluator tests."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest
import yaml

from rasa.builder.copilot.models import ResponseCategory
from rasa.builder.evaluator.evaluators.classification.models import (
    ClassificationResult,
)


@pytest.fixture()
def copilot_category() -> ResponseCategory:
    return ResponseCategory.COPILOT


@pytest.fixture()
def error_category() -> ResponseCategory:
    return ResponseCategory.ERROR_FALLBACK


@pytest.fixture()
def make_classification_result():
    """Factory for ClassificationResult instances."""

    def _factory(
        prediction: ResponseCategory,
        expected: ResponseCategory,
        input_text: Optional[str] = None,
    ) -> ClassificationResult:
        return ClassificationResult(
            prediction=prediction,
            expected=expected,
            input_text=input_text,
        )

    return _factory


@pytest.fixture()
def make_item_result():
    """Factory for mock ExperimentItemResult objects."""

    def _factory(
        output: Any = None,
        expected_output: Any = None,
        item_id: str = "item-1",
        item_input: Any = None,
    ) -> SimpleNamespace:
        item = SimpleNamespace(
            id=item_id,
            expected_output=expected_output,
            input=item_input,
        )
        return SimpleNamespace(output=output, item=item)

    return _factory


@pytest.fixture()
def valid_config_dict() -> Dict[str, Any]:
    return {
        "name": "test-experiment",
        "description": "A test experiment",
        "dataset_name": "test-dataset",
        "task": "classification",
        "results_dir": "results",
        "formats": ["langfuse", "yaml"],
    }


@pytest.fixture()
def valid_config_yaml(tmp_path: Path, valid_config_dict: Dict[str, Any]) -> str:
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_dict, f)
    return str(config_file)


@pytest.fixture()
def make_runner(tmp_path: Path):
    """Factory that creates an ExperimentRunner without calling __init__.

    Sets internal attributes directly to avoid Langfuse/config dependencies.
    """
    from rasa.builder.evaluator.configs.models import ExperimentConfig
    from rasa.builder.evaluator.runner import (
        AvailableLevels,
        AvailableTasks,
        EvaluatorEntry,
        ExperimentRunner,
    )

    def _factory(**overrides: Any) -> ExperimentRunner:
        runner = object.__new__(ExperimentRunner)

        defaults = {
            "_config": ExperimentConfig(
                name="test",
                description="test desc",
                dataset_name="ds",
                task="classification",
                results_dir=str(tmp_path / "results"),
                formats=["langfuse", "yaml"],
            ),
            "_langfuse": MagicMock(),
            "_task": AvailableTasks.CLASSIFICATION,
            "_registry": {
                AvailableTasks.CLASSIFICATION: EvaluatorEntry(
                    task_cls=MagicMock,
                    eval_cls=MagicMock,
                    level=AvailableLevels.RUN,
                ),
            },
            "_dataset": MagicMock(),
            "_output_dir": tmp_path / "results",
            "_evaluator": MagicMock(),
            "_exporter": MagicMock(),
        }
        defaults.update(overrides)

        for attr, value in defaults.items():
            setattr(runner, attr, value)

        # Ensure output dir exists
        runner._output_dir.mkdir(parents=True, exist_ok=True)

        return runner

    return _factory
