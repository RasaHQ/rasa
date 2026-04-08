"""Pydantic models for experiment configuration and loader."""

from pathlib import Path
from typing import List, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    # Required entries
    name: str
    description: str
    dataset_name: str
    task: Literal["classification"] = Field(description="Available: classification")
    results_dir: str
    formats: List[Literal["langfuse", "yaml", "txt"]] = Field(
        description="Select all applicable."
    )

    # Optional (task-dependent) entries to be added if needed


def load_config(config_path: str) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Validated ExperimentConfig.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValidationError: If the YAML does not match the expected schema.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return ExperimentConfig.model_validate(raw)
