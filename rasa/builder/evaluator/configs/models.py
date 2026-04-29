"""Pydantic models for experiment configuration and loader."""

from pathlib import Path
from typing import List, Literal, Optional

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class RetrievalTaskConfig(BaseModel):
    """Configuration specific to the retrieval task."""

    backend: Literal["inkeep"] = Field(
        default="inkeep",
        description="Retrieval backend to evaluate.",
    )


class ExperimentConfig(BaseModel):
    # Required entries
    name: str
    description: str
    dataset_name: str
    task: Literal["classification", "retrieval"] = Field(
        description="Available: classification, retrieval"
    )
    results_dir: str
    formats: List[Literal["langfuse", "yaml"]] = Field(
        description="Select all applicable."
    )

    # Optional task-dependent entries
    retrieval: Optional[RetrievalTaskConfig] = Field(
        default=None,
        description="Configuration for the retrieval task.",
    )


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
