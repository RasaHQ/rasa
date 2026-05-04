"""Pydantic models for experiment / labeling configuration and loaders."""

from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Union, overload

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class ConfigType(str, Enum):
    EXPERIMENT = "experiment"
    LABELING = "labeling"


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

    # Optional (task-dependent) entries to be added if needed
    retrieval: Optional[RetrievalTaskConfig] = Field(
        default=None,
        description="Configuration for the retrieval task.",
    )


class RetrievalLabelingConfig(BaseModel):
    """Knobs for the retrieval two-stage labeling pipeline."""

    docs_repo_path: str


class ClassifierLabelingConfig(BaseModel):
    """Knobs for the classifier labeling pipeline."""

    valid_categories: List[str] = Field(
        description="Allowed ResponseCategory values the LLM judge may assign."
    )


class LabelingConfig(BaseModel):
    # Required entries
    dataset_name: str
    dataset_description: str
    output_dir: str
    queries_path: str

    model: str
    temperature: float = 0.0
    max_page_content_chars: int = 15_000
    max_retries: int = 1
    batch_size: int = 5
    batch_pause_seconds: float = 3.0

    # Optional task-specific entries
    retrieval: Optional[RetrievalLabelingConfig] = None
    classifier: Optional[ClassifierLabelingConfig] = None


@overload
def load_config(
    config_path: str,
    config_type: Literal[ConfigType.EXPERIMENT],
) -> ExperimentConfig: ...


@overload
def load_config(
    config_path: str,
    config_type: Literal[ConfigType.LABELING],
) -> LabelingConfig: ...


def load_config(
    config_path: str,
    config_type: ConfigType,
) -> Union[ExperimentConfig, LabelingConfig]:
    """Load and validate a config from a YAML file.

    Args:
        config_path: Path to the YAML config file.
        config_type: Which config schema to validate against.

    Returns:
        Validated ``ExperimentConfig`` or ``LabelingConfig`` based on
        ``config_type``.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValidationError: If the YAML does not match the expected schema.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if config_type is ConfigType.EXPERIMENT:
        return ExperimentConfig.model_validate(raw)
    return LabelingConfig.model_validate(raw)
