"""Artifact descriptors produced by evaluators and written by the exporter.

Keeping this module dependency-free.
"""

from typing import Any, List, Mapping, Sequence, Union

from pydantic import BaseModel, ConfigDict


class CSVArtifact(BaseModel):
    """A CSV file: rows keyed by a fixed set of fieldnames."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: str
    fieldnames: Sequence[str]
    rows: List[Mapping[str, Any]]


class YAMLArtifact(BaseModel):
    """A YAML file: a pre-composed data dict."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: str
    data: Mapping[str, Any]


class JSONLArtifact(BaseModel):
    """A JSONL file: one pydantic model serialized per line."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: str
    records: List[BaseModel]


Artifact = Union[CSVArtifact, YAMLArtifact, JSONLArtifact]
