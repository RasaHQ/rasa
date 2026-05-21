"""Data models for tool call evaluation results and metrics."""

from typing import Dict, List

from pydantic import BaseModel, Field


class ToolCallResult(BaseModel):
    """Per-query tool call evaluation result after matching against ground truth.

    Attributes:
        query: The user query.
        category: Metadata category for per-category breakdowns.
        called_tools: Ordered list of tool names actually invoked by the copilot.
        expected_tools: Ordered list of tool names the copilot was expected to call.
        precision: Fraction of called tools that match expected tools.
        efficiency: Unclamped efficiency = ``len(expected) / len(called)``.
            1.0 is ideal; <1.0 means the copilot over-called; >1.0 means it
            under-called. Both directions are inefficient.
        latency_ms: Wall-clock time of the copilot call.
        had_error: Whether the underlying copilot call errored.
    """

    query: str
    category: str
    called_tools: List[str]
    expected_tools: List[str]
    precision: float
    efficiency: float
    latency_ms: float
    had_error: bool


class ToolCallExportRecord(BaseModel):
    """Per-example record exported as a JSONL artifact."""

    query: str
    expected_tools: List[str]
    called_tools: List[str]
    precision: float
    efficiency: float


class LatencyStats(BaseModel):
    """Latency statistics across all queries (excluding errors)."""

    mean_ms: float
    p50_ms: float
    p95_ms: float


class PerCategoryToolCallMetrics(BaseModel):
    """Metrics for a single query category."""

    mean_precision: float
    mean_efficiency: float
    mean_efficiency_abs_error: float = Field(
        description="Mean of |1 - efficiency|; symmetric inefficiency measure."
    )
    query_count: int


class ToolCallMetricsSummary(BaseModel):
    """Aggregate tool call metrics across the full dataset."""

    mean_precision: float = Field(description="Mean of per-query precision scores.")
    mean_efficiency: float = Field(
        description=(
            "Mean of per-query efficiency = expected/called. 1.0 is ideal; "
            "deviation in either direction is bad."
        )
    )
    mean_efficiency_abs_error: float = Field(
        description=(
            "Mean of |1 - efficiency|. Symmetric inefficiency measure; lower is better."
        )
    )
    error_rate: float
    no_tools_called_rate: float = Field(
        description="Fraction of queries where the copilot called zero tools."
    )
    no_expected_tools_count: int = Field(
        description="Queries with empty expected_tools; excluded from precision/efficiency."  # noqa: E501
    )
    latency: LatencyStats
    per_category: Dict[str, PerCategoryToolCallMetrics]
