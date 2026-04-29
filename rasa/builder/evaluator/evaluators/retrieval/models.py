"""Data models for retrieval evaluation results and metrics."""

from typing import Dict, List

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Per-query retrieval result after matching against ground truth.

    All URLs (retrieved and relevant) are already normalized at this point.
    """

    query: str
    category: str
    retrieved_urls: List[str]
    relevant_urls: List[str]
    latency_ms: float
    had_error: bool


class BiasEntry(BaseModel):
    """A URL that appears frequently in retrieval results.

    High ``bias_score`` indicates the URL is returned often but is rarely
    the relevant document for the query. Useful for detecting retrieval
    blind spots and over-weighted generic pages.
    """

    url: str
    retrieved_in: int = Field(description="Number of queries that returned this URL.")
    relevant_in: int = Field(
        description="Of those queries, how many had this URL as relevant."
    )
    frequency: float = Field(description="retrieved_in / total_queries.")
    relevance_rate: float = Field(description="relevant_in / retrieved_in.")
    bias_score: float = Field(description="frequency * (1 - relevance_rate).")


class LatencyStats(BaseModel):
    """Latency statistics across all queries (excluding errors)."""

    mean_ms: float
    p50_ms: float
    p95_ms: float


class PerCategoryMetrics(BaseModel):
    """Metrics for a single query category."""

    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    query_count: int


class RetrievalMetricsSummary(BaseModel):
    """Aggregate retrieval metrics across the full dataset."""

    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    empty_result_rate: float
    error_rate: float
    no_ground_truth_count: int = Field(
        description="Queries with empty relevant_urls; excluded from recall/MRR."
    )
    latency: LatencyStats
    per_category: Dict[str, PerCategoryMetrics]
    top_biased_urls: List[BiasEntry] = Field(
        description="Top URLs by bias_score; useful for debugging retrieval patterns."
    )
