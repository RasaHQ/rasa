"""Metric computation helpers for retrieval evaluation.

All metrics are URL-based (not chunk-based) — callers must pre-deduplicate
retrieved URLs if they want to avoid counting the same page twice.
"""

from collections import defaultdict
from typing import Dict, List, Set
from urllib.parse import urlparse

import numpy as np

from rasa.builder.evaluator.evaluators.retrieval.models import (
    BiasEntry,
    RetrievalResult,
)


def normalize_url(url: str) -> str:
    """Normalize a URL for consistent comparison.

    Strips trailing slashes, query strings, and fragments. Lowercases the
    scheme and host (paths remain case-sensitive). Returns empty string on
    invalid input rather than raising.
    """
    if not url:
        return ""

    try:
        parsed = urlparse(url.strip())
    except (ValueError, AttributeError):
        return url.strip()

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")

    if not scheme and not netloc:
        # Relative or malformed URL — return the trimmed original
        return url.strip().rstrip("/").split("#")[0].split("?")[0]

    return f"{scheme}://{netloc}{path}"


def deduplicate_urls(urls: List[str]) -> List[str]:
    """Return a list with duplicates removed, preserving order."""
    seen: Set[str] = set()
    result: List[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Page-level Recall@K.

    Args:
        retrieved: Ordered list of URLs returned by the retriever.
        relevant: Set of ground-truth relevant URLs.
        k: Cutoff for top results.

    Returns:
        Fraction of relevant URLs found in the top-k retrieved results.
        Returns 0.0 if ``relevant`` is empty (caller should filter these).
    """
    if not relevant:
        return 0.0

    top_k = deduplicate_urls(retrieved[:k])
    hits = sum(1 for url in top_k if url in relevant)
    return hits / len(relevant)


def mrr_single(retrieved: List[str], relevant: Set[str]) -> float:
    """Reciprocal rank of first relevant URL in retrieved list.

    Uses deduplicated ranks so two chunks from the same URL count as one hit.
    Returns 0.0 if no relevant URL is found or ``relevant`` is empty.
    """
    if not relevant:
        return 0.0

    deduped = deduplicate_urls(retrieved)
    for rank, url in enumerate(deduped, start=1):
        if url in relevant:
            return 1.0 / rank
    return 0.0


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def compute_bias(results: List[RetrievalResult], top_n: int = 10) -> List[BiasEntry]:
    """Identify URLs that are frequently retrieved but rarely relevant.

    Bias score combines retrieval frequency and irrelevance rate:
        bias_score = frequency * (1 - relevance_rate)

    Args:
        results: Per-query retrieval results (errors already filtered out).
        top_n: Number of top biased URLs to return.

    Returns:
        Top-N URLs sorted by bias_score descending.
    """
    if not results:
        return []

    retrieved_count: Dict[str, int] = defaultdict(int)
    relevant_count: Dict[str, int] = defaultdict(int)
    total_queries = len(results)

    for result in results:
        relevant_set = set(result.relevant_urls)
        seen_urls: Set[str] = set()

        for url in result.retrieved_urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)

            retrieved_count[url] += 1
            if url in relevant_set:
                relevant_count[url] += 1

    entries: List[BiasEntry] = []
    for url, count in retrieved_count.items():
        frequency = count / total_queries
        relevance_rate = relevant_count[url] / count
        bias_score = frequency * (1 - relevance_rate)

        entries.append(
            BiasEntry(
                url=url,
                retrieved_in=count,
                relevant_in=relevant_count[url],
                frequency=round(frequency, 4),
                relevance_rate=round(relevance_rate, 4),
                bias_score=round(bias_score, 4),
            )
        )

    entries.sort(key=lambda e: e.bias_score, reverse=True)
    return entries[:top_n]
