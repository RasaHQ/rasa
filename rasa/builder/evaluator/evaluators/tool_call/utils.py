"""Metric helpers for tool-call evaluation."""

from typing import List


def precision_score(called: List[str], expected: List[str]) -> float:
    """Precision = |called ∩ expected| / |called|.

    Edge cases:
        - both empty → 1.0 (correctly called nothing)
        - called empty, expected non-empty → 0.0
    """
    if not called and not expected:
        return 1.0
    if not called:
        return 0.0
    expected_set = set(expected)
    hits = sum(1 for tool in called if tool in expected_set)
    return hits / len(called)


def efficiency_score(called: List[str], expected: List[str]) -> float:
    """Unclamped efficiency = |expected| / |called|.

    1.0 is ideal. <1.0 = over-called, >1.0 = under-called. Both sides are
    inefficient, so aggregate this alongside |1 - efficiency| to capture
    symmetric inefficiency.

    Edge cases:
        - both empty → 1.0
        - called empty, expected non-empty → 0.0 (treated as worst-case;
          we return a finite value because metrics aggregate via mean)
    """
    if not called and not expected:
        return 1.0
    if not called:
        return 0.0
    return len(expected) / len(called)
