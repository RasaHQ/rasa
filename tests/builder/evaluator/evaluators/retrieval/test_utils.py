"""Tests for retrieval metric computation helpers."""

import pytest

from rasa.builder.evaluator.evaluators.retrieval.models import RetrievalResult
from rasa.builder.evaluator.evaluators.retrieval.utils import (
    compute_bias,
    mrr_single,
    normalize_url,
    percentile,
    recall_at_k,
)


class TestNormalizeUrl:
    def test_strips_trailing_slash(self):
        assert (
            normalize_url("https://rasa.com/docs/slots/")
            == "https://rasa.com/docs/slots"
        )

    def test_strips_fragments(self):
        assert (
            normalize_url("https://rasa.com/docs/slots#types")
            == "https://rasa.com/docs/slots"
        )

    def test_strips_query_strings(self):
        assert (
            normalize_url("https://rasa.com/docs/slots?version=1")
            == "https://rasa.com/docs/slots"
        )

    def test_lowercases_scheme_and_host(self):
        assert (
            normalize_url("HTTPS://RASA.COM/docs/Slots")
            == "https://rasa.com/docs/Slots"
        )

    def test_empty_string_returns_empty(self):
        assert normalize_url("") == ""

    def test_handles_relative_url(self):
        # Relative URL has no scheme/netloc — function returns trimmed path
        assert normalize_url("/docs/slots/") == "/docs/slots"


class TestRecallAtK:
    def test_full_hit(self):
        retrieved = ["https://a.com", "https://b.com"]
        relevant = {"https://a.com", "https://b.com"}
        assert recall_at_k(retrieved, relevant, 5) == 1.0

    def test_partial_hit(self):
        retrieved = ["https://a.com", "https://x.com"]
        relevant = {"https://a.com", "https://b.com"}
        assert recall_at_k(retrieved, relevant, 5) == 0.5

    def test_deduplicates_urls_so_two_chunks_from_same_page_count_once(self):
        # Two chunks of page A should count as one hit toward recall
        retrieved = ["https://a.com", "https://a.com", "https://a.com"]
        relevant = {"https://a.com", "https://b.com"}
        assert recall_at_k(retrieved, relevant, 5) == 0.5

    def test_respects_k_cutoff(self):
        retrieved = ["https://x.com", "https://y.com", "https://a.com"]
        relevant = {"https://a.com"}
        assert recall_at_k(retrieved, relevant, 2) == 0.0
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["https://a.com"], set(), 5) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert recall_at_k([], {"https://a.com"}, 5) == 0.0


class TestMrrSingle:
    def test_hit_at_first_position(self):
        assert mrr_single(["https://a.com", "https://b.com"], {"https://a.com"}) == 1.0

    def test_hit_at_third_position(self):
        retrieved = ["https://x.com", "https://y.com", "https://a.com"]
        assert mrr_single(retrieved, {"https://a.com"}) == pytest.approx(1 / 3)

    def test_no_hit_returns_zero(self):
        assert mrr_single(["https://x.com"], {"https://a.com"}) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert mrr_single(["https://a.com"], set()) == 0.0

    def test_deduplicates_before_ranking(self):
        # First occurrence defines the rank even if URL repeats earlier
        retrieved = ["https://a.com", "https://a.com", "https://b.com"]
        assert mrr_single(retrieved, {"https://b.com"}) == 0.5


class TestPercentile:
    def test_p50_of_ten_items(self):
        # Median interpolates between 5th and 6th values (5 and 6)
        assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5.5

    def test_p0_returns_min(self):
        assert percentile([10, 20, 30], 0) == 10

    def test_p100_returns_max(self):
        assert percentile([10, 20, 30], 100) == 30

    def test_single_value_returns_that_value(self):
        assert percentile([42.0], 95) == 42.0

    def test_empty_returns_zero(self):
        assert percentile([], 50) == 0.0


class TestComputeBias:
    def test_identifies_over_retrieved_url(self):
        # URL "noise.com" returned every time but never relevant → high bias
        results = [
            RetrievalResult(
                query=f"q{i}",
                category="test",
                retrieved_urls=["https://noise.com", "https://correct.com"],
                relevant_urls=["https://correct.com"],
                latency_ms=10.0,
                had_error=False,
            )
            for i in range(5)
        ]
        report = compute_bias(results, top_n=10)

        # The noise URL should be at the top with high bias_score
        assert report[0].url == "https://noise.com"
        assert report[0].frequency == 1.0
        assert report[0].relevance_rate == 0.0
        assert report[0].bias_score == 1.0

    def test_respects_top_n_limit(self):
        results = [
            RetrievalResult(
                query=f"q{i}",
                category="test",
                retrieved_urls=[f"https://u{j}.com" for j in range(5)],
                relevant_urls=[],
                latency_ms=10.0,
                had_error=False,
            )
            for i in range(3)
        ]
        report = compute_bias(results, top_n=2)
        assert len(report) == 2

    def test_empty_results_returns_empty(self):
        assert compute_bias([], top_n=10) == []

    def test_deduplicates_urls_within_query(self):
        # URL retrieved 3 times in one query counts as 1 retrieval
        results = [
            RetrievalResult(
                query="q1",
                category="test",
                retrieved_urls=[
                    "https://dup.com",
                    "https://dup.com",
                    "https://dup.com",
                ],
                relevant_urls=[],
                latency_ms=10.0,
                had_error=False,
            ),
        ]
        report = compute_bias(results, top_n=10)
        assert report[0].retrieved_in == 1
