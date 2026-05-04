"""Unit tests for the retrieval two-stage labeling functions."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.builder.evaluator.configs.models import (
    LabelingConfig,
    RetrievalLabelingConfig,
)
from rasa.builder.evaluator.dataset.data_gen.retrieval import label_pages as mod
from rasa.builder.evaluator.dataset.data_gen.retrieval.docs_index import (
    DocIndex,
    DocPage,
)


def _make_config(batch_size: int = 10) -> LabelingConfig:
    return LabelingConfig(
        dataset_name="ds",
        dataset_description="desc",
        output_dir="/tmp/out.jsonl",
        queries_path="/tmp/q.jsonl",
        model="gpt-4o-mini",
        batch_size=batch_size,
        batch_pause_seconds=0.0,
        retrieval=RetrievalLabelingConfig(docs_repo_path="/tmp/repo"),
    )


def _make_index() -> DocIndex:
    return DocIndex(
        pages=[
            DocPage(doc_id="a", url="https://x/a", title="A", content="content A"),
            DocPage(doc_id="b", url="https://x/b", title="B", content="content B"),
        ],
        docs_repo_commit="sha",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture(autouse=True)
def _stub_prompts():
    with patch.object(mod, "fetch_and_compile_prompt", return_value="prompt"):
        yield


class TestShortlistPages:
    async def test_returns_only_known_doc_ids(self):
        index = _make_index()
        with patch.object(
            mod,
            "call_llm",
            new=AsyncMock(return_value=json.dumps({"doc_ids": ["a", "ghost", "b"]})),
        ):
            result = await mod.shortlist_pages("q", index, MagicMock(), _make_config())
        assert result == ["a", "b"]

    async def test_raises_on_llm_error(self):
        with patch.object(
            mod, "call_llm", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            with pytest.raises(mod.LabelingError, match="Shortlist failed"):
                await mod.shortlist_pages(
                    "q", _make_index(), MagicMock(), _make_config()
                )

    async def test_raises_on_invalid_schema(self):
        with patch.object(
            mod, "call_llm", new=AsyncMock(return_value=json.dumps({"wrong": "shape"}))
        ):
            with pytest.raises(mod.LabelingError):
                await mod.shortlist_pages(
                    "q", _make_index(), MagicMock(), _make_config()
                )


class TestConfirmRelevance:
    async def test_returns_relevant_page_when_relevant(self):
        page = DocPage(doc_id="a", url="https://x/a", title="A", content="C")
        with patch.object(
            mod,
            "call_llm",
            new=AsyncMock(
                return_value=json.dumps({"relevant": True, "confidence": "HIGH"})
            ),
        ):
            result = await mod.confirm_relevance("q", page, MagicMock(), _make_config())
        assert result is not None
        assert result.doc_id == "a"
        assert result.confidence == "high"

    async def test_returns_none_when_not_relevant(self):
        page = DocPage(doc_id="a", url="https://x/a", title="A", content="C")
        with patch.object(
            mod,
            "call_llm",
            new=AsyncMock(
                return_value=json.dumps({"relevant": False, "confidence": "low"})
            ),
        ):
            result = await mod.confirm_relevance("q", page, MagicMock(), _make_config())
        assert result is None

    async def test_raises_on_llm_error(self):
        page = DocPage(doc_id="a", url="https://x/a", title="A", content="C")
        with patch.object(
            mod, "call_llm", new=AsyncMock(side_effect=RuntimeError("x"))
        ):
            with pytest.raises(mod.LabelingError, match="Confirm failed"):
                await mod.confirm_relevance("q", page, MagicMock(), _make_config())


class TestLabelQueriesBatch:
    async def test_two_stage_collects_relevant_and_pauses(self):
        config = _make_config(batch_size=2)
        index = _make_index()
        sleep_mock = AsyncMock()

        async def fake_shortlist(query, doc_index, llm, cfg):
            return ["a", "b"]

        async def fake_confirm(query, page, llm, cfg):
            from rasa.builder.evaluator.dataset.retrieval_models import RelevantPage

            if page.doc_id == "a":
                return RelevantPage(doc_id="a", url=page.url, confidence="high")
            return None

        with (
            patch.object(mod, "shortlist_pages", side_effect=fake_shortlist),
            patch.object(mod, "confirm_relevance", side_effect=fake_confirm),
            patch.object(mod.asyncio, "sleep", new=sleep_mock),
        ):
            labeled, skipped = await mod.label_queries(
                [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}],
                index,
                MagicMock(),
                config,
            )

        assert len(labeled) == 3
        assert [len(pages) for _, pages in labeled] == [1, 1, 1]
        assert labeled[0][1][0].doc_id == "a"
        assert skipped == []
        sleep_mock.assert_awaited_once_with(0.0)

    async def test_skips_query_when_shortlist_fails(self):
        config = _make_config()
        index = _make_index()

        async def fake_shortlist(query, doc_index, llm, cfg):
            raise mod.LabelingError("nope")

        with patch.object(mod, "shortlist_pages", side_effect=fake_shortlist):
            labeled, skipped = await mod.label_queries(
                [{"query": "q1"}], index, MagicMock(), config
            )

        assert labeled == []
        assert len(skipped) == 1
        assert skipped[0].query == "q1"
        assert "nope" in skipped[0].reason
