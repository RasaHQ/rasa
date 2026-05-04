"""Unit tests for the classifier labeling functions."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.builder.evaluator.configs.models import (
    ClassifierLabelingConfig,
    LabelingConfig,
)
from rasa.builder.evaluator.dataset.data_gen.classifier import label_queries as mod


def _make_config(batch_size: int = 10) -> LabelingConfig:
    return LabelingConfig(
        dataset_name="ds",
        dataset_description="desc",
        output_dir="/tmp/out.jsonl",
        queries_path="/tmp/q.jsonl",
        model="gpt-4o-mini",
        batch_size=batch_size,
        batch_pause_seconds=0.0,
        classifier=ClassifierLabelingConfig(
            valid_categories=["copilot", "error_fallback"]
        ),
    )


@pytest.fixture(autouse=True)
def _stub_prompts():
    with patch.object(mod, "fetch_and_compile_prompt", return_value="prompt"):
        yield


class TestLabelQuery:
    async def test_returns_labeled_query_on_valid_response(self):
        llm = MagicMock()
        with patch.object(
            mod,
            "call_llm",
            new=AsyncMock(
                return_value=json.dumps({"category": "copilot", "confidence": "HIGH"})
            ),
        ):
            result = await mod.label_query(
                "hi", {"copilot", "error_fallback"}, llm, _make_config()
            )
        assert result.category == "copilot"
        assert result.confidence == "high"

    async def test_raises_on_unknown_category(self):
        with patch.object(
            mod,
            "call_llm",
            new=AsyncMock(
                return_value=json.dumps({"category": "bogus", "confidence": "low"})
            ),
        ):
            with pytest.raises(mod.LabelingError, match="Unknown category"):
                await mod.label_query("hi", {"copilot"}, MagicMock(), _make_config())

    async def test_raises_on_llm_error(self):
        with patch.object(
            mod, "call_llm", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            with pytest.raises(mod.LabelingError, match="Label failed"):
                await mod.label_query("hi", {"copilot"}, MagicMock(), _make_config())

    async def test_raises_on_invalid_json(self):
        with patch.object(mod, "call_llm", new=AsyncMock(return_value="not json")):
            with pytest.raises(mod.LabelingError):
                await mod.label_query("hi", {"copilot"}, MagicMock(), _make_config())


class TestLabelQueriesBatch:
    async def test_collects_labeled_and_skipped_and_pauses(self):
        config = _make_config(batch_size=2)
        responses = [
            json.dumps({"category": "copilot", "confidence": "high"}),
            json.dumps({"category": "unknown", "confidence": "low"}),  # → skipped
            json.dumps({"category": "error_fallback", "confidence": "medium"}),
        ]
        sleep_mock = AsyncMock()
        with (
            patch.object(mod, "call_llm", new=AsyncMock(side_effect=responses)),
            patch.object(mod.asyncio, "sleep", new=sleep_mock),
        ):
            labeled, skipped = await mod.label_queries(
                [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}],
                MagicMock(),
                config,
            )

        assert len(labeled) == 2
        assert labeled[0][0]["query"] == "q1"
        assert labeled[1][0]["query"] == "q3"
        assert len(skipped) == 1
        assert skipped[0].query == "q2"
        # Pause triggered after batch_size=2 with more queries remaining
        sleep_mock.assert_awaited_once_with(0.0)

    async def test_raises_when_classifier_config_missing(self):
        config = _make_config()
        config.classifier = None
        with pytest.raises(ValueError, match="classifier"):
            await mod.label_queries([{"query": "q"}], MagicMock(), config)
