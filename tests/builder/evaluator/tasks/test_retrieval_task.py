"""Tests for RetrievalTask."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.builder.evaluator.configs.models import (
    ExperimentConfig,
    RetrievalTaskConfig,
)
from rasa.builder.evaluator.tasks.base import RetrievalTaskResult
from rasa.builder.evaluator.tasks.retrieval_task import RetrievalTask

INKEEP_PATH = (
    "rasa.builder.document_retrieval.inkeep_document_retrieval.InKeepDocumentRetrieval"
)


def _make_config(backend: str = "inkeep") -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        description="test",
        dataset_name="test",
        task="retrieval",
        results_dir="/tmp/test",
        formats=["yaml"],
        retrieval=RetrievalTaskConfig(backend=backend),
    )


def _make_item() -> SimpleNamespace:
    return SimpleNamespace(
        id="item-1",
        input={"query": "how to add a slot"},
        expected_output={
            "relevant_pages": [
                {
                    "doc_id": "reference/primitives/slots",
                    "url": "https://rasa.com/docs/reference/primitives/slots",
                    "confidence": "high",
                }
            ]
        },
        metadata={
            "category": "how-to",
            "source": "manual",
            "docs_repo_commit": "abc123",
        },
    )


class TestRetrievalTaskInit:
    def test_builds_inkeep_retriever(self):
        with patch(INKEEP_PATH) as mock_cls:
            task = RetrievalTask(config=_make_config())
            mock_cls.assert_called_once()
            assert task._retriever is mock_cls.return_value

    def test_uses_default_when_retrieval_config_missing(self):
        config = ExperimentConfig(
            name="test",
            description="test",
            dataset_name="test",
            task="retrieval",
            results_dir="/tmp/test",
            formats=["yaml"],
        )
        with patch(INKEEP_PATH) as mock_cls:
            task = RetrievalTask(config=config)
            mock_cls.assert_called_once()
            assert task._retriever is mock_cls.return_value

    def test_raises_on_unknown_backend(self):
        # Directly pass a config with a mocked backend — skip Pydantic Literal check
        config = _make_config()
        with patch.object(
            config, "retrieval", RetrievalTaskConfig.model_construct(backend="invalid")
        ):
            with pytest.raises(ValueError, match="Unknown retrieval backend"):
                RetrievalTask(config=config)


class TestRetrievalTaskRunTask:
    async def test_success_populates_urls_and_latency(self):
        doc = SimpleNamespace(
            url="https://rasa.com/docs/reference/primitives/slots",
            title="Slots",
        )
        mock_retriever = MagicMock()
        mock_retriever.retrieve_documents = AsyncMock(return_value=[doc])

        with patch(INKEEP_PATH, return_value=mock_retriever):
            task = RetrievalTask(config=_make_config())

        result = await task.run_task(item=_make_item())

        assert isinstance(result, RetrievalTaskResult)
        assert result.query == "how to add a slot"
        assert result.retrieved_urls == [
            "https://rasa.com/docs/reference/primitives/slots"
        ]
        assert result.retrieved_titles == ["Slots"]
        assert result.latency_ms > 0
        assert result.error is None

    async def test_retrieval_error_returns_result_with_error_field(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve_documents = AsyncMock(side_effect=RuntimeError("boom"))

        with patch(INKEEP_PATH, return_value=mock_retriever):
            task = RetrievalTask(config=_make_config())

        result = await task.run_task(item=_make_item())

        assert isinstance(result, RetrievalTaskResult)
        assert result.error == "boom"
        assert result.retrieved_urls == []
        assert result.query == "how to add a slot"

    async def test_item_parse_error_returns_none(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve_documents = AsyncMock(return_value=[])

        with patch(INKEEP_PATH, return_value=mock_retriever):
            task = RetrievalTask(config=_make_config())

        bad_item = SimpleNamespace(
            id="bad-item",
            input="not a dict",
            expected_output="not a dict",
            metadata="not a dict",
        )
        result = await task.run_task(item=bad_item)
        assert result is None

    async def test_filters_out_documents_without_url(self):
        docs = [
            SimpleNamespace(url="https://rasa.com/docs/a", title="A"),
            SimpleNamespace(url=None, title="No URL"),
            SimpleNamespace(url="https://rasa.com/docs/b", title="B"),
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve_documents = AsyncMock(return_value=docs)

        with patch(INKEEP_PATH, return_value=mock_retriever):
            task = RetrievalTask(config=_make_config())

        result = await task.run_task(item=_make_item())

        assert result.retrieved_urls == [
            "https://rasa.com/docs/a",
            "https://rasa.com/docs/b",
        ]
        assert len(result.retrieved_titles) == 2
