"""Task that runs a document retriever on a single dataset item.

Instantiates the retriever once (based on ``config.retrieval.backend``) and
reuses it across all items. The retriever contract is simply:
``async def retrieve_documents(query: str) -> List[Document]``.
"""

import time
from typing import Any, Optional

import structlog

from rasa.builder.evaluator.configs.models import (
    ExperimentConfig,
    RetrievalTaskConfig,
)
from rasa.builder.evaluator.dataset.retrieval_models import RetrievalDatasetEntry
from rasa.builder.evaluator.tasks.base import BaseTask, RetrievalTaskResult

structlogger = structlog.get_logger()


class RetrievalTask(BaseTask):
    """Callable task that runs a document retriever on a single dataset item."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        retrieval_config = config.retrieval or RetrievalTaskConfig()
        self._retriever = self._build_retriever(retrieval_config)

    @staticmethod
    def _build_retriever(config: RetrievalTaskConfig) -> Any:
        """Instantiate the retriever based on the configured backend."""
        if config.backend == "inkeep":
            from rasa.builder.document_retrieval.inkeep_document_retrieval import (
                InKeepDocumentRetrieval,
            )

            return InKeepDocumentRetrieval()

        raise ValueError(f"Unknown retrieval backend: {config.backend}")

    async def run_task(
        self,
        *,
        item: Any,
        **_: Any,
    ) -> Optional[RetrievalTaskResult]:
        """Run the retriever on a single dataset item.

        Args:
            item: Langfuse ExperimentItem.

        Returns:
            RetrievalTaskResult. On retrieval failure, the result contains
            ``error`` set and empty ``retrieved_urls``. Returns None only if
            the item itself cannot be parsed.
        """
        try:
            dataset_entry = RetrievalDatasetEntry.from_raw_data(
                id=item.id,
                input_data=item.input,
                expected_output_data=item.expected_output,
                metadata_data=item.metadata,
            )
        except Exception as e:
            structlogger.error(
                "tasks.retrieval_task.item_parse_failed",
                item_id=getattr(item, "id", None),
                error=str(e),
            )
            return None

        query = dataset_entry.input.query

        try:
            start = time.monotonic()
            documents = await self._retriever.retrieve_documents(query)
            latency_ms = (time.monotonic() - start) * 1000

            retrieved_urls = [d.url for d in documents if d.url]
            retrieved_titles = [d.title or "" for d in documents if d.url]

            return RetrievalTaskResult(
                query=query,
                retrieved_urls=retrieved_urls,
                retrieved_titles=retrieved_titles,
                latency_ms=latency_ms,
            )
        except Exception as e:
            structlogger.error(
                "tasks.retrieval_task.retrieval_failed",
                item_id=item.id,
                query=query,
                error=str(e),
            )
            return RetrievalTaskResult(
                query=query,
                retrieved_urls=[],
                retrieved_titles=[],
                latency_ms=0.0,
                error=str(e),
            )
