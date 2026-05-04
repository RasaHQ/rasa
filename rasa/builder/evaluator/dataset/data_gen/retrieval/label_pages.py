"""Two-stage LLM labeling for retrieval evaluation ground truth.

Stage 1 (shortlist): Feed all page titles to the LLM, get 5-10 candidate doc_ids.
Stage 2 (confirm):   For each candidate, feed full page content, get relevance judgment.
"""

import asyncio
import json
from typing import Optional

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from rasa.builder.evaluator.configs.models import LabelingConfig
from rasa.builder.evaluator.dataset.data_gen.retrieval.constants import (
    LABELING_CONFIRM_SYSTEM_PROMPT,
    LABELING_CONFIRM_USER_PROMPT,
    LABELING_SHORTLIST_SYSTEM_PROMPT,
    LABELING_SHORTLIST_USER_PROMPT,
)
from rasa.builder.evaluator.dataset.data_gen.retrieval.docs_index import (
    DocIndex,
    DocPage,
)
from rasa.builder.evaluator.dataset.retrieval_models import RelevantPage
from rasa.builder.evaluator.helpers import call_llm, fetch_and_compile_prompt

structlogger = structlog.get_logger()


class ShortlistResponse(BaseModel):
    """Expected JSON output from the shortlist LLM call."""

    doc_ids: list[str] = Field(
        description="List of 5-10 candidate doc_ids most likely to answer the query."
    )


class RelevanceJudgment(BaseModel):
    """Expected JSON output from the confirmation LLM call."""

    relevant: bool = Field(description="Whether the document is relevant to the query.")
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'.")


class SkippedQuery(BaseModel):
    """Record of a query that was skipped during labeling."""

    query: str
    reason: str


class LabelingError(Exception):
    """Raised when LLM labeling fails after retries."""


async def shortlist_pages(
    query: str,
    doc_index: DocIndex,
    llm_client: AsyncOpenAI,
    config: LabelingConfig,
) -> list[str]:
    """Stage 1: identify candidate doc_ids using page titles only.

    Args:
        query: The search query to label.
        doc_index: Full documentation index.
        llm_client: OpenAI async client.
        config: Retrieval labeling run knobs.

    Returns:
        List of candidate doc_id strings.

    Raises:
        LabelingError: If the LLM call fails after retries.
    """
    system_prompt = fetch_and_compile_prompt(LABELING_SHORTLIST_SYSTEM_PROMPT)
    titles = "\n".join(f'- {page.doc_id}: "{page.title}"' for page in doc_index.pages)
    user_prompt = fetch_and_compile_prompt(
        LABELING_SHORTLIST_USER_PROMPT, titles=titles, query=query
    )

    try:
        content = await call_llm(
            llm_client=llm_client,
            model=config.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=config.temperature,
            max_retries=config.max_retries,
            log_event="label_pages.shortlist",
        )
        parsed = ShortlistResponse.model_validate(json.loads(content))
    except Exception as e:
        raise LabelingError(f"Shortlist failed for query '{query}': {e}") from e

    valid_ids = {page.doc_id for page in doc_index.pages}
    result = []
    for doc_id in parsed.doc_ids:
        if doc_id in valid_ids:
            result.append(doc_id)
        else:
            structlogger.warning(
                "label_pages.shortlist.unknown_doc_id",
                doc_id=doc_id,
                query=query,
            )
    return result


async def confirm_relevance(
    query: str,
    page: DocPage,
    llm_client: AsyncOpenAI,
    config: LabelingConfig,
) -> Optional[RelevantPage]:
    """Stage 2: confirm a single page's relevance using full content.

    Args:
        query: The search query.
        page: The documentation page to evaluate.
        llm_client: OpenAI async client.
        config: Retrieval labeling run knobs.

    Returns:
        RelevantPage if the page is relevant, None otherwise.

    Raises:
        LabelingError: If the LLM call fails after retries.
    """
    page_content = page.content[: config.max_page_content_chars]
    system_prompt = fetch_and_compile_prompt(LABELING_CONFIRM_SYSTEM_PROMPT)
    user_prompt = fetch_and_compile_prompt(
        LABELING_CONFIRM_USER_PROMPT,
        query=query,
        title=page.title,
        doc_id=page.doc_id,
        content=page_content,
    )

    try:
        content = await call_llm(
            llm_client=llm_client,
            model=config.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=config.temperature,
            max_retries=config.max_retries,
            log_event="label_pages.confirm",
        )
        judgment = RelevanceJudgment.model_validate(json.loads(content))
    except Exception as e:
        raise LabelingError(
            f"Confirm failed for query '{query}', page '{page.doc_id}': {e}"
        ) from e

    if not judgment.relevant:
        return None

    return RelevantPage(
        doc_id=page.doc_id,
        url=page.url,
        confidence=judgment.confidence.lower(),
    )


async def label_queries(
    queries: list[dict[str, str]],
    doc_index: DocIndex,
    llm_client: AsyncOpenAI,
    config: LabelingConfig,
) -> tuple[list[tuple[dict[str, str], list[RelevantPage]]], list[SkippedQuery]]:
    """Label a batch of queries, collecting results and skipped queries.

    Pauses for ``config.batch_pause_seconds`` after every ``config.batch_size``
    queries to avoid rate limiting.

    Args:
        queries: List of dicts with at least ``query`` and ``category`` keys.
        doc_index: Full documentation index.
        llm_client: OpenAI async client.
        config: Retrieval labeling run knobs.

    Returns:
        Tuple of:
        - List of (query_dict, relevant_pages) for successfully labeled queries.
        - List of SkippedQuery for queries that failed.
    """
    labeled: list[tuple[dict[str, str], list[RelevantPage]]] = []
    skipped: list[SkippedQuery] = []

    for i, query_entry in enumerate(queries):
        query_text = query_entry["query"]
        structlogger.info(
            "label_pages.label_queries.progress",
            current=i + 1,
            total=len(queries),
            query=query_text,
        )

        try:
            # Build a lookup for fast page retrieval
            pages_by_id = {page.doc_id: page for page in doc_index.pages}

            # Stage 1: shortlist
            candidate_ids = await shortlist_pages(
                query_text, doc_index, llm_client, config
            )

            # Stage 2: confirm each candidate
            relevant_pages: list[RelevantPage] = []
            for doc_id in candidate_ids:
                page = pages_by_id.get(doc_id)
                if page is None:
                    continue

                result = await confirm_relevance(query_text, page, llm_client, config)
                if result is not None:
                    relevant_pages.append(result)

            labeled.append((query_entry, relevant_pages))
        except LabelingError as e:
            structlogger.warning(
                "label_pages.label_queries.skipped",
                query=query_text,
                reason=str(e),
            )
            skipped.append(SkippedQuery(query=query_text, reason=str(e)))

        # Pause after every batch to respect rate limits
        if (i + 1) % config.batch_size == 0 and i < len(queries) - 1:
            structlogger.info(
                "label_pages.label_queries.batch_pause",
                completed=i + 1,
                pause_seconds=config.batch_pause_seconds,
            )
            await asyncio.sleep(config.batch_pause_seconds)

    structlogger.info(
        "label_pages.label_queries.summary",
        total=len(queries),
        labeled=len(labeled),
        skipped=len(skipped),
    )

    return labeled, skipped
