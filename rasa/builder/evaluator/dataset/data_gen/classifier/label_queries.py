"""Single-stage LLM labeling for classifier evaluation ground truth.

For each query, ask the LLM judge to assign a single ``ResponseCategory`` value.
"""

import asyncio
import json

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from rasa.builder.evaluator.configs.models import LabelingConfig
from rasa.builder.evaluator.dataset.data_gen.classifier.constants import (
    LABELING_SYSTEM_PROMPT,
    LABELING_USER_PROMPT,
)
from rasa.builder.evaluator.helpers import call_llm, fetch_and_compile_prompt

structlogger = structlog.get_logger()


class LabelJudgment(BaseModel):
    """Expected JSON output from the labeling LLM call."""

    category: str = Field(description="A ResponseCategory value assigned to the query.")
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'.")


class LabeledQuery(BaseModel):
    """A successfully labeled query."""

    category: str
    confidence: str


class SkippedQuery(BaseModel):
    """Record of a query that was skipped during labeling."""

    query: str
    reason: str


class LabelingError(Exception):
    """Raised when LLM labeling fails after retries."""


async def label_query(
    query: str,
    valid_categories: set[str],
    llm_client: AsyncOpenAI,
    config: LabelingConfig,
) -> LabeledQuery:
    """Assign a single ResponseCategory label to a query.

    Args:
        query: The user-input message to label.
        valid_categories: Allowed ResponseCategory values.
        llm_client: OpenAI async client.
        config: Labeling run knobs.

    Returns:
        LabeledQuery with category + confidence.

    Raises:
        LabelingError: If the LLM call fails after retries or returns
            an unknown category.
    """
    categories_str = "\n".join(f"- {c}" for c in sorted(valid_categories))
    system_prompt = fetch_and_compile_prompt(LABELING_SYSTEM_PROMPT)
    user_prompt = fetch_and_compile_prompt(
        LABELING_USER_PROMPT, query=query, categories=categories_str
    )

    try:
        content = await call_llm(
            llm_client=llm_client,
            model=config.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=config.temperature,
            max_retries=config.max_retries,
            log_event="label_queries.label",
        )
        judgment = LabelJudgment.model_validate(json.loads(content))
    except Exception as e:
        raise LabelingError(f"Label failed for query '{query}': {e}") from e

    if judgment.category not in valid_categories:
        raise LabelingError(
            f"Unknown category '{judgment.category}' for query '{query}'."
        )

    return LabeledQuery(
        category=judgment.category,
        confidence=judgment.confidence.lower(),
    )


async def label_queries(
    queries: list[dict[str, str]],
    llm_client: AsyncOpenAI,
    config: LabelingConfig,
) -> tuple[list[tuple[dict[str, str], LabeledQuery]], list[SkippedQuery]]:
    """Label a batch of queries, collecting results and skipped queries.

    Pauses for ``config.batch_pause_seconds`` after every ``config.batch_size``
    queries to avoid rate limiting.

    Args:
        queries: List of dicts with at least a ``query`` key.
        llm_client: OpenAI async client.
        config: Labeling run knobs.

    Returns:
        Tuple of:
        - List of (query_dict, LabeledQuery) for successfully labeled queries.
        - List of SkippedQuery for queries that failed.
    """
    if config.classifier is None:
        raise ValueError(
            "LabelingConfig.classifier is required for classifier labeling."
        )
    valid_categories = set(config.classifier.valid_categories)

    labeled: list[tuple[dict[str, str], LabeledQuery]] = []
    skipped: list[SkippedQuery] = []

    for i, query_entry in enumerate(queries):
        query_text = query_entry["query"]
        structlogger.info(
            "label_queries.progress",
            current=i + 1,
            total=len(queries),
            query=query_text,
        )

        try:
            result = await label_query(query_text, valid_categories, llm_client, config)
            labeled.append((query_entry, result))
        except LabelingError as e:
            structlogger.warning(
                "label_queries.skipped",
                query=query_text,
                reason=str(e),
            )
            skipped.append(SkippedQuery(query=query_text, reason=str(e)))

        if (i + 1) % config.batch_size == 0 and i < len(queries) - 1:
            structlogger.info(
                "label_queries.batch_pause",
                completed=i + 1,
                pause_seconds=config.batch_pause_seconds,
            )
            await asyncio.sleep(config.batch_pause_seconds)

    structlogger.info(
        "label_queries.summary",
        total=len(queries),
        labeled=len(labeled),
        skipped=len(skipped),
    )

    return labeled, skipped
