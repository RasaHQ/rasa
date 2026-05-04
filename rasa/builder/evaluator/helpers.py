import os
import sys
from typing import Any, Optional

import structlog
from openai import AsyncOpenAI

from rasa.builder.evaluator.tasks.base import AvailableTasks

structlogger = structlog.get_logger()


REQUIRED_ENV_VARS = ["OPENAI_API_KEY"]
RETRIEVAL_ENV_VARS = ["INKEEP_API_KEY"]
LANGFUSE_ENV_VARS = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]


def validate_env(push_langfuse: bool, task: Optional[AvailableTasks] = None) -> None:
    """Check that required env vars are set."""
    required = list(REQUIRED_ENV_VARS)
    if push_langfuse:
        required.extend(LANGFUSE_ENV_VARS)

    if task == AvailableTasks.RETRIEVAL:
        required.extend(REQUIRED_ENV_VARS)

    missing = [var for var in required if not os.getenv(var)]
    if missing:
        structlogger.error(
            "build_dataset.missing_env_vars",
            missing=missing,
        )
        sys.exit(1)


async def call_llm(
    *,
    llm_client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_retries: int = 0,
    log_event: str = "call_llm",
) -> str:
    """Call the LLM with a JSON-object response format and return raw content.

    Retries on errors and empty responses. Callers are responsible for parsing
    and validating the returned content.

    Args:
        llm_client: OpenAI async client.
        model: Model identifier.
        system_prompt: System message content.
        user_prompt: User message content.
        temperature: Sampling temperature.
        max_retries: Number of retries after the initial attempt.
        log_event: Prefix used for retry warning log events.

    Returns:
        Raw JSON string content returned by the LLM.

    Raises:
        Exception: the last underlying error if all retries are exhausted.
    """
    for attempt in range(1 + max_retries):
        try:
            response = await llm_client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM.")
            return content
        except Exception as e:
            if attempt >= max_retries:
                raise
            structlogger.warning(
                f"{log_event}.retry",
                attempt=attempt + 1,
                error=str(e),
            )

    raise RuntimeError("call_llm: unreachable")


def fetch_and_compile_prompt(prompt_name: str, **kwargs: Any) -> str:
    """Fetch a prompt template from Langfuse and optionally compile it.

    Args:
        prompt_name: Name of the prompt to fetch from Langfuse.
        **kwargs: Variables to render into the prompt template.

    Returns:
        The prompt string, compiled with kwargs if any were provided.
    """
    from rasa.builder.telemetry.langfuse_integration.langfuse_compat import langfuse

    lf_client = langfuse.get_client()

    prompt = lf_client.get_prompt(prompt_name, label="latest")
    if kwargs:
        return prompt.compile(**kwargs)

    return prompt.prompt


def push_to_langfuse(
    entries: list[Any],
    dataset_name: str,
    dataset_description: Optional[str] = None,
) -> None:
    """Create or reuse a Langfuse dataset and upsert all items."""
    from rasa.builder.telemetry.langfuse_integration.langfuse_compat import langfuse

    client = langfuse.get_client()

    client.create_dataset(
        name=dataset_name,
        description=dataset_description,
    )
    structlogger.info(
        "build_dataset.langfuse.dataset_ready",
        dataset_name=dataset_name,
    )

    for entry in entries:
        client.create_dataset_item(
            dataset_name=dataset_name,
            **entry.to_langfuse_item_kwargs(),
        )

    client.flush()
    structlogger.info(
        "build_dataset.langfuse.pushed",
        dataset_name=dataset_name,
        item_count=len(entries),
    )
