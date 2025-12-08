"""Copilot execution utilities for evaluators.

This module provides utilities for running copilot operations in evaluation contexts,
independent of specific evaluation frameworks like Langfuse.
"""

from typing import List, Optional

import structlog
from pydantic import BaseModel

from rasa.builder.config import COPILOT_HANDLER_ROLLING_BUFFER_SIZE
from rasa.builder.copilot.models import (
    CopilotContext,
    CopilotGenerationContext,
    GeneratedContent,
    ReferenceSection,
    ResponseCategory,
)
from rasa.builder.llm_service import llm_service

structlogger = structlog.get_logger()


class CopilotRunResult(BaseModel):
    """Result from running the copilot with response handler."""

    complete_response: Optional[str]
    response_category: Optional[ResponseCategory]
    reference_section: Optional[ReferenceSection]
    generation_context: CopilotGenerationContext


async def run_copilot_with_response_handler(
    context: CopilotContext,
) -> Optional[CopilotRunResult]:
    """Run the copilot with response handler on the given context.

    This function encapsulates the core copilot execution logic. It handles:
    - Instantiating the copilot and response handler
    - Generating a response and extracting the reference section from the given context
    - Returning structured results

    Args:
        context: The copilot context to process.

    Returns:
        CopilotRunResult containing the complete response, category, and generation
        context, or None if execution fails.

    Raises:
        Any exceptions from the copilot or response handler execution.
    """
    # Instantiate the copilot and response handler
    copilot = llm_service.instantiate_copilot()
    copilot_response_handler = llm_service.instantiate_handler(
        COPILOT_HANDLER_ROLLING_BUFFER_SIZE
    )

    # Call the copilot to generate a response and handle it with the response
    # handler
    (original_stream, generation_context) = await copilot.generate_response(context)
    intercepted_stream = copilot_response_handler.handle_response(original_stream)

    # Exhaust the stream to get the complete response for evaluation
    response_chunks: List[str] = []
    response_category = None
    async for chunk in intercepted_stream:
        if not isinstance(chunk, GeneratedContent):
            continue
        response_chunks.append(chunk.content)
        response_category = chunk.response_category

    complete_response = "".join(response_chunks) if response_chunks else None

    # Extract the reference section from the response handler
    if generation_context.relevant_documents:
        reference_section = copilot_response_handler.extract_references(
            generation_context.relevant_documents
        )
    else:
        reference_section = None

    return CopilotRunResult(
        complete_response=complete_response,
        response_category=response_category,
        reference_section=reference_section,
        generation_context=generation_context,
    )
