"""String processing utilities for copilot response handling."""

from typing import List, Optional, Set, Tuple

import structlog
from agents import RawResponsesStreamEvent, RunItemStreamEvent, StreamEvent
from agents.items import ToolCallItem, ToolCallOutputItem
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_item import McpCall

from rasa.builder.copilot.models import GeneratedContent, ResponseCategory
from rasa.builder.copilot.response_handling.constants import (
    LLM_PREFIXES_TO_SUFFIX_REMOVE,
)

structlogger = structlog.get_logger()


def remove_prefix(content: str) -> Tuple[str, Optional[str]]:
    """Process the initial content from the buffer, handling prefix removal.

    Args:
        content: The content to process.

    Returns:
        A tuple of (processed_content, prefix_found). prefix_found is None if no prefix
        was found, otherwise it's the prefix string that was removed.
    """
    # Check if content starts with any of the known prefixes
    for prefix in LLM_PREFIXES_TO_SUFFIX_REMOVE.keys():
        if content.startswith(prefix):
            structlogger.debug(
                "response_handling.utils.remove_prefix.prefix_detected",
                prefix=prefix,
                content_length=len(content),
            )
            return content[len(prefix) :], prefix

    return content, None


def remove_suffix(content: str) -> Tuple[str, Optional[str]]:
    """Process the rolling buffer content, handling suffix removal.

    Args:
        content: The content to process.

    Returns:
        A tuple of (processed_content, suffix_found). suffix_found is None if no suffix
        was found, otherwise it's the suffix string that was removed.
    """
    # Check if content ends with any of the known suffixes
    for suffix in LLM_PREFIXES_TO_SUFFIX_REMOVE.values():
        if content.endswith(suffix):
            structlogger.debug(
                "response_handling.utils.remove_suffix.suffix_detected",
                suffix=suffix,
                content_length=len(content),
            )
            return content[: -len(suffix)], suffix

    return content, None


def remove_prefix_and_suffix(content: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Remove the prefix and suffix from the content.

    Args:
        content: The content to process.

    Returns:
        A tuple of (processed_content, prefix_found, suffix_found). prefix_found and
        suffix_found are None if not found, otherwise they contain the removed strings.
    """
    processed_content, prefix = remove_prefix(content)
    processed_content, suffix = remove_suffix(processed_content)
    return processed_content, prefix, suffix


def extract_text_by_categories(
    responses: List[GeneratedContent],
    primary_categories: Set[ResponseCategory],
    controlled_prediction_categories: Set[ResponseCategory],
) -> str:
    """Extract and join all content from generated responses matching given categories.

    Combines primary categories with controlled prediction categories and extracts
    text content from matching responses.

    Args:
        responses: List of generated content responses to filter.
        primary_categories: Set of primary response categories for the handler.
        controlled_prediction_categories: Set of controlled prediction categories.

    Returns:
        Concatenated text from all responses matching the combined categories.
    """
    content_parts: List[str] = []
    all_text_categories = primary_categories | controlled_prediction_categories

    for response in responses or []:
        if isinstance(response, GeneratedContent):
            if response.response_category in all_text_categories:
                content_parts.append(response.content)
    return "".join(content_parts)


def is_text_content_part_start_event(event: StreamEvent) -> bool:
    """Check if this event indicates the start of a Copilot's textual content part.

    This checks if the event is a ResponseContentPartAddedEvent and the added
    part is a ResponseOutputText instance, which explicitly marks when a new
    text content part begins in the main Copilot reply.

    Args:
        event: The stream event to check.

    Returns:
        True if the event is a content_part.added event with a text output part,
        False otherwise.
    """
    if not isinstance(event, RawResponsesStreamEvent):
        return False

    if not isinstance(event.data, ResponseContentPartAddedEvent):
        return False

    return isinstance(event.data.part, ResponseOutputText)


def is_text_content_part_end_event(event: StreamEvent) -> bool:
    """Check if this event indicates the end of a Copilot's textual content part.

    This checks if the event is a ResponseContentPartDoneEvent and the done
    part is a ResponseOutputText instance, which explicitly marks when a text
    content part has finished in the main Copilot reply.

    Args:
        event: The stream event to check.

    Returns:
        True if the event is a content_part.done event with a text output part,
        False otherwise.
    """
    if not isinstance(event, RawResponsesStreamEvent):
        return False
    if not isinstance(event.data, ResponseContentPartDoneEvent):
        return False
    return isinstance(event.data.part, ResponseOutputText)


def is_text_content_part_delta_event(event: StreamEvent) -> bool:
    """Check if this event indicates the delta of a content part.

    Args:
        event: The stream event to check.

    Returns:
        True if the event is a content_part.delta event, False otherwise.
    """
    if not isinstance(event, RawResponsesStreamEvent):
        return False
    return isinstance(event.data, ResponseTextDeltaEvent)


def is_text_content_part_event(event: StreamEvent) -> bool:
    """Check if this event indicates a text content part.

    Args:
        event: The stream event to check.

    Returns:
        True if the event is a text content part event, False otherwise.
    """
    return any(
        [
            is_text_content_part_start_event(event),
            is_text_content_part_end_event(event),
            is_text_content_part_delta_event(event),
        ]
    )


def is_content_part_end_event(event: StreamEvent) -> bool:
    """Check if this event indicates the end of a content part.

    Args:
        event: The stream event to check.

    Returns:
        True if the event is a content_part.done event, False otherwise.
    """
    if not isinstance(event, RawResponsesStreamEvent):
        return False
    return isinstance(event.data, ResponseContentPartDoneEvent)


def extract_text_content_from_events(events: List[StreamEvent]) -> str:
    """Extract the text content from a list of stream events.

    This function extracts text deltas from multiple stream events and concatenates
    them into a single string.

    Args:
        events: List of stream events to extract text content from.

    Returns:
        Concatenated text content from all text delta events in the list.
    """
    text_parts: List[str] = []
    for event in events:
        if is_text_content_part_delta_event(event):
            text_parts.append(event.data.delta)  # type: ignore[union-attr]
    return "".join(text_parts)


def is_tool_call_event(event: StreamEvent) -> bool:
    """Check if this event indicates a tool call.

    This checks if the event is a RunItemStreamEvent and the item is a
    ToolCallItem, which marks when a tool call occurs.

    Args:
        event: The stream event to check.

    Returns:
        True if the event is a tool call event, False otherwise.
    """
    if not isinstance(event, RunItemStreamEvent):
        return False

    return isinstance(event.item, ToolCallItem)


def is_tool_output_event(event: StreamEvent) -> bool:
    """Check if this event indicates a tool output.

    This checks if the event is a RunItemStreamEvent and the item is a
    ToolCallOutputItem, which marks when a tool execution completes and returns output.

    Args:
        event: The stream event to check.

    Returns:
        True if the event is a tool output event, False otherwise.
    """
    if not isinstance(event, RunItemStreamEvent):
        return False

    return isinstance(event.item, ToolCallOutputItem)


def is_function_tool_call_event(event: StreamEvent) -> bool:
    """Check if this event indicates a function tool call.

    This checks if the event is a tool call event and the raw item is a
    ResponseFunctionToolCall instance.

    Args:
        event: The stream event to check.

    Returns:
        True if the event is a function tool call event, False otherwise.
    """
    if not is_tool_call_event(event):
        return False

    # We know from is_tool_call_content_part_event that event is RunItemStreamEvent
    # and event.item is ToolCallItem with a raw_item attribute
    return isinstance(event.item.raw_item, ResponseFunctionToolCall)  # type: ignore[union-attr]


def is_mcp_tool_call_event(event: StreamEvent) -> bool:
    """Check if this event indicates an MCP tool call.

    This checks if the event is a tool call event and the raw item is a
    McpCall instance.

    Args:
        event: The stream event to check.

    Returns:
        True if the event is an MCP tool call event, False otherwise.
    """
    if not is_tool_call_event(event):
        return False

    # We know from is_tool_call_event that event is RunItemStreamEvent
    # and event.item is ToolCallItem with a raw_item attribute
    return isinstance(event.item.raw_item, McpCall)  # type: ignore[union-attr]


def is_function_tool_output_event(event: StreamEvent) -> bool:
    """Check if this event indicates a function tool output.

    This checks if the event is a tool output event and the raw item is a
    FunctionCallOutput instance.

    This covers both regular function tool calls and MCP tool calls, as they both use
    FunctionCallOutput.

    Args:
        event: The stream event to check.

    Returns:
        True if the event is a function tool output event, False otherwise.
    """
    if not is_tool_output_event(event):
        return False

    # We know from is_tool_output_event that event is RunItemStreamEvent
    # and event.item is ToolCallOutputItem with a raw_item attribute
    raw_item = event.item.raw_item  # type: ignore[union-attr]

    # FunctionCallOutput is a TypedDict, so we check for the type field instead.
    # "function_call_output" is the Literal type value from FunctionCallOutput
    # TypedDict definition. It's not exported as a constant by the OpenAI SDK.
    return isinstance(raw_item, dict) and raw_item.get("type") == "function_call_output"


def is_response_completed_event(event: StreamEvent) -> bool:
    """Check if this event indicates a completed response.

    This checks if the event is a RawResponsesStreamEvent and the data is a
    ResponseCompletedEvent, which contains the final response with usage statistics.

    Args:
        event: The stream event to check.

    Returns:
        True if the event is a response.completed event, False otherwise.
    """
    if not isinstance(event, RawResponsesStreamEvent):
        return False

    return isinstance(event.data, ResponseCompletedEvent)
