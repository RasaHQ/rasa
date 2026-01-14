from functools import wraps
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List

import structlog
from agents import RawResponsesStreamEvent, RunItemStreamEvent, StreamEvent

from rasa.builder.copilot.response_handling.utils import (
    is_function_tool_call_event,
    is_function_tool_output_event,
    is_mcp_tool_call_event,
    is_text_content_part_event,
    is_tool_call_event,
    is_tool_output_event,
)
from rasa.builder.telemetry.langfuse.langfuse_compat import (
    is_langfuse_available,
    langfuse,
)
from rasa.builder.telemetry.langfuse.shared import (
    update_generation_span_with_usage_statistics,
)

if TYPE_CHECKING:
    from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

structlogger = structlog.get_logger()


class AgentCopilotLangfuseTelemetry:
    """Telemetry for agent copilot LLM streaming generation with Langfuse."""

    @staticmethod
    def trace_streaming_generation(
        func: Callable[..., AsyncGenerator[StreamEvent, None]],
    ) -> Callable[..., AsyncGenerator[StreamEvent, None]]:
        """Custom decorator for tracing async streaming of the Copilot's LLM generation.

        This decorator handles Langfuse tracing for async streaming of the Agent
        Copilot's LLM generation by manually managing the generation span and updating
        it with usage statistics after the stream completes.
        """
        if not is_langfuse_available():
            return func

        @wraps(func)
        async def wrapper(
            self: "AgentCopilot", system_prompt: str, messages: List[Dict[str, Any]]
        ) -> AsyncGenerator[StreamEvent, None]:
            langfuse_client = langfuse.get_client()

            with langfuse_client.start_as_current_generation(
                name=f"{self.__class__.__name__}.{func.__name__}",
                input={"system_prompt": system_prompt, "messages": messages},
            ) as generation:
                output: list[dict[str, Any]] = []
                # Call the original streaming function and start capturing the output
                async for stream_event in func(self, system_prompt, messages):
                    parsed_stream_event = AgentCopilotLangfuseTelemetry._parse_stream_event_output_for_tracing(  # noqa: E501
                        stream_event
                    )
                    output.append(parsed_stream_event)
                    yield stream_event

                # Update the span's model parameters and output after streaming is
                # complete
                generation.update(model_parameters=self.llm_config, output=output)

                # Update the span's usage statistics after streaming is complete
                if self.usage_statistics:
                    update_generation_span_with_usage_statistics(
                        generation, self.usage_statistics
                    )

        return wrapper

    @staticmethod
    def _parse_stream_event_output_for_tracing(
        stream_event: StreamEvent,
    ) -> dict[str, Any]:
        """Parse the output of a stream event.

        Extracts the event type and routes to specific extractors for detailed
        information.

        Args:
            stream_event: The stream event.

        Returns:
            Dictionary with stream event information.
        """
        result: dict[str, Any] = (
            AgentCopilotLangfuseTelemetry._parse_event_types_for_tracing(stream_event)
        )

        # If the stream event is part of the text content part stream events
        if is_text_content_part_event(stream_event):
            text_content_part_info = AgentCopilotLangfuseTelemetry._parse_text_content_part_event_for_tracing(  # noqa: E501
                stream_event
            )
            result["event_data"] = text_content_part_info

        # If the stream event is tool call related
        elif is_tool_call_event(stream_event):
            tool_call_info = (
                AgentCopilotLangfuseTelemetry._parse_tool_call_event_for_tracing(
                    stream_event
                )
            )
            result["event_data"] = tool_call_info

        # If the stream event is tool output related
        elif is_tool_output_event(stream_event):
            tool_output_info = (
                AgentCopilotLangfuseTelemetry._parse_tool_output_event_for_tracing(
                    stream_event
                )
            )
            result["event_data"] = tool_output_info

        structlogger.debug(
            "agent_copilot_langfuse_telemetry.parsed_stream_event",
            result=result,
        )

        return result

    @staticmethod
    def _parse_event_types_for_tracing(
        stream_event: StreamEvent,
    ) -> dict[str, str]:
        """Extract event type information from a stream event.

        Args:
            stream_event: The stream event to extract types from.

        Returns:
            Dictionary with event_type, event_data_type, and optionally raw_item_type.
        """
        result: dict[str, str] = {"event_type": stream_event.type}

        # The RawResponsesStreamEvent has a data attribute
        if isinstance(stream_event, RawResponsesStreamEvent):
            result["event_data_type"] = type(stream_event.data).__name__

        # While the RunItemStreamEvent has an item attribute
        elif isinstance(stream_event, RunItemStreamEvent):
            result["event_data_type"] = type(stream_event.item).__name__
            # raw_item_type is the type of the actual payload inside the item. Depending
            # on the the item, the output can be a dict or a Pydantic BaseModel.
            # For TypedDict instances, type(...).__name__ returns "dict", so we use
            # the type field from the dict if available
            if (
                isinstance(stream_event.item.raw_item, dict)
                and "type" in stream_event.item.raw_item
            ):
                result["raw_item_type"] = str(stream_event.item.raw_item.get("type"))
            else:
                result["raw_item_type"] = type(stream_event.item.raw_item).__name__
        return result

    @staticmethod
    def _parse_text_content_part_event_for_tracing(
        event: RawResponsesStreamEvent,
    ) -> dict[str, Any]:
        """Extract information from text content part stream events.

        Args:
            event: The raw responses stream event.

        Returns:
            Dictionary with content event information.
        """
        # ResponseTextDeltaEvent, ResponseContentPartAddedEvent, and
        # ResponseContentPartDoneEvent are all Pydantic BaseModel instances
        try:
            return event.data.model_dump()
        except Exception as e:
            structlogger.error(
                "agent_copilot_langfuse_telemetry.error_parsing_text_content_part_event",
                event_info=(
                    "Error parsing text content part event. " "Returning empty info."
                ),
                error=e,
                event_data=event.data,
            )
            return {}

    @staticmethod
    def _parse_tool_call_event_for_tracing(
        event: RunItemStreamEvent,
    ) -> dict[str, Any]:
        """Extract information from tool call stream events.

        Args:
            event: The run item stream event.

        Returns:
            Dictionary with tool call information, or None if not a tool call
            event.
        """
        raw_item = event.item.raw_item

        if is_function_tool_call_event(event) or is_mcp_tool_call_event(event):
            # ResponseFunctionToolCall and McpCall are Pydantic BaseModel instances
            try:
                return raw_item.model_dump()  # type: ignore[union-attr]
            except Exception as e:
                structlogger.error(
                    "agent_copilot_langfuse_telemetry.error_parsing_tool_call_event",
                    event_info="Error parsing tool call event. Returning empty info.",
                    error=e,
                    event_data=raw_item,
                )

        return {}

    @staticmethod
    def _parse_tool_output_event_for_tracing(
        event: RunItemStreamEvent,
    ) -> dict[str, Any]:
        """Extract information from tool output stream events.

        Args:
            event: The run item stream event.

        Returns:
            Dictionary with tool output information, or None if not a tool output
            event.
        """
        raw_item = event.item.raw_item
        info: dict[str, Any] = {}

        if is_function_tool_output_event(event):
            # FunctionCallOutput is a TypedDict, so raw_item is a dict
            if isinstance(raw_item, dict):
                info.update(raw_item)

        return info
