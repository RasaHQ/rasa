import asyncio
import copy
import json
from collections import defaultdict, deque
from contextvars import Token
from typing import AsyncGenerator, Deque, Dict, List, Optional, Tuple, Union

import structlog
from agents import StreamEvent

from rasa.builder.copilot.agent_sdk.planning_context import (
    get_final_plan,
    init_planning_context,
    reset_planning_context,
)
from rasa.builder.copilot.agent_sdk.planning_tools import (
    reset_plan_queue,
    set_plan_queue,
)
from rasa.builder.copilot.exceptions import (
    CopilotFinalBufferReached,
    CopilotStreamEndedEarly,
)
from rasa.builder.copilot.mcp_server.models import DocumentSearchResponse
from rasa.builder.copilot.models import (
    CopilotOutput,
    CopilotTextContent,
    CopilotTextEndContent,
    CopilotTextStartContent,
    ExceptionContent,
    GeneratedContent,
    MCPToolCall,
    ResponseCategory,
    ResponseCompleteness,
    TodoItem,
    TodoPlanUpdate,
)
from rasa.builder.copilot.response_handling.base_copilot_response_handler import (
    BaseCopilotResponseHandler,
)
from rasa.builder.copilot.response_handling.constants import (
    AGENT_PRIMARY_TEXT_CATEGORIES,
    CONTROLLED_PREDICTION_CATEGORIES,
    EXCEPTION_RESPONSE,
    PREDICTION_RESPONSES,
)
from rasa.builder.copilot.response_handling.utils import (
    extract_text_by_categories,
    extract_text_content_from_events,
    is_content_part_end_event,
    is_document_retrieval_mcp_tool_output_event,
    is_text_content_part_delta_event,
    is_text_content_part_start_event,
    remove_prefix,
    remove_prefix_and_suffix,
    remove_suffix,
)
from rasa.builder.document_retrieval.models import Document

structlogger = structlog.get_logger()


class AgentCopilotResponseHandler(BaseCopilotResponseHandler):
    """Handles the controlled responses from Copilot.

    This handler manages two types of data:
    - llm_streamed_events: A dictionary mapping content part indices to lists of
      stream events from the LLM during processing.
    - generated_responses: A list of CopilotOutput objects representing cleaned
      and processed responses.

    Parameters:
        rolling_buffer_size: Size of the rolling buffer for prefix/suffix handling.
    """

    def __init__(
        self,
        response_stream: AsyncGenerator[StreamEvent, None],
        rolling_buffer_size: int = 20,
        mcp_tool_queue: Optional[asyncio.Queue[MCPToolCall]] = None,
        plan_queue: Optional[asyncio.Queue[TodoPlanUpdate]] = None,
    ):
        self._rolling_buffer_size = rolling_buffer_size
        self._response_stream = response_stream

        # Queues for MCP tool calls and planning updates.
        # These queues are updated by the Copilot class, and since Python passes
        # objects by reference, any updates made in the Copilot class will be
        # reflected here in the response handler.
        self._mcp_tool_queue = mcp_tool_queue
        self._plan_queue = plan_queue

        # Rolling buffer that allows look-ahead for handling special tokens and
        # prefix/suffix removal.
        self._rolling_buffer: Deque[StreamEvent] = deque(
            maxlen=self._rolling_buffer_size
        )

        # Storage for the LLM stream events split by content parts, filled by the
        # rolling buffer.
        # Key: content part index; Value: list of stream events.
        self._llm_streamed_events: defaultdict[int, List[StreamEvent]] = defaultdict(
            list
        )

        # Current content part index (incremented for each new content part)
        self._current_content_part_index: int = 0

        # A list of cleaned and generated responses.
        self._generated_responses: List[GeneratedContent] = []

        # A list of retrieved documents from document retrieval's tool call
        self._retrieved_documents: List[Document] = []

        # Tokens for planning context cleanup
        self._planning_token: Optional[Token] = None
        self._plan_queue_token: Optional[
            Token[Optional[asyncio.Queue[TodoPlanUpdate]]]
        ] = None

        # Maximum number of tokens to check for special responses (e.g. roleplay,
        # out-of-scope, etc.).
        self._max_expected_special_response_tokens: int = 20

        # Prefix/suffix tracking state
        self._prefix_found: Optional[str] = None
        self._suffix_found: Optional[str] = None

        # Task planning - captures the latest plan state during streaming
        # This is needed because the planning context is reset after streaming ends
        self._final_plan: Optional[List[TodoItem]] = None

    @property
    def generated_responses(self) -> List[GeneratedContent]:
        return copy.deepcopy(self._generated_responses)

    @property
    def generated_responses_count(self) -> int:
        return len(self._generated_responses)

    @property
    def raw_llm_stream_data(self) -> Dict[int, List[StreamEvent]]:
        return copy.deepcopy(self._llm_streamed_events)

    @property
    def raw_llm_stream_text_content(self) -> str:
        content_part_text_content: List[str] = []
        content_part_separator = "\n"
        for content_part_index in sorted(self._llm_streamed_events.keys()):
            events = self._llm_streamed_events[content_part_index]
            text = extract_text_content_from_events(events)
            content_part_text_content.append(text)
        return content_part_separator.join(content_part_text_content)

    @property
    def raw_llm_stream_item_count(self) -> int:
        """Get the total number of stream events across all content parts.

        Returns:
            Total number of stream events across all content parts.
        """
        total = 0
        for buffer in self._llm_streamed_events.values():
            total += len(buffer)
        return total

    @property
    def retrieved_documents(self) -> List[Document]:
        return copy.deepcopy(self._retrieved_documents)

    def reset(self) -> None:
        """Clear all buffers and reset the handler.

        This method also cleans up any planning context state from the previous
        stream by resetting the ContextVar tokens if they exist.
        """
        # Clean up planning context state from previous stream
        if self._planning_token is not None:
            reset_planning_context(self._planning_token)
            self._planning_token = None
        if self._plan_queue_token is not None:
            reset_plan_queue(self._plan_queue_token)
            self._plan_queue_token = None

        # Clear buffers and state
        self._rolling_buffer.clear()
        self._llm_streamed_events.clear()
        self._current_content_part_index = 0
        self._generated_responses.clear()
        self._prefix_found = None
        self._suffix_found = None
        self._final_plan = None

        # Clear the retrieved documents list
        self._retrieved_documents.clear()

    def _drain_queues_without_yielding(self, capture_final_plan: bool = False) -> None:
        """Drain MCP tool and plan queues without yielding events.

        This method clears any remaining events from the queues to prevent
        leakage to subsequent requests. The queues are instance variables on
        AgentCopilot and persist across requests, so they must be cleared
        even when an exception occurs during streaming.

        Args:
            capture_final_plan: If True, update _final_plan with the last plan
                event from the queue before discarding. This ensures the plan
                is persisted even when an exception occurs.

        Note: This discards events rather than yielding them, which is
        appropriate for cleanup after exceptions where we don't want to
        send potentially incomplete/stale events to the client.
        """
        if self._mcp_tool_queue is not None:
            drained_mcp_count = 0
            try:
                while True:
                    self._mcp_tool_queue.get_nowait()
                    drained_mcp_count += 1
            except asyncio.QueueEmpty:
                pass
            if drained_mcp_count > 0:
                structlogger.debug(
                    "copilot_response_handler.drain_queues.mcp_drained",
                    drained_count=drained_mcp_count,
                )

        if self._plan_queue is not None:
            drained_plan_count = 0
            try:
                while True:
                    plan_event = self._plan_queue.get_nowait()
                    drained_plan_count += 1
                    # Capture the final plan if requested (for persistence)
                    if capture_final_plan:
                        self._final_plan = plan_event.tasks
            except asyncio.QueueEmpty:
                pass
            if drained_plan_count > 0:
                structlogger.debug(
                    "copilot_response_handler.drain_queues.plan_drained",
                    drained_count=drained_plan_count,
                    captured_final_plan=capture_final_plan,
                )

    def _reset_for_content_part(self) -> None:
        """Reset the handler for processing a new content part.

        This method resets state and initializes buffers for a new content part:
        - Resets prefix/suffix tracking state
        - Increments the content part index if the current index already has
          a buffer
        - Initializes a new buffer entry for the current content part index
        - Clears the rolling buffer to prevent cross-contamination between
          content parts
        """
        self._prefix_found = None
        self._suffix_found = None
        # Content part indices start from 0
        # If current index already has a buffer, this is a new content part,
        # so increment
        if self._current_content_part_index in self._llm_streamed_events:
            self._current_content_part_index += 1
        # Initialize the buffer for the current content part index
        self._llm_streamed_events[self._current_content_part_index] = []
        # Clear the rolling buffer for the new content part to prevent
        # cross-contamination between content parts.
        self._rolling_buffer.clear()

    # Streaming methods ----------------------------------------------------------------

    def _capture_final_plan_from_context(self, planning_token: Optional[Token]) -> None:
        """Capture the final plan from the planning context.

        This method retrieves the final plan from the planning context (if available)
        and updates the internal _final_plan state before the context is reset.

        Args:
            planning_token: Token for the planning context, or None if planning
                           context was not initialized.
        """
        if planning_token is not None:
            context_plan = get_final_plan()
            if context_plan:
                self._final_plan = context_plan
                structlogger.debug(
                    "copilot_response_handler.capture_final_plan_from_context",
                    task_count=len(context_plan),
                    task_statuses=[t.status for t in context_plan],
                )

    async def _yield_queued_events(
        self,
    ) -> AsyncGenerator[Union[MCPToolCall, TodoPlanUpdate], None]:
        """Yield and process any queued events.

        This method drains the MCP tool and plan queues non-blocking and yields
        and processes all events.
        """
        # Yield and process any queued MCP tool call events that accumulated since
        # the last stream event
        async for queued_event in self._yield_queued_mcp_tool_events():
            # Update the retrieved documents list with the documents from the
            # document retrieval MCP tool call created by the hook.
            if is_document_retrieval_mcp_tool_output_event(queued_event):
                self._update_retrieved_documents(queued_event)

            # Yield the MCP tool call event.
            yield queued_event

        # Yield any queued plan update events that accumulated since the
        # last stream event
        async for plan_event in self._yield_queued_plan_events():
            yield plan_event

    async def _yield_queued_mcp_tool_events(
        self,
    ) -> AsyncGenerator[MCPToolCall, None]:
        """Yield any queued MCP tool call events.

        This method drains the MCP tool queue non-blocking and yields all events.

        Yields:
            MCPToolCall: MCP tool call events from the queue.
        """
        if self._mcp_tool_queue is not None:
            try:
                while True:
                    mcp_event = self._mcp_tool_queue.get_nowait()
                    structlogger.debug(
                        "copilot_response_handler.yield_queued_mcp_tool_events.mcp_event_yielded",
                        mcp_event=mcp_event,
                    )
                    yield mcp_event
            except asyncio.QueueEmpty:
                structlogger.debug(
                    "copilot_response_handler.yield_queued_mcp_tool_events.mcp_drained",
                    event_info="No MCP tool events to yield",
                )

    async def _yield_queued_plan_events(self) -> AsyncGenerator[TodoPlanUpdate, None]:
        """Yield any queued plan update events.

        This method drains the plan queue non-blocking and yields all events.
        It also updates _final_plan with the latest plan state.

        Yields:
            TodoPlanUpdate: Plan update events from the queue.
        """
        if self._plan_queue is not None:
            try:
                while True:
                    plan_event = self._plan_queue.get_nowait()
                    self._final_plan = plan_event.tasks
                    structlogger.debug(
                        "copilot_response_handler.yield_queued_plan_events.plan_captured",
                        task_count=len(self._final_plan),
                        task_statuses=[t.status for t in self._final_plan],
                    )
                    yield plan_event
            except asyncio.QueueEmpty:
                structlogger.debug(
                    "copilot_response_handler.yield_queued_plan_events.plan_drained",
                    event_info="No plan update events to yield",
                )

    async def stream(self) -> AsyncGenerator[CopilotOutput, None]:
        """Stream and process Copilot responses from the response stream.

        This method processes stream events, detects text content parts, and yields
        processed CopilotOutput objects. It handles multiple content parts and
        processes each text content part through the streaming pipeline.

        It also manages:
        - Planning context lifecycle (ContextVar initialization/cleanup)
        - Queue draining for MCP tool calls and planning updates
        - Interleaving of StreamEvents with MCP/planning events

        Queue events (MCP tool calls, plan updates) are yielded:
        - After each stream event from the LLM
        - After text content streaming completes
        - At the end of the stream

        Note: Queue events may be delayed during long-running tool calls where
        the LLM stream is blocked waiting for tool results. This is a limitation
        of the current architecture where queue polling is tied to stream events.

        Yields:
            CopilotOutput: Processed output objects including text content, controlled
                predictions, exceptions, content part markers, MCP tool calls, and
                planning updates.

        Note:
            This method resets the handler state before processing.
        """
        self.reset()

        # Initialize planning context if we have the necessary components
        # Both tokens use ContextVar to ensure request isolation
        if self._plan_queue is not None:
            self._planning_token = init_planning_context()
            self._plan_queue_token = set_plan_queue(self._plan_queue)

        try:
            async for stream_event in self._response_stream:
                # Yield and process any queued events (MCP tool calls, plan updates)
                # that have accumulated since the last stream event
                async for queued_event in self._yield_queued_events():
                    yield queued_event

                # Check if the stream event signals the start of the text content part
                # streaming.
                if is_text_content_part_start_event(stream_event):
                    self._reset_for_content_part()
                    async for generated_content in self._stream_text_content_part():
                        yield generated_content
                    # After text streaming, yield any queued events that accumulated
                    async for queued_event in self._yield_queued_events():
                        yield queued_event

                # TODO: Add handling for other types of the content parts
                #      (reasoning, tool calls, audio, refusals, etc.)

                # For now, continue processing until the stream ends naturally
                # (don't break on non-text events as the Agent SDK sends many
                # event types before text content parts)

            # Final drain of queues to catch any remaining events
            # (tools may have added events after the last stream event)
            async for queued_event in self._yield_queued_events():
                yield queued_event

        except Exception as e:
            exception_content = ExceptionContent(
                content=EXCEPTION_RESPONSE,
                original_exception=e,
            )
            self._generated_responses.append(exception_content)
            yield exception_content
        finally:
            self._drain_queues_without_yielding(capture_final_plan=True)

            # Capture the final plan from planning context before it's reset
            self._capture_final_plan_from_context(self._planning_token)

    async def _stream_text_content_part(self) -> AsyncGenerator[GeneratedContent, None]:
        """Stream the text content part.

        Yields:
            GeneratedContent objects representing the text content part.
        """
        # The first event in the text content part streaming is the start event.
        text_content_part_start = CopilotTextStartContent()
        self._generated_responses.append(text_content_part_start)
        yield text_content_part_start

        try:
            # Exhaust the buffer early to check for controlled predictions and prefix
            # detection. The buffer is exhausted when the max expected special response
            # tokens are reached or the end of the text content part is reached before
            # that.
            (
                exhausted_events,
                final_buffer_reached,
            ) = await self._exhaust_text_content_part_buffer_for_early_detection()
            if final_buffer_reached:
                raise CopilotStreamEndedEarly()

            # Check for controlled predictions in the exhausted events
            category = self._check_events_for_controlled_predictions(exhausted_events)
            if category is not None:
                controlled_prediction = self.respond_to_controlled_prediction(category)
                text_content_part_end = CopilotTextEndContent()
                self._generated_responses.append(controlled_prediction)
                self._generated_responses.append(text_content_part_end)
                yield controlled_prediction
                yield text_content_part_end
                return

            # At this point, no controlled predictions were found. Check if the stream
            # started with a prefix and if present, remove it. Yield the clean content.
            initial_content = self._remove_prefix_from_events(exhausted_events)
            if initial_content is not None:
                cleaned_text_delta_content = CopilotTextContent(content=initial_content)
                self._generated_responses.append(cleaned_text_delta_content)
                yield cleaned_text_delta_content

            # Continue streaming remaining chunks with rolling buffer handling
            async for event in self._rolling_buffer_stream(
                yield_remaining_events_on_end=False,
                stop_streaming_on_content_part_end=True,
            ):
                if is_text_content_part_delta_event(event):
                    text_delta_content = CopilotTextContent(content=event.data.delta)  # type: ignore[union-attr]
                    self._generated_responses.append(text_delta_content)
                    yield text_delta_content

        # Stream ended "early" (before the max expected special response tokens were
        # reached).
        except CopilotStreamEndedEarly:
            # Check for controlled predictions in the collected tokens
            category = self._check_events_for_controlled_predictions(exhausted_events)
            if category is not None:
                controlled_prediction = self.respond_to_controlled_prediction(category)
                text_content_part_end = CopilotTextEndContent()
                self._generated_responses.append(controlled_prediction)
                self._generated_responses.append(text_content_part_end)
                yield controlled_prediction
                yield text_content_part_end
                return

            # At this point, no controlled predictions were found. Clean the content
            # from both the prefix and suffix if present. Yield the clean content.
            final_content = self._remove_prefix_and_suffix_from_events(exhausted_events)
            if final_content is not None:
                generated_content = CopilotTextContent(
                    content=final_content,
                    response_completeness=ResponseCompleteness.TOKEN,
                )
                self._generated_responses.append(generated_content)
                yield generated_content

            # Yield the end of the text content part.
            text_content_part_end = CopilotTextEndContent()
            self._generated_responses.append(text_content_part_end)
            yield text_content_part_end

        # Stream has ended either naturally or when the content part end event was
        # detected, process the final rolling buffer. Remove the suffix if present.
        except CopilotFinalBufferReached:
            # Access remaining events from the rolling buffer
            remaining_events = list(self._rolling_buffer)

            final_content = self._process_final_content_with_suffix(remaining_events)
            if final_content is not None:
                generated_content = CopilotTextContent(
                    content=final_content,
                    response_completeness=ResponseCompleteness.TOKEN,
                )
                self._generated_responses.append(generated_content)
                yield generated_content

            text_content_part_end = CopilotTextEndContent()
            self._generated_responses.append(text_content_part_end)
            yield text_content_part_end

        # Unexpected error occurred.
        except Exception as e:
            structlogger.exception(
                "copilot_response_handler.handle_response.unexpected_error",
            )
            raise e

        # Text content processing is complete. Some events LLM streamed may still be in
        # the rolling buffer. Preserve them in llm_streamed_events.
        finally:
            self._preserve_remaining_rolling_buffer_events()

    async def _exhaust_text_content_part_buffer_for_early_detection(
        self,
    ) -> Tuple[List[StreamEvent], bool]:
        """Exhaust the text content part buffer for early detection.

        This method collects the first few stream events from the streaming response to
        detect controlled predictions (e.g., roleplay detection, out-of-scope detection,
        error fallback, knowledge base access requests) that may appear in the early
        tokens. The buffering continues until one of the following conditions is met:
        - The maximum expected stream events are reached.
        - The text content part end event is detected.
        - The stream ends (CopilotFinalBufferReached is raised).

        Returns:
            A tuple containing:
            - List of exhausted stream events
            - Boolean flag indicating if the final buffer was reached (True if stream
              ended before max expected tokens, False otherwise)
        """
        exhausted_events: List[StreamEvent] = []
        final_buffer_reached: bool = False
        try:
            async for event in self._rolling_buffer_stream(
                yield_remaining_events_on_end=False,
                stop_streaming_on_content_part_end=True,
            ):
                exhausted_events.append(event)
                if len(exhausted_events) >= self._max_expected_special_response_tokens:
                    break
            return exhausted_events, final_buffer_reached

        # The rolling buffer ended before the max expected special response tokens were
        # reached. Return the exhausted events and set the final buffer reached flag to
        # True.
        except CopilotFinalBufferReached:
            exhausted_events.extend(self._rolling_buffer)
            final_buffer_reached = True
            return exhausted_events, final_buffer_reached

        # For unexpected errors, propagate the exception to the caller.
        except Exception as e:
            structlogger.exception(
                "copilot_response_handler"
                "._exhaust_text_content_part_buffer_for_early_detection"
                ".unexpected_error",
                error=e,
            )
            raise e

    async def _rolling_buffer_stream(
        self,
        yield_remaining_events_on_end: bool = False,
        stop_streaming_on_content_part_end: bool = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Wrapper that performs rolling buffer handling and automatically saves chunks.

        This method maintains a rolling buffer of stream events. Events are only yielded
        when the buffer is full. All events are automatically added to
        `_llm_streamed_events` for post-processing (e.g., prefix/suffix removal,
        controlled prediction detection).

        Args:
            yield_remaining_events_on_end: If True, yields all remaining events from the
                rolling buffer individually when the stream ends (and removes them from
                the buffer). If False, events remain in the buffer.
            stop_streaming_on_content_part_end: If True, stops iteration when a content
                part end event is detected. If False, continues streaming until the
                stream ends naturally.

        Yields:
            StreamEvent: Events from the original stream. Events are yielded when:
                - The rolling buffer is full (oldest event is yielded)
                - The stream ends and `yield_remaining_events_on_end=True` (all
                  remaining events are yielded individually and removed from the buffer)

        Raises:
            CopilotFinalBufferReached: When the stream ends (either naturally or when
                `stop_streaming_on_content_part_end=True` and a content part end event
                is detected). All remaining events in the rolling buffer are
                transferred to `_llm_streamed_events` before this exception is raised.
        """
        try:
            while True:
                # Get the next chunk from LLM stream and add it to the rolling buffer
                chunk = await anext(self._response_stream)
                self._rolling_buffer.append(chunk)

                # When the content part end event is detected, stop the rolling buffer
                # stream.
                if stop_streaming_on_content_part_end and is_content_part_end_event(
                    chunk
                ):
                    raise StopAsyncIteration

                # Only yield when buffer is full to maintain the rolling buffer
                # behavior
                if len(self._rolling_buffer) == self._rolling_buffer_size:
                    # Yield the oldest element.
                    oldest_element = self._rolling_buffer.popleft()
                    self._llm_streamed_events[self._current_content_part_index].append(
                        oldest_element
                    )
                    yield oldest_element

        except StopAsyncIteration:
            # Optionally yield remaining events individually, removing them from buffer
            if yield_remaining_events_on_end:
                while self._rolling_buffer:
                    event = self._rolling_buffer.popleft()
                    self._llm_streamed_events[self._current_content_part_index].append(
                        event
                    )
                    yield event

            # Raise an exception to signal that the rolling buffer has ended.
            raise CopilotFinalBufferReached()

        except Exception as e:
            structlogger.exception(
                "copilot_response_handler._buffer_stream.unexpected_error",
            )
            raise e

    def _preserve_remaining_rolling_buffer_events(self) -> None:
        """Preserve any remaining events from rolling buffer."""
        self._llm_streamed_events[self._current_content_part_index].extend(
            self._rolling_buffer
        )

    def _update_retrieved_documents(self, mcp_tool_call: MCPToolCall) -> None:
        # Check if the MCP tool call is a document retrieval tool call.
        if not is_document_retrieval_mcp_tool_output_event(mcp_tool_call):
            return

        if not mcp_tool_call.output:
            structlogger.warning(
                "copilot_response_handler._update_retrieved_documents"
                ".no_output_from_documentation_search",
                event_info="No output from documentation search. Skipping.",
            )
            return

        try:
            # # MCP tool search results are wrapped into a "text" content block:
            # # {"type":"text","text":"{...DocumentSearchResponse json string...}", ...}
            if isinstance(mcp_tool_call.output, str):
                raw_output = json.loads(mcp_tool_call.output)
            else:
                structlogger.error(
                    "copilot_response_handler._update_retrieved_documents"
                    ".invalid_output_type",
                    event_info=(
                        f"Invalid output type. Got {type(mcp_tool_call.output)} "
                        "instead of json string. Skipping."
                    ),
                    output_type=type(mcp_tool_call.output),
                )
                return

            # The raw output is a dictionary with a "type" and "text" key. Under the
            # "text" key is a JSON string that contains the DocumentSearchResponse.
            if raw_output.get("type") == "text" and raw_output.get("text"):
                raw_document_search_response = json.loads(raw_output.get("text"))
            else:
                structlogger.error(
                    "copilot_response_handler._update_retrieved_documents"
                    ".invalid_output_format",
                    event_info=(
                        f"Invalid output format. Got {raw_output} instead of a text "
                        "content block with a JSON string. Skipping."
                    ),
                    output_format=raw_output,
                )
                return

            documentation_search_results = DocumentSearchResponse.model_validate(
                raw_document_search_response
            )

        except Exception as e:
            structlogger.error(
                "copilot_response_handler._update_retrieved_documents"
                ".documentation_search_result_validation_error",
                event_info=(
                    "Documentation search result cannot be parsed due to the "
                    "validation error. Skipping."
                ),
                error=e,
            )
            return

        if documentation_search_results.error:
            structlogger.error(
                "copilot_response_handler._update_retrieved_documents"
                ".documentation_search_resulted_in_error",
                event_info="Documentation search resulted in error. Skipping.",
                error=documentation_search_results.error,
            )
            return

        for found_document in documentation_search_results.documents:
            document = Document(
                content=found_document.content,
                url=found_document.url,
                title=found_document.title,
            )
            self._retrieved_documents.append(document)

        structlogger.debug(
            "copilot_response_handler._update_retrieved_documents"
            ".retrieved_documents_updated",
            event_info="Retrieved documents updated.",
            retrieved_documents=self._retrieved_documents,
        )

    # Stream event processing methods ------------------------------------------------

    def _extract_text_content_part_delta(
        self, event: StreamEvent
    ) -> Optional[CopilotTextContent]:
        """Extract a text delta from a text content part stream event.

        Args:
            event: The stream event to extract the text delta from.

        Returns:
            GeneratedContent with COPILOT_TEXT_PART_DELTA category if the event
            contains a text delta, None otherwise.
        """
        if is_text_content_part_delta_event(event):
            # We know the event is a text content part delta event, so we can safely
            # safely access the delta property.
            return CopilotTextContent(content=event.data.delta)  # type: ignore[union-attr]
        return None

    def _check_events_for_controlled_predictions(
        self,
        events: List[StreamEvent],
    ) -> Optional[ResponseCategory]:
        """Check the provided stream events for controlled predictions.

        Args:
            events: List of stream events to check for controlled prediction markers.

        Returns:
            The response category if a controlled prediction is found, None otherwise.
        """
        text_content = extract_text_content_from_events(events)
        for prediction_marker, (_, category) in PREDICTION_RESPONSES.items():
            if prediction_marker in text_content:
                structlogger.info(
                    f"copilot_response_handler.{category.value}_detected",
                    event_info=f"Controlled prediction detected: {prediction_marker}",
                    category=category,
                )
                return category

        return None

    # Content cleaning methods ---------------------------------------------------------

    def _remove_prefix_from_events(self, events: List[StreamEvent]) -> Optional[str]:
        """Process initial content by removing prefix if present.

        Updates the prefix tracking state if a prefix is found.

        Returns:
            The processed content with prefix removed, or None if no content remains.
        """
        dirty_event_content = extract_text_content_from_events(events)
        clean_event_content, prefix = remove_prefix(dirty_event_content)
        if prefix:
            self._prefix_found = prefix
        return clean_event_content if clean_event_content else None

    def _remove_prefix_and_suffix_from_events(
        self, events: List[StreamEvent]
    ) -> Optional[str]:
        """Process final content by removing prefix and suffix if present.

        Updates the prefix and suffix tracking state if found.

        Returns:
            The processed content with prefix and suffix removed, or None if no content
            remains.
        """
        dirty_event_content = extract_text_content_from_events(events)
        final_content, prefix, suffix = remove_prefix_and_suffix(dirty_event_content)
        if prefix:
            self._prefix_found = prefix
        if suffix:
            self._suffix_found = suffix
        return final_content if final_content else None

    def _process_final_content_with_suffix(
        self, events: List[StreamEvent]
    ) -> Optional[str]:
        """Process final content by removing suffix if present.

        Updates the suffix tracking state if a suffix is found.

        Returns:
            The processed content with suffix removed, or None if no content remains.
        """
        dirty_event_content = extract_text_content_from_events(events)
        final_content, suffix = remove_suffix(dirty_event_content)
        if suffix:
            self._suffix_found = suffix
        return final_content if final_content else None

    # Response generation methods ------------------------------------------------------

    @staticmethod
    def respond_to_mcp_tool_hook() -> GeneratedContent:
        raise NotImplementedError("respond_to_mcp_tool_hook is not implemented")

    @staticmethod
    def respond_to_reasoning(
        content: str, completeness: ResponseCompleteness = ResponseCompleteness.TOKEN
    ) -> GeneratedContent:
        """Create a GeneratedContent response for reasoning.

        Args:
            content: The reasoning content to include in the response.
            completeness: Whether this is a streaming token or complete response.

        Returns:
            GeneratedContent with the reasoning content.
        """
        raise NotImplementedError("respond_to_reasoning is not implemented")

    def extract_text_from_stream_events(self) -> str:
        """Extract text from raw stream events across all content parts.

        Returns:
            Concatenated text deltas from all text content parts' stream events.
        """
        all_events: List[StreamEvent] = []
        for content_part_index in sorted(self._llm_streamed_events.keys()):
            all_events.extend(self._llm_streamed_events[content_part_index])
        return extract_text_content_from_events(all_events)

    def extract_text_from_generated_responses(self) -> str:
        """Extract and join all content from processed generated responses.

        Returns:
            str: Concatenated text from all complete generated content responses.
        """
        return extract_text_by_categories(
            responses=self.generated_responses,
            primary_categories=AGENT_PRIMARY_TEXT_CATEGORIES,
            controlled_prediction_categories=CONTROLLED_PREDICTION_CATEGORIES,
        )

    def extract_final_plan(self) -> Optional[List[TodoItem]]:
        """Extract the final task plan captured during streaming.

        This method returns the plan state that was captured during streaming.
        The plan is updated each time a TodoPlanUpdate event is received from
        the planning tools queue.

        Returns:
            List of TodoItem objects representing the final plan state,
            or None if no plan was created during the stream.
        """
        return self._final_plan
