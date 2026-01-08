import copy
from collections import defaultdict, deque
from typing import AsyncGenerator, Deque, Dict, List, Optional, Tuple

import structlog
from agents import StreamEvent

from rasa.builder.copilot.exceptions import (
    CopilotFinalBufferReached,
    CopilotStreamEndedEarly,
)
from rasa.builder.copilot.models import (
    CopilotOutput,
    CopilotTextContent,
    CopilotTextEndContent,
    CopilotTextStartContent,
    ExceptionContent,
    GeneratedContent,
    ResponseCategory,
    ResponseCompleteness,
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
    is_text_content_part_delta_event,
    is_text_content_part_start_event,
    remove_prefix,
    remove_prefix_and_suffix,
    remove_suffix,
)

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
    ):
        self._rolling_buffer_size = rolling_buffer_size
        self._response_stream = response_stream

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

        # Maximum number of tokens to check for special responses (e.g. roleplay,
        # out-of-scope, etc.).
        self._max_expected_special_response_tokens: int = 20

        # Prefix/suffix tracking state
        self._prefix_found: Optional[str] = None
        self._suffix_found: Optional[str] = None

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

    def reset(self) -> None:
        """Clear all buffers and reset the handler."""
        self._rolling_buffer.clear()
        self._llm_streamed_events.clear()
        self._current_content_part_index = 0
        self._generated_responses.clear()
        self._prefix_found = None
        self._suffix_found = None

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

    async def stream(self) -> AsyncGenerator[CopilotOutput, None]:
        """Stream and process Copilot responses from the response stream.

        This method processes stream events, detects text content parts, and yields
        processed CopilotOutput objects. It handles multiple content parts and
        processes each text content part through the streaming pipeline.

        Yields:
            CopilotOutput: Processed output objects including text content, controlled
                predictions, exceptions, and content part markers.

        Note:
            This method resets the handler state before processing.
        """
        self.reset()
        try:
            async for stream_event in self._response_stream:
                # Check if the stream event signals the start of the text content part
                # streaming.
                if is_text_content_part_start_event(stream_event):
                    self._reset_for_content_part()
                    async for generated_content in self._stream_text_content_part():
                        yield generated_content

                # TODO: Add handling for other types of the content parts
                #      (reasoning, tool calls, audio, refusals, etc.)

        except Exception as e:
            exception_content = ExceptionContent(
                content=EXCEPTION_RESPONSE,
                original_exception=e,
            )
            self._generated_responses.append(exception_content)
            yield exception_content

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
                # and the LLM stream buffer.
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
