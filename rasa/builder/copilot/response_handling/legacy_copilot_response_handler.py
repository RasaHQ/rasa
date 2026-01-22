import copy
from collections import deque
from typing import AsyncGenerator, Deque, List, Optional, Tuple

import structlog

from rasa.builder.copilot.exceptions import (
    CopilotFinalBufferReached,
    CopilotStreamEndedEarly,
)
from rasa.builder.copilot.models import (
    CopilotOutput,
    GeneratedContent,
    ResponseCategory,
    ResponseCompleteness,
)
from rasa.builder.copilot.response_handling.base_copilot_response_handler import (
    BaseCopilotResponseHandler,
)
from rasa.builder.copilot.response_handling.constants import (
    CONTROLLED_PREDICTION_CATEGORIES,
    LEGACY_PRIMARY_TEXT_CATEGORIES,
    PREDICTION_RESPONSES,
)
from rasa.builder.copilot.response_handling.utils import (
    extract_text_by_categories,
    remove_prefix,
    remove_suffix,
)
from rasa.builder.document_retrieval.models import Document

structlogger = structlog.get_logger()


class LegacyCopilotResponseHandler(BaseCopilotResponseHandler):
    """Handles the controlled responses from Copilot.

    This handler manages two types of data:
    - llm_stream_buffer: A list of tokens streamed from the LLM during processing.
    - generated_responses: A list of cleaned responses.
    - relevant_documents: A list of relevant documents retrieved from the InKeep API.

    Parameters:
        rolling_buffer_size: Size of the rolling buffer for prefix/suffix handling.
    """

    def __init__(
        self,
        response_stream: AsyncGenerator[str, None],
        rolling_buffer_size: int = 20,
        relevant_documents: Optional[List[Document]] = None,
    ):
        self._rolling_buffer_size = rolling_buffer_size
        self._response_stream = response_stream

        # Rolling buffer for handling special tokens and prefix/suffix removal.
        self._rolling_buffer: Deque[str] = deque(maxlen=self._rolling_buffer_size)

        # A list of tokens streamed from the LLM during processing. Tokens are added
        # to the buffer when the rolling buffer is full.
        self._llm_stream_buffer: List[str] = []

        # A list of cleaned and generated responses.
        self._generated_responses: List[GeneratedContent] = []

        # Maximum number of tokens to check for special responses (e.g. roleplay,
        # out-of-scope, etc.).
        self._max_expected_special_response_tokens: int = 20

        # Prefix/suffix tracking
        self._prefix_found: Optional[str] = None
        self._suffix_found: Optional[str] = None

        self._retrieved_documents: List[Document] = relevant_documents or []

    @property
    def generated_responses(self) -> List[GeneratedContent]:
        return copy.deepcopy(self._generated_responses)

    @property
    def generated_responses_count(self) -> int:
        return len(self._generated_responses)

    @property
    def raw_llm_stream_data(self) -> List[str]:
        return copy.deepcopy(self._llm_stream_buffer)

    @property
    def raw_llm_stream_text_content(self) -> str:
        return "".join(self._llm_stream_buffer)

    @property
    def raw_llm_stream_item_count(self) -> int:
        return len(self._llm_stream_buffer)

    @property
    def retrieved_documents(self) -> List[Document]:
        return copy.deepcopy(self._retrieved_documents)

    def reset(self) -> None:
        """Clear all buffers and reset the handler."""
        self._rolling_buffer.clear()
        self._llm_stream_buffer.clear()
        self._generated_responses.clear()
        self._prefix_found = None
        self._suffix_found = None

    # Streaming methods ----------------------------------------------------------------

    async def stream(self) -> AsyncGenerator[CopilotOutput, None]:
        """Intercept a streaming response and handle special responses from the Copilot.

        Yields:
            ResponseEvent objects representing either generated tokens, default
            responses, or reference sections.
        """
        # Clear the stream buffer and reference buffer at the start
        self.reset()

        try:
            # Exhaust the buffer early to check for controlled predictions and prefix
            # detection.
            (
                exhausted_tokens,
                final_buffer_reached,
            ) = await self._exhaust_buffer_for_early_detection(self._response_stream)
            if final_buffer_reached:
                raise CopilotStreamEndedEarly()

            # Check for controlled predictions in the collected tokens
            category = self._check_for_controlled_predictions(exhausted_tokens)
            if category is not None:
                controlled_prediction = self.respond_to_controlled_prediction(category)
                self._generated_responses.append(controlled_prediction)
                yield controlled_prediction
                return

            # At this point, no controlled predictions were found. Check if the stream
            # started with a prefix and if present, remove it. Yield the clean content.
            initial_content = self._remove_prefix(exhausted_tokens)
            if initial_content:
                generated_content = GeneratedContent(
                    content=initial_content,
                    response_category=ResponseCategory.COPILOT,
                    response_completeness=ResponseCompleteness.TOKEN,
                )
                self._generated_responses.append(generated_content)
                yield generated_content

            # Continue streaming remaining chunks with rolling buffer handling
            async for chunk in self._rolling_buffer_stream(self._response_stream):
                generated_content = GeneratedContent(
                    content=chunk,
                    response_category=ResponseCategory.COPILOT,
                    response_completeness=ResponseCompleteness.TOKEN,
                )
                self._generated_responses.append(generated_content)
                yield generated_content

        # Stream ended early
        except CopilotStreamEndedEarly:
            # Check for controlled predictions in the collected tokens
            category = self._check_for_controlled_predictions(exhausted_tokens)
            if category is not None:
                controlled_prediction = self.respond_to_controlled_prediction(category)
                self._generated_responses.append(controlled_prediction)
                yield controlled_prediction
                return

            # At this point, no controlled predictions were found. Clean the content
            # from the prefix and suffix if present.
            final_content = self._remove_prefix_and_suffix(exhausted_tokens)
            if final_content:
                generated_content = GeneratedContent(
                    content=final_content,
                    response_category=ResponseCategory.COPILOT,
                    response_completeness=ResponseCompleteness.TOKEN,
                )
                self._generated_responses.append(generated_content)
                yield generated_content

        # Stream has ended, process the final rolling buffer. Remove the suffix if
        # present.
        except CopilotFinalBufferReached:
            remaining_tokens = list(self._rolling_buffer)
            final_content = self._remove_suffix(remaining_tokens)
            if final_content:
                generated_content = GeneratedContent(
                    content=final_content,
                    response_category=ResponseCategory.COPILOT,
                    response_completeness=ResponseCompleteness.TOKEN,
                )
                self._generated_responses.append(generated_content)
                yield generated_content

        # Unexpected error occurred.
        except Exception as e:
            structlogger.exception(
                "copilot_response_handler.handle_response.unexpected_error",
            )
            raise e

        finally:
            self._llm_stream_buffer.extend(self._rolling_buffer)
            self._rolling_buffer.clear()

    async def _exhaust_buffer_for_early_detection(
        self, response_stream: AsyncGenerator[str, None]
    ) -> Tuple[List[str], bool]:
        """Exhaust the buffer for early detection.

        Args:
            response_stream: The original streaming response from the LLM.
        """
        exhausted_tokens: List[str] = []
        final_buffer_reached: bool = False
        try:
            async for token in self._rolling_buffer_stream(
                response_stream,
                yield_remaining_events_on_end=False,
            ):
                exhausted_tokens.append(token)
                if len(exhausted_tokens) >= self._max_expected_special_response_tokens:
                    break
            return exhausted_tokens, final_buffer_reached

        # The rolling buffer ended before the max expected special response tokens were
        # reached. Return the exhausted tokens and set the final buffer reached flag to
        # True.
        except CopilotFinalBufferReached:
            exhausted_tokens.extend(self._rolling_buffer)
            final_buffer_reached = True
            return exhausted_tokens, final_buffer_reached

        # For unexpected errors, propagate the exception to the caller.
        except Exception as e:
            structlogger.exception(
                "copilot_response_handler"
                "._exhaust_buffer_for_early_detection"
                ".unexpected_error",
                error=e,
            )
            raise e

    async def _rolling_buffer_stream(
        self,
        response_stream: AsyncGenerator[str, None],
        yield_remaining_events_on_end: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Wrapper that performs rolling buffer handling and automatically saves chunks.

        Args:
            response_stream: The original streaming response from the LLM.
            yield_remaining_events_on_end: If True, yields all remaining events from the
                rolling buffer individually when the stream ends (and removes them from
                the buffer). If False, events remain in the buffer.

        Yields:
            The same chunks from the original stream, but persisted in the buffer.

        Raises:
            StopAsyncIteration: If the stream has ended.
        """
        try:
            while True:
                # Get the next chunk from LLM stream and add it to the rolling buffer
                # and the LLM stream buffer.
                chunk = await anext(response_stream)
                self._rolling_buffer.append(chunk)

                # Only yield when buffer is full to maintain the rolling buffer
                # behavior
                if len(self._rolling_buffer) == self._rolling_buffer_size:
                    # Yield the oldest element.
                    oldest_element = self._rolling_buffer.popleft()
                    self._llm_stream_buffer.append(oldest_element)
                    yield oldest_element

        except StopAsyncIteration:
            # Optionally yield remaining events individually, removing them from buffer
            if yield_remaining_events_on_end:
                while self._rolling_buffer:
                    event = self._rolling_buffer.popleft()
                    self._llm_stream_buffer.append(event)
                    yield event

            # Raise an exception to signal that the rolling buffer has ended.
            raise CopilotFinalBufferReached()

        except Exception as e:
            structlogger.exception(
                "copilot_response_handler._buffer_stream.unexpected_error",
            )
            raise e

    # Controlled prediction detection methods ------------------------------------------

    def _check_for_controlled_predictions(
        self, tokens: List[str]
    ) -> Optional[ResponseCategory]:
        """Check for controlled predictions in the collected tokens.

        Returns:
            The response category if a controlled prediction is found, None otherwise.
        """
        # Check for controlled predictions and return the category
        content = "".join(tokens)
        for prediction_marker, (_, category) in PREDICTION_RESPONSES.items():
            if prediction_marker in content:
                structlogger.info(
                    f"copilot_response_handler.{category.value}_detected",
                    event_info=f"Controlled prediction detected: {prediction_marker}",
                    category=category,
                )
                return category

        return None

    # Content extraction methods -------------------------------------------------------

    def extract_text_from_generated_responses(self) -> str:
        """Extract and join all content from processed generated responses.

        Returns:
            str: Concatenated text from all complete generated content responses.
        """
        return extract_text_by_categories(
            responses=self.generated_responses,
            primary_categories=LEGACY_PRIMARY_TEXT_CATEGORIES,
            controlled_prediction_categories=CONTROLLED_PREDICTION_CATEGORIES,
        )

    # Content cleaning methods ---------------------------------------------------------

    def _remove_prefix(self, tokens: List[str]) -> str:
        """Process the initial content from the buffer, handling prefix removal.

        Returns:
            Processed content with prefix removed if applicable.
        """
        dirty_content = "".join(tokens)
        clean_content, prefix = remove_prefix(dirty_content)
        if prefix:
            self._prefix_found = prefix
            structlogger.debug(
                "copilot_response_handler.handle_response.prefix_detected",
                prefix=prefix,
                dirty_content_length=len(dirty_content),
                clean_content_length=len(clean_content),
            )
        return clean_content

    def _remove_suffix(self, tokens: List[str]) -> str:
        """Process the rolling buffer content, handling suffix removal.

        Returns:
            Processed content with suffix removed if applicable.
        """
        dirty_content = "".join(tokens)
        clean_content, suffix = remove_suffix(dirty_content)
        if suffix:
            self._suffix_found = suffix
            structlogger.debug(
                "copilot_response_handler.handle_response.suffix_detected",
                suffix=suffix,
                dirty_content_length=len(dirty_content),
                clean_content_length=len(clean_content),
            )
        return clean_content

    def _remove_prefix_and_suffix(self, tokens: List[str]) -> str:
        """Remove the prefix and suffix from the content.

        Returns:
            Processed content with prefix and suffix removed if applicable.
        """
        dirty_content = "".join(tokens)
        clean_content, prefix = remove_prefix(dirty_content)
        clean_content, suffix = remove_suffix(clean_content)
        if prefix:
            self._prefix_found = prefix
        if suffix:
            self._suffix_found = suffix
        return clean_content
