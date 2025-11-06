import copy
import re
from collections import deque
from typing import AsyncGenerator, Deque, Dict, List, Optional, Tuple

import structlog

from rasa.builder.copilot.copilot_templated_message_provider import (
    load_copilot_handler_default_responses,
)
from rasa.builder.copilot.exceptions import (
    CopilotFinalBufferReached,
    CopilotStreamEndedEarly,
)
from rasa.builder.copilot.models import (
    CopilotOutput,
    GeneratedContent,
    ReferenceEntry,
    ReferenceSection,
    ResponseCategory,
    ResponseCompleteness,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.guardrails.constants import BLOCK_SCOPE_USER, BlockScope

structlogger = structlog.get_logger()

# Controlled prediction markers
ROLEPLAY_PREDICTION = "[ROLEPLAY_REQUEST_DETECTED]"
OUT_OF_SCOPE_PREDICTION = "[OUT_OF_SCOPE_REQUEST_DETECTED]"
ERROR_FALLBACK_PREDICTION = "[ERROR_FALLBACK]"
KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION = "[NO_KNOWLEDGE_BASE_ACCESS]"

# Load predefined for controlled predictions from YAML
_handler_responses = load_copilot_handler_default_responses()

# Prediction marker to response mapping
PREDICTION_RESPONSES = {
    ROLEPLAY_PREDICTION: (
        _handler_responses.get("roleplay_response", ""),
        ResponseCategory.ROLEPLAY_DETECTION,
    ),
    OUT_OF_SCOPE_PREDICTION: (
        _handler_responses.get("out_of_scope_response", ""),
        ResponseCategory.OUT_OF_SCOPE_DETECTION,
    ),
    ERROR_FALLBACK_PREDICTION: (
        _handler_responses.get("error_fallback_response", ""),
        ResponseCategory.ERROR_FALLBACK,
    ),
    KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION: (
        _handler_responses.get("knowledge_base_access_requested_response", ""),
        ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED,
    ),
}

# Other response constants
GUARDRAIL_POLICY_VIOLATION_RESPONSE = _handler_responses.get(
    "guardrail_policy_violation_response", ""
)
COPILOT_REDACTED_MESSAGE = _handler_responses.get("copilot_redacted_message", "")

# Guardrails blocked responses
GUARDRAIL_BLOCKED_USER_RESPONSE = _handler_responses.get(
    "guardrail_blocked_user_response", ""
)
GUARDRAIL_BLOCKED_PROJECT_RESPONSE = _handler_responses.get(
    "guardrail_blocked_project_response", ""
)

# Common LLM response prefixes and suffixes before the actual content. These are removed
# from the content.
LLM_PREFIXES_TO_SUFFIX_REMOVE = {
    "```": "```",
    "```markdown": "```",
    '"""': '"""',
}

# Regex pattern to match inline citations in the generated markdown.
INLINE_CITATION_PATTERN = r"\[([^\]]+)\]\(([^)]+)\)"


class CopilotResponseHandler:
    """Handles the controlled responses from Copilot.

    This handler manages two types of data:
    - llm_stream_buffer: A list of tokens streamed from the LLM during processing.
    - generated_responses: A list of cleaned responses.

    Parameters:
        rolling_buffer_size: Size of the rolling buffer for prefix/suffix handling.
    """

    def __init__(
        self,
        rolling_buffer_size: int = 20,
    ):
        self._rolling_buffer_size = rolling_buffer_size

        # Rolling buffer for handling special tokens and prefix/suffix removal.
        self._rolling_buffer: Deque[str] = deque(maxlen=self._rolling_buffer_size)

        # A list of tokens streamed from the LLM during processing. Tokens are added
        # to the buffer when the rolling buffer is full.
        self._llm_stream_buffer: List[str] = []

        # A list of cleaned and generated responses.
        self._generated_responses: List[GeneratedContent] = []

        # Flag for checking if the stream has ended (raised by the async generator).
        self._stream_ended: bool = False
        # Flag for checking if the stream ended before the max expected special response
        # tokens were reached.
        self._stream_ended_early: bool = False

        # Maximum number of tokens to check for special responses (e.g. roleplay,
        # out-of-scope, etc.).
        self._max_expected_special_response_tokens: int = 20

        # Prefix/suffix tracking
        self._stream_started_with_prefix: bool = False
        self._prefix_found: Optional[str] = None
        self._suffix_found: Optional[str] = None

    @property
    def llm_stream_buffer(self) -> List[str]:
        return copy.deepcopy(self._llm_stream_buffer)

    @property
    def generated_responses(self) -> List[GeneratedContent]:
        return copy.deepcopy(self._generated_responses)

    @property
    def llm_stream_buffer_content(self) -> str:
        return "".join(self._llm_stream_buffer)

    @property
    def token_count(self) -> int:
        return len(self._llm_stream_buffer)

    def clear_buffers(self) -> None:
        """Clear all buffers and reset the handler."""
        self._rolling_buffer.clear()
        self._llm_stream_buffer.clear()
        self._generated_responses.clear()
        self._stream_started_with_prefix = False
        self._prefix_found = None
        self._suffix_found = None
        self._stream_ended = False

    async def _buffer_stream(
        self, response_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """Wrapper that performs rolling buffer handling and automatically saves chunks.

        Args:
            response_stream: The original streaming response from the LLM.

        Yields:
            The same chunks from the original stream, but persisted in the buffer.

        Raises:
            StopAsyncIteration: If the stream has ended.
        """
        try:
            while True:
                # Get the next chunk from LLM stream and add it to the rolling buffer
                # and the LLM stream buffer.
                chunk = await response_stream.__anext__()
                self._rolling_buffer.append(chunk)

                # Only yield when buffer is full to maintain the rolling buffer
                # behavior
                if len(self._rolling_buffer) == self._rolling_buffer_size:
                    # Yield the oldest element.
                    oldest_element = self._rolling_buffer.popleft()
                    self._llm_stream_buffer.append(oldest_element)
                    yield oldest_element

        except StopAsyncIteration:
            self._llm_stream_buffer.extend(self._rolling_buffer)
            self._stream_ended = True
            # Need to raise a new instance of StopAsyncIteration to avoid the
            # RuntimeError: generator raised StopIteration
            raise CopilotFinalBufferReached()

        except Exception as e:
            structlogger.exception(
                "copilot_response_handler._buffer_stream.unexpected_error",
            )
            raise e

    async def handle_response(
        self, response_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[CopilotOutput, None]:
        """Intercept a streaming response and handle special responses from the Copilot.

        Args:
            response_stream: The original streaming response from the LLM.

        Yields:
            ResponseEvent objects representing either generated tokens, default
            responses, or reference sections.
        """
        # Clear the stream buffer and reference buffer at the start
        self.clear_buffers()

        try:
            # Exhaust the buffer early to check for controlled predictions and prefix
            # detection.
            await self._exhaust_buffer_for_early_detection(response_stream)

            # Check for controlled predictions in the collected tokens
            controlled_response = self._check_for_controlled_predictions(
                self.llm_stream_buffer_content,
            )
            if controlled_response is not None:
                self._generated_responses.append(controlled_response)
                yield controlled_response
                return

            # At this point, no controlled predictions were found. Check if the stream
            # started with a prefix and if present, remove it. Yield the clean content.
            initial_content = self._remove_prefix(self.llm_stream_buffer_content)
            if initial_content:
                generated_content = GeneratedContent(
                    content=initial_content,
                    response_category=ResponseCategory.COPILOT,
                    response_completeness=ResponseCompleteness.TOKEN,
                )
                self._generated_responses.append(generated_content)
                yield generated_content

            # Continue streaming remaining chunks with rolling buffer handling
            async for chunk in self._buffer_stream(response_stream):
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
            controlled_response = self._check_for_controlled_predictions(
                self.llm_stream_buffer_content,
            )
            if controlled_response is not None:
                self._generated_responses.append(controlled_response)
                yield controlled_response
                return

            # At this point, no controlled predictions were found. Clean the content
            # from the prefix and suffix if present.
            final_content = self._remove_prefix_and_suffix(
                self.llm_stream_buffer_content,
            )
            if final_content:
                generated_content = GeneratedContent(
                    content=final_content,
                    response_category=ResponseCategory.COPILOT,
                    response_completeness=ResponseCompleteness.COMPLETE,
                )
                self._generated_responses.append(generated_content)
                yield generated_content

        # Stream has ended, process the final rolling buffer. Remove the suffix if
        # present.
        except CopilotFinalBufferReached:
            final_content = self._remove_suffix(self.llm_stream_buffer_content)
            if final_content:
                generated_content = GeneratedContent(
                    content=final_content,
                    response_category=ResponseCategory.COPILOT,
                    response_completeness=ResponseCompleteness.COMPLETE,
                )
                self._generated_responses.append(generated_content)
                yield generated_content

        # Unexpected error occurred.
        except Exception as e:
            structlogger.exception(
                "copilot_response_handler.handle_response.unexpected_error",
            )
            raise e

    async def _exhaust_buffer_for_early_detection(
        self, response_stream: AsyncGenerator[str, None]
    ) -> None:
        """Exhaust the buffer for early detection.

        Args:
            response_stream: The original streaming response from the LLM.
        """
        try:
            async for _ in self._buffer_stream(response_stream):
                if self.token_count >= self._max_expected_special_response_tokens:
                    break
        except CopilotFinalBufferReached:
            # The stream ended before the max expected special response tokens
            # were reached.
            raise CopilotStreamEndedEarly()

    def _check_for_controlled_predictions(
        self, content: str
    ) -> Optional[CopilotOutput]:
        """Check for controlled predictions in the collected tokens.

        Returns:
            Controlled response if found, None otherwise.
        """
        # Check for controlled predictions and return appropriate responses
        for prediction_marker, (response, category) in PREDICTION_RESPONSES.items():
            if prediction_marker in content:
                log_message = f"copilot_response_handler.{category.value}_detected"
                structlogger.info(log_message)
                return GeneratedContent(
                    response_category=category,
                    content=response,
                    response_completeness=ResponseCompleteness.COMPLETE,
                )

        return None

    def _remove_prefix(self, content: str) -> str:
        """Process the initial content from the buffer, handling prefix removal.

        Returns:
            Processed content with prefix removed if applicable.
        """
        # Check if content starts with any of the known prefixes
        for prefix in LLM_PREFIXES_TO_SUFFIX_REMOVE.keys():
            if content.startswith(prefix):
                self._stream_started_with_prefix = True
                self._prefix_found = prefix
                self._code_block_depth = 1
                structlogger.debug(
                    "copilot_response_handler.handle_response.prefix_detected",
                    prefix=prefix,
                    content_length=len(content),
                )
                return content[len(prefix) :]

        return content

    def _remove_suffix(self, content: str) -> str:
        """Process the rolling buffer content, handling suffix removal.

        Returns:
            Processed content with suffix removed if applicable.
        """
        # Check if content ends with any of the known suffixes
        for suffix in LLM_PREFIXES_TO_SUFFIX_REMOVE.values():
            if content.endswith(suffix):
                self._stream_ended_with_suffix = True
                self._suffix_found = suffix
                structlogger.debug(
                    "copilot_response_handler.handle_response.suffix_detected",
                    suffix=suffix,
                    content_length=len(content),
                )
                return content[: -len(suffix)]

        return content

    def _remove_prefix_and_suffix(self, content: str) -> str:
        """Remove the prefix and suffix from the content.

        Returns:
            Processed content with prefix and suffix removed if applicable.
        """
        content = self._remove_prefix(content)
        content = self._remove_suffix(content)
        return content

    @staticmethod
    def respond_to_guardrail_policy_violations() -> GeneratedContent:
        """Respond to guardrail policy violations.

        Returns:
            CopilotOutput object with the response.
        """
        return GeneratedContent(
            response_category=ResponseCategory.GUARDRAILS_POLICY_VIOLATION,
            content=GUARDRAIL_POLICY_VIOLATION_RESPONSE,
            response_completeness=ResponseCompleteness.COMPLETE,
        )

    @staticmethod
    def respond_to_guardrail_blocked(scope: BlockScope) -> GeneratedContent:
        """Return a blocked response for user or project scope.

        Args:
            scope: 'user' for user-level block, 'project' for project-level block.

        Returns:
            GeneratedContent with GUARDRAILS_BLOCKED category.
        """
        content = (
            GUARDRAIL_BLOCKED_USER_RESPONSE
            if scope == BLOCK_SCOPE_USER
            else GUARDRAIL_BLOCKED_PROJECT_RESPONSE
        )
        return GeneratedContent(
            response_category=ResponseCategory.GUARDRAILS_BLOCKED,
            content=content,
            response_completeness=ResponseCompleteness.COMPLETE,
        )

    @staticmethod
    def get_copilot_redacted_message() -> str:
        """Get the redacted message for copilot responses.

        Returns:
            Redacted message for copilot responses.
        """
        return COPILOT_REDACTED_MESSAGE

    def extract_references(self, documents: List[Document]) -> ReferenceSection:
        """Extract references from the LLM stream buffer content.

        This method performs regex matching to find markdown links in the format:
        [text](url).

        The matched links are validated against the provided documents, and a
        ReferenceSection is returned with valid references.

        Note:
            This method can only be called after `handle_response` has been called,
            as it relies on the LLM stream buffer content that gets populated
            during the response handling process.

        Args:
            documents: List of Document objects to match URLs against

        Returns:
            ReferenceSection containing reference entries ordered by reference text.

        Raises:
            RuntimeError: If called before `handle_response` has been called.
        """
        if not self.llm_stream_buffer_content:
            message = (
                "`extract_references` can only be called after `handle_response` "
                "has been called to populate the LLM stream buffer."
            )
            structlogger.error(
                "copilot_response_handler.extract_references.buffer_not_populated",
                event_info=message,
                llm_stream_buffer_content=self.llm_stream_buffer_content,
            )
            raise RuntimeError(message)

        # Find all matches in the buffered content
        matches = re.findall(INLINE_CITATION_PATTERN, self.llm_stream_buffer_content)

        # Create document lookup for O(1) access
        document_urls_to_documents: Dict[str, Document] = {
            document.url: document for document in documents if document.url
        }

        # Use regular dict to collect references, keyed by reference_text
        used_references: Dict[str, ReferenceEntry] = {}

        for reference_text, reference_url in matches:
            # Validate reference text format
            if not reference_text.isdigit():
                structlogger.warning(
                    "copilot_response_handler"
                    ".extract_references."
                    "invalid_reference_number",
                    event_info="Reference text is not in expected number format.",
                    reference_text=reference_text,
                    reference_url=reference_url,
                )
                continue

            # Check if URL exists in documents
            if reference_url not in document_urls_to_documents:
                structlogger.warning(
                    "copilot_response_handler.extract_references.url_not_found",
                    event_info=(
                        "URL not found in provided documents. Omitted from reference "
                        "section.",
                    ),
                    reference_url=reference_url,
                    available_urls=list(document_urls_to_documents.keys()),
                )
                continue

            # Check for duplicate reference text
            # (same reference number used multiple times)
            if reference_text in used_references:
                existing_entry = used_references[reference_text]
                if existing_entry.url != reference_url:
                    structlogger.warning(
                        "copilot_response_handler.extract_references.duplicate_reference_text",
                        event_info=(
                            "Same reference text used for different URLs. "
                            "Keeping first occurrence."
                        ),
                        reference_text=reference_text,
                        first_url=existing_entry.url,
                        second_url=reference_url,
                    )
                # Skip this duplicate reference text
                continue

            # Create reference entry
            document = document_urls_to_documents[reference_url]
            used_references[reference_text] = ReferenceEntry(
                index=int(reference_text),
                title=document.title or f"Reference {reference_text}",
                url=reference_url,
            )

        # Create and sort the reference section
        reference_section = ReferenceSection(references=list(used_references.values()))
        reference_section.sort_references()
        return reference_section

    def extract_full_text(self) -> str:
        """Extract and join all text content from the handler's responses.

        Returns:
            str: Concatenated text from all generated content responses.
        """
        text_parts: List[str] = []

        for response in self.generated_responses or []:
            if isinstance(response, GeneratedContent) and response.content:
                text_parts.append(response.content)

        return "".join(text_parts)

    def extract_response_category(self) -> ResponseCategory:
        """Extract the last non-reference response category from the handler.

        Returns:
            ResponseCategory: The last response category, excluding REFERENCE and
            REFERENCE_ENTRY categories. Defaults to COPILOT if none found.
        """
        last_category: Optional[ResponseCategory] = None

        for response in self.generated_responses or []:
            if (
                isinstance(response, GeneratedContent)
                and response.content
                and response.response_category
                not in {
                    ResponseCategory.REFERENCE,
                    ResponseCategory.REFERENCE_ENTRY,
                }
            ):
                last_category = response.response_category

        return last_category or ResponseCategory.COPILOT

    def extract_full_text_and_category(self) -> Tuple[str, ResponseCategory]:
        """Extract full text and response category from the handler's responses.

        Returns:
            Tuple[str, ResponseCategory]: Text and the last response category.
        """
        return self.extract_full_text(), self.extract_response_category()
