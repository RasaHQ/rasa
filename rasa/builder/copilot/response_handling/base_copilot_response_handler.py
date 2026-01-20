import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import structlog

from rasa.builder.copilot.models import (
    CommitInformationContent,
    ControlledPredictionContent,
    GeneratedContent,
    GuardrailBlockedContent,
    GuardrailPolicyViolationContent,
    ReferenceEntry,
    ReferenceSection,
    ResponseCategory,
    TodoItem,
)
from rasa.builder.copilot.response_handling.constants import (
    CONTROLLED_PREDICTION_CATEGORIES,
    COPILOT_REDACTED_MESSAGE,
    GUARDRAIL_BLOCKED_PROJECT_RESPONSE,
    GUARDRAIL_BLOCKED_USER_RESPONSE,
    GUARDRAIL_POLICY_VIOLATION_RESPONSE,
    INLINE_CITATION_PATTERN,
    PREDICTION_RESPONSES,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.guardrails.constants import BLOCK_SCOPE_USER, BlockScope

structlogger = structlog.get_logger()


class BaseCopilotResponseHandler(ABC):
    """Base class for Copilot response handlers.

    This class provides common functionality for handling Copilot responses,
    including reference extraction, text extraction, and guardrail responses.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the handler state.

        This method should reset the handler state to its initial state.
        """
        pass

    @property
    @abstractmethod
    def generated_responses(self) -> List[GeneratedContent]:
        """Get the list of generated responses.

        Returns:
            List of GeneratedContent objects representing processed responses.
            All returned types have a 'content' attribute.
        """
        pass

    @property
    @abstractmethod
    def generated_responses_count(self) -> int:
        """Get the number of generated responses.

        Returns:
            Number of generated responses.
        """
        pass

    @property
    @abstractmethod
    def raw_llm_stream_data(self) -> Any:
        """Get the raw LLM stream data.

        This property returns the raw, unprocessed data from the LLM stream.
        The exact type and structure depends on the handler implementation.

        Returns:
            Raw LLM stream data. The type is implementation-specific.
        """
        pass

    @property
    @abstractmethod
    def raw_llm_stream_text_content(self) -> str:
        """Get the raw LLM stream text content.

        Returns:
            Raw LLM stream text content.
        """
        pass

    @property
    @abstractmethod
    def raw_llm_stream_item_count(self) -> int:
        """Get the total number of stream items.

        Returns:
            Total number of stream items across all content parts.
        """
        pass

    def has_been_run(self) -> bool:
        """Check if the Copilot response has been run.

        Returns:
            True if the copilot response has been run, False otherwise.
        """
        return self.raw_llm_stream_item_count > 0 or self.generated_responses_count > 0

    # Response generation methods ------------------------------------------------------

    @staticmethod
    def respond_to_guardrail_policy_violations() -> GuardrailPolicyViolationContent:
        """Respond to guardrail policy violations.

        Returns:
            GuardrailPolicyViolationContent with the response.
        """
        return GuardrailPolicyViolationContent(
            content=GUARDRAIL_POLICY_VIOLATION_RESPONSE,
        )

    @staticmethod
    def respond_to_guardrail_blocked(scope: BlockScope) -> GuardrailBlockedContent:
        """Return a blocked response for user or project scope.

        Args:
            scope: 'user' for user-level block, 'project' for project-level block.

        Returns:
            GuardrailBlockedContent with the appropriate content.
        """
        content = (
            GUARDRAIL_BLOCKED_USER_RESPONSE
            if scope == BLOCK_SCOPE_USER
            else GUARDRAIL_BLOCKED_PROJECT_RESPONSE
        )
        return GuardrailBlockedContent(content=content)

    @staticmethod
    def respond_to_controlled_prediction(
        category: ResponseCategory,
    ) -> ControlledPredictionContent:
        """Return a controlled prediction response for the given category.

        Args:
            category: The response category for the controlled prediction.

        Returns:
            ControlledPredictionContent with the appropriate response for the category.
        """
        # Find the response for this category
        for prediction_marker, (
            response,
            response_category,
        ) in PREDICTION_RESPONSES.items():
            if response_category == category:
                return ControlledPredictionContent(
                    content=response,
                    response_category=category,
                )

        # Fallback (should not happen if category is valid)
        structlogger.error(
            "copilot_response_handler.respond_to_controlled_prediction.no_response_found",
            event_info=f"No response found for category: {category}",
            category=category,
        )
        raise ValueError(f"No response found for category: {category}")

    @staticmethod
    async def respond_to_commit(
        git_service: Any, commit_sha: str, training_success: bool = True
    ) -> CommitInformationContent:
        """Create a response by fetching commit information.

        Args:
            git_service: The git service to fetch commit info from.
            commit_sha: The commit SHA to fetch information for.
            training_success: Whether training was successful (default: True).

        Returns:
            CommitInformationContent with the commit data formatted for SSE streaming.

        Raises:
            Exception: If fetching commit info fails.
        """
        commit_info_dict = await git_service.get_commit_info(commit_sha)
        commit_info_dict["training_success"] = training_success
        return CommitInformationContent(commit=commit_info_dict)

    @staticmethod
    def get_copilot_redacted_message() -> str:
        """Get the redacted message for copilot responses.

        Returns:
            Redacted message for copilot responses.
        """
        return COPILOT_REDACTED_MESSAGE

    # Content extraction methods -------------------------------------------------------

    def extract_references(self, documents: List[Document]) -> ReferenceSection:
        """Extract references from the generated responses content.

        This method performs regex matching to find markdown links in the format:
        [text](url).

        The matched links are validated against the provided documents, and a
        ReferenceSection is returned with valid references.

        Args:
            documents: List of Document objects to match URLs against

        Returns:
            ReferenceSection containing reference entries ordered by reference text.

        Raises:
            RuntimeError: If called before the copilot response has been run.
        """
        if not self.has_been_run():
            message = (
                "`extract_references` can only be called after the copilot response "
                "has been run."
            )
            structlogger.error(
                "copilot_response_handler.extract_references.copilot_not_run",
                event_info=message,
            )
            raise RuntimeError(message)

        all_content = self.extract_text_from_generated_responses()

        # Return empty reference section when content is empty (e.g., for controlled
        # predictions that don't contain references)
        if not all_content:
            structlogger.debug(
                "copilot_response_handler.extract_references.empty_content",
                event_info=(
                    "No content available for reference extraction. "
                    "Returning empty reference section.",
                ),
            )
            return ReferenceSection(references=[])

        # Find all matches in the buffered content
        matches = re.findall(INLINE_CITATION_PATTERN, all_content)

        # No matches found, return empty reference section
        if not matches:
            return ReferenceSection(references=[])

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
                        "section."
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

    @abstractmethod
    def extract_text_from_generated_responses(self) -> str:
        """Extract and join all content from processed generated responses.

        Returns:
            str: Concatenated text from all generated content responses.
        """
        pass

    def extract_response_category(self) -> ResponseCategory:
        """Extract the response category from the handler.

        This method categorizes the response as a whole. It returns controlled
        prediction categories or exceptions immediately when found, otherwise
        returns COPILOT for all regular content.

        Returns:
            ResponseCategory: Controlled prediction categories or EXCEPTION if found,
            otherwise COPILOT for all regular content.
        """
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
                # Return controlled predictions immediately
                if response.response_category in CONTROLLED_PREDICTION_CATEGORIES:
                    return response.response_category

                # Return exceptions immediately (they take priority over content)
                if response.response_category == ResponseCategory.EXCEPTION:
                    return response.response_category

        # Return COPILOT for all regular content (DELTA, START, END, etc.)
        return ResponseCategory.COPILOT

    def extract_final_plan(self) -> Optional[List[TodoItem]]:
        """Extract the final task plan captured during streaming.

        This method returns the plan state that was captured during streaming.
        The plan is updated each time a TodoPlanUpdate event is received from
        the planning tools queue.

        Returns:
            List of TodoItem objects representing the final plan state,
            or None if no plan was created during the stream.

        Note:
            The default implementation returns None. Subclasses that support
            task planning should override this method.
        """
        return None
