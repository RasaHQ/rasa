from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rasa.builder.copilot.models import CopilotTurnRequest, EventContent
from rasa.builder.document_retrieval.models import Document
from rasa.builder.models import BotFiles
from rasa.builder.shared.tracker_context import TrackerContext
from rasa.builder.telemetry.langfuse.langfuse_compat import with_langfuse

if TYPE_CHECKING:
    from rasa.builder.copilot import BaseCopilotResponseHandler
    from rasa.builder.copilot.models import CopilotContext


class CopilotEndpointLangfuseTelemetry:
    """Telemetry for copilot endpoint calls (not copilot LLM generation)."""

    @staticmethod
    def trace_copilot_tracker_context(
        tracker_context: Optional[TrackerContext],
        max_conversation_turns: int,
        session_id: str,
    ) -> None:
        """Trace the copilot tracker context.

        Args:
            tracker_context: The tracker context.
            max_conversation_turns: The maximum number of conversation turns to be
                fetched from the tracker.
            session_id: The session ID used to fetch the right tracker.
        """
        with with_langfuse() as lf:
            if not lf:
                return
            langfuse_client = lf.get_client()
            # Use `update_current_span` to update the current span of the trace.
            langfuse_client.update_current_span(
                output={
                    "tracker_context": (
                        tracker_context.model_dump() if tracker_context else None
                    ),
                },
                metadata={
                    "max_conversation_turns": max_conversation_turns,
                    "session_id": session_id,
                },
            )

    @staticmethod
    def trace_copilot_relevant_assistant_files(
        relevant_assistant_files: BotFiles,
    ) -> None:
        """Trace the copilot relevant assistant files.

        Args:
            relevant_assistant_files: The relevant assistant files.
        """
        with with_langfuse() as lf:
            if not lf:
                return
            langfuse_client = lf.get_client()
            # Use `update_current_span` to update the current span of the trace.
            langfuse_client.update_current_span(
                output={
                    "relevant_assistant_files": relevant_assistant_files,
                },
            )

    @staticmethod
    def setup_copilot_endpoint_call_trace_attributes(
        hello_rasa_project_id: str,
        chat_id: str,
        user_id: str,
        request: CopilotTurnRequest,
        handler: "BaseCopilotResponseHandler",
        relevant_documents: list[Document],
        copilot_context: "CopilotContext",
    ) -> None:
        """Set up the current langfuse trace with project and user context.

        Args:
            hello_rasa_project_id: The Hello Rasa project ID.
            chat_id: The chat/conversation ID.
            user_id: The user ID.
            request: The parsed CopilotTurnRequest object.
            handler: The response handler containing generated responses.
            relevant_documents: The relevant documents used to generate the response.
            copilot_context: The copilot context containing additional context.

        Returns:
            None
        """
        with with_langfuse() as lf:
            if not lf:
                return
            langfuse_client = lf.get_client()
            user_message = request.message.get_flattened_text_content()
            tracker_event_attachments = CopilotEndpointLangfuseTelemetry._extract_tracker_event_attachments_from_turn(  # noqa: E501
                request
            )
            response_category = handler.extract_response_category().value
            reference_section_entries = (
                CopilotEndpointLangfuseTelemetry._extract_references(handler)
            )

            # Create a session ID as a composite ID from project id, user id and chat id
            session_id = CopilotEndpointLangfuseTelemetry._create_session_id(
                hello_rasa_project_id, user_id, chat_id
            )
            # Extract the final response (output) for the trace.
            output: Dict[str, Any]
            if (exception_response := handler.extract_exception_response()) is not None:
                output = {
                    "answer": exception_response.content,
                    "response_category": exception_response.response_category.value,
                    "references": [],
                    "original_exception": exception_response.stringified_original_exception,  # noqa: E501
                }
            else:
                output = {
                    "answer": handler.extract_text_from_generated_responses(),
                    "response_category": response_category,
                    "references": reference_section_entries,
                }
            # Use `update_current_trace` to update the top level trace.
            langfuse_client.update_current_trace(
                user_id=user_id,
                session_id=session_id,
                input={
                    "message": user_message,
                    "tracker_event_attachments": tracker_event_attachments,
                },
                output=output,
                metadata={
                    "ids": {
                        "user_id": user_id,
                        "project_id": hello_rasa_project_id,
                        "chat_history_id": chat_id,
                    },
                    "copilot_additional_context": {
                        "relevant_documents": [
                            doc.model_dump() for doc in relevant_documents
                        ],
                        "relevant_assistant_files": copilot_context.assistant_files,
                        "assistant_tracker_context": (
                            copilot_context.tracker_context.model_dump()
                            if copilot_context.tracker_context
                            else None
                        ),
                        "assistant_logs": copilot_context.assistant_logs,
                        "copilot_chat_history": [
                            message.model_dump()
                            for message in copilot_context.copilot_chat_history
                        ],
                    },
                },
                tags=[response_category],
            )

    @staticmethod
    def _extract_tracker_event_attachments_from_turn(
        request: CopilotTurnRequest,
    ) -> list[Dict[str, Any]]:
        """Extract tracker event attachments from the user message.

        Args:
            request: The CopilotTurnRequest object.

        Returns:
            The event content block sent with the user message in the
            dictionary format.
        """
        return [
            attachment.model_dump()
            for attachment in request.message.get_content_blocks_by_type(EventContent)
        ]

    @staticmethod
    def _extract_references(
        handler: "BaseCopilotResponseHandler",
    ) -> List[Dict[str, Any]]:
        """Extract reference entries from the response handler.

        Args:
            handler: The response handler containing generated responses.

        Returns:
            A list of reference entries in dictionary format.
        """
        reference_entries: list[Dict[str, Any]] = []
        reference_section = handler.extract_references()
        for reference_entry in reference_section.references:
            reference_entries.append(
                reference_entry.model_dump(
                    exclude={"response_category", "response_completeness"}
                )
            )

        return reference_entries

    @staticmethod
    def _create_session_id(
        hello_rasa_project_id: str,
        user_id: str,
        chat_id: str,
    ) -> str:
        """Create a session ID as a composite from project id, user id and chat id."""
        pattern = "PID-{project_id}-UID-{user_id}-CID-{chat_id}"
        return pattern.format(
            project_id=hello_rasa_project_id,
            user_id=user_id,
            chat_id=chat_id,
        )
