from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
)

from rasa.builder.copilot.models import CopilotTurnRequest

if TYPE_CHECKING:
    from rasa.builder.copilot.copilot import Copilot
    from rasa.builder.copilot.models import CopilotContext
    from rasa.builder.document_retrieval.inkeep_document_retrieval import (
        InKeepDocumentRetrieval,
    )

import langfuse
import structlog

from rasa.builder.copilot.copilot_response_handler import CopilotResponseHandler
from rasa.builder.copilot.models import (
    EventContent,
    UsageStatistics,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.models import BotFiles
from rasa.builder.shared.tracker_context import TrackerContext

structlogger = structlog.get_logger()


class CopilotLangfuseTelemetry:
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
        langfuse_client = langfuse.get_client()
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
        langfuse_client = langfuse.get_client()
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
        handler: CopilotResponseHandler,
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
        """
        langfuse_client = langfuse.get_client()
        user_message = request.message.get_flattened_text_content()
        tracker_event_attachments = (
            CopilotLangfuseTelemetry._extract_tracker_event_attachments_from_turn(
                request
            )
        )
        response_category = CopilotLangfuseTelemetry._extract_response_category(handler)
        reference_section_entries = CopilotLangfuseTelemetry._extract_references(
            handler, relevant_documents
        )

        # Create a session ID as a composite ID from project id, user id and chat id
        session_id = CopilotLangfuseTelemetry._create_session_id(
            hello_rasa_project_id, user_id, chat_id
        )
        # Use `update_current_trace` to update the top level trace.
        langfuse_client.update_current_trace(
            user_id=user_id,
            session_id=session_id,
            input={
                "message": user_message,
                "tracker_event_attachments": tracker_event_attachments,
            },
            output={
                "answer": CopilotLangfuseTelemetry._full_text(handler),
                "response_category": response_category,
                "references": reference_section_entries,
            },
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
            tags=[response_category] if response_category else [],
        )

    @staticmethod
    def trace_copilot_streaming_generation(
        func: Callable[..., AsyncGenerator[str, None]],
    ) -> Callable[..., AsyncGenerator[str, None]]:
        """Custom decorator for tracing async streaming of the Copilot's LLM generation.

        This decorator handles Langfuse tracing for async streaming of the Copilot's LLM
        generation by manually managing the generation span and updating it with usage
        statistics after the stream completes.
        """

        @wraps(func)
        async def wrapper(
            self: "Copilot", messages: List[Dict[str, Any]]
        ) -> AsyncGenerator[str, None]:
            langfuse_client = langfuse.get_client()

            with langfuse_client.start_as_current_generation(
                name=f"{self.__class__.__name__}.{func.__name__}",
                input={"messages": messages},
            ) as generation:
                output = []
                # Call the original streaming function and start capturing the output
                async for chunk in func(self, messages):
                    output.append(chunk)
                    yield chunk

                # Update the span's model parameters and output after streaming is
                # complete
                generation.update(
                    model_parameters=self.llm_config, output="".join(output)
                )

                # Update the span's usage statistics after streaming is complete
                if self.usage_statistics:
                    CopilotLangfuseTelemetry._update_generation_span_with_usage_statistics(
                        generation, self.usage_statistics
                    )

        return wrapper

    @staticmethod
    def trace_document_retrieval_generation(
        func: Callable[..., Any],
    ) -> Callable[..., Any]:
        """Custom decorator for tracing document retrieval generation with Langfuse.

        This decorator handles Langfuse tracing for document retrieval API calls
        by manually managing the generation span and updating it with usage statistics.
        """

        @wraps(func)
        async def wrapper(
            self: "InKeepDocumentRetrieval",
            query: str,
            temperature: float,
            timeout: float,
        ) -> Any:
            langfuse_client = langfuse.get_client()

            with langfuse_client.start_as_current_generation(
                name=f"{self.__class__.__name__}.{func.__name__}",
                input={
                    "query": query,
                    "temperature": temperature,
                    "timeout": timeout,
                },
            ) as generation:
                # Call the original function
                response = await func(self, query, temperature, timeout)

                # Update the span with response content
                generation.update(
                    output=response,
                    model_parameters={
                        "temperature": str(temperature),
                        "timeout": str(timeout),
                    },
                )

                # Update usage statistics if available
                usage_statistics = UsageStatistics.from_chat_completion_response(
                    response
                )
                if usage_statistics:
                    CopilotLangfuseTelemetry._update_generation_span_with_usage_statistics(
                        generation, usage_statistics
                    )

                return response

        return wrapper

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
    def _extract_response_category(handler: CopilotResponseHandler) -> Optional[str]:
        """Extract the response category from the response handler.

        Args:
            handler: The response handler containing generated response.

        Returns:
            The response category of the first generated response, or None if no
            responses.
        """
        if not handler.generated_responses:
            return None
        # The handler contains multiple chunks of one response. We use the first chunk's
        # response category.
        return handler.generated_responses[0].response_category.value

    @staticmethod
    def _full_text(handler: CopilotResponseHandler) -> str:
        """Extract full text from the response handler.

        Args:
            handler: The response handler containing generated responses.

        Returns:
            The concatenated content of all generated responses.
        """
        return "".join(
            response.content
            for response in handler.generated_responses
            if getattr(response, "content", None)
        )

    @staticmethod
    def _extract_references(
        handler: CopilotResponseHandler,
        relevant_documents: list[Document],
    ) -> List[Dict[str, Any]]:
        """Extract reference entries from the response handler.

        Args:
            handler: The response handler containing generated responses.
            relevant_documents: The relevant documents used to generate the response.

        Returns:
            A list of reference entries in dictionary format.
        """
        if not relevant_documents:
            return []

        reference_entries: list[Dict[str, Any]] = []
        reference_section = handler.extract_references(relevant_documents)
        for reference_entry in reference_section.references:
            reference_entries.append(
                reference_entry.model_dump(
                    exclude={"response_category", "response_completeness"}
                )
            )

        return reference_entries

    @staticmethod
    def _update_generation_span_with_usage_statistics(
        generation_span: langfuse.LangfuseGeneration,
        usage_statistics: UsageStatistics,
    ) -> None:
        """Update the generation span with the usage statistics.

        Args:
            generation_span: The generation span.
            usage_statistics: The usage statistics of the generation.
        """
        generation_span.update(
            usage_details={
                "input_non_cached_usage": (
                    usage_statistics.non_cached_prompt_tokens or 0
                ),
                "input_cached_usage": usage_statistics.cached_prompt_tokens or 0,
                "output_usage": usage_statistics.completion_tokens or 0,
                "total": usage_statistics.total_tokens or 0,
            },
            cost_details={
                "input_non_cached_cost": usage_statistics.non_cached_cost or 0,
                "input_cached_cost": usage_statistics.cached_cost or 0,
                "output_cost": usage_statistics.output_cost or 0,
                "total": usage_statistics.total_cost or 0,
            },
            model=usage_statistics.model,
        )

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
