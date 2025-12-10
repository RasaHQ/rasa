import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple

import openai
import structlog
from jinja2 import Template
from typing_extensions import AsyncGenerator

from rasa.builder import config
from rasa.builder.config import COPILOT_DOCUMENTATION_SEARCH_QUERY_HISTORY_MESSAGES
from rasa.builder.copilot.base_copilot import BaseCopilot
from rasa.builder.copilot.constants import (
    COPILOT_PROMPTS_DIR,
    COPILOT_PROMPTS_FILE,
    ROLE_COPILOT,
    ROLE_COPILOT_INTERNAL,
    ROLE_USER,
)
from rasa.builder.copilot.copilot_response_handler import (
    CopilotResponseHandler,
)
from rasa.builder.copilot.exceptions import CopilotStreamError
from rasa.builder.copilot.models import (
    CopilotChatMessage,
    CopilotContext,
    CopilotGenerationContext,
    CopilotSystemMessage,
    InternalCopilotRequestChatMessage,
    ResponseCategory,
    UsageStatistics,
    UserChatMessage,
)
from rasa.builder.document_retrieval.inkeep_document_retrieval import (
    InKeepDocumentRetrieval,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.exceptions import (
    DocumentRetrievalError,
)
from rasa.builder.telemetry.copilot_langfuse_telemetry import CopilotLangfuseTelemetry
from rasa.utils.io import read_text_from_package

structlogger = structlog.get_logger()


def _system_message_prompt_template() -> Template:
    return Template(read_text_from_package(COPILOT_PROMPTS_DIR, COPILOT_PROMPTS_FILE))


class LegacyCopilot(BaseCopilot):
    """Legacy copilot implementation using OpenAI API."""

    def __init__(self) -> None:
        """Initialize the legacy copilot."""
        self._inkeep_document_retrieval = InKeepDocumentRetrieval()

        # The final stream chunk includes usage statistics.
        self._usage_statistics = UsageStatistics(
            input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
            output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
            cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
        )

    @property
    def usage_statistics(self) -> UsageStatistics:
        """Get usage statistics for the copilot."""
        return self._usage_statistics

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[openai.AsyncOpenAI, None]:
        """Create a fresh OpenAI client, yield it, and always close it."""
        client = openai.AsyncOpenAI(timeout=config.OPENAI_TIMEOUT)
        try:
            yield client
        except Exception as e:
            structlogger.error("copilot.llm_client_error", error=str(e))
            raise
        finally:
            try:
                await client.close()
            except Exception as exc:
                # Closing should not break request processing, but we log it
                structlogger.warning(
                    "copilot.llm_client_close_error",
                    event_info="Failed to close OpenAI client cleanly.",
                    error=str(exc),
                )

    @property
    def llm_config(self) -> Dict[str, Any]:
        """The LLM config used to generate the response."""
        return {
            "model": config.OPENAI_MODEL,
            "temperature": config.OPENAI_TEMPERATURE,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    async def search_rasa_documentation(
        self,
        context: CopilotContext,
    ) -> List[Document]:
        """Search Rasa documentation for relevant information.

        Args:
            context: The context of the copilot.

        Returns:
            A list of Document objects. Empty list is returned if the search fails.
        """
        try:
            query = self._create_documentation_search_query(context)
            documents = await self._inkeep_document_retrieval.retrieve_documents(query)
            # TODO: (agent-sdk) Log documentation retrieval to Langfuse
            return documents
        except DocumentRetrievalError as e:
            structlogger.error(
                "copilot.search_rasa_documentation.error",
                event_info=(
                    f"Copilot: Searching Rasa documentation for query '{query}' "
                    f"failed with the following error: {e}. Returning empty list."
                ),
                query=query,
                error=str(e),
            )
            return []

    async def generate_response(
        self,
        context: CopilotContext,
    ) -> Tuple[CopilotResponseHandler, CopilotGenerationContext]:
        """Generate a response from the copilot.

        This method performs document retrieval and response generation as a single
        atomic operation. The returned documents are the supporting evidence used
        to generate the response, ensuring consistency between the response content
        and its sources.

        Args:
            context: The context of the copilot.

        Returns:
            A tuple containing the async response stream and a
            CopilotGenerationContext object with relevant documents, and all the
            messages used to generate the response.

        Raises:
            CopilotStreamError: If the stream fails.
            Exception: If an unexpected error occurs.
        """
        relevant_documents = await self.search_rasa_documentation(context)
        tracker_event_attachments = self._extract_tracker_event_attachments(
            context.copilot_chat_history[-1]
        )
        messages = await self._build_messages(context, relevant_documents)

        # TODO: (agent-sdk) Delete this after Langfuse is implemented
        support_evidence = CopilotGenerationContext(
            relevant_documents=relevant_documents,
            system_message=messages[0],
            chat_history=messages[1:-1],
            last_user_message=messages[-1],
            tracker_event_attachments=tracker_event_attachments,
        )

        copilot_response_handler = CopilotResponseHandler(
            response_stream=self._stream_response(messages),
            rolling_buffer_size=config.COPILOT_HANDLER_ROLLING_BUFFER_SIZE,
        )

        return (
            copilot_response_handler,
            support_evidence,
        )

    @CopilotLangfuseTelemetry.trace_copilot_streaming_generation
    async def _stream_response(
        self, messages: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """Stream markdown chunks one by one."""
        self.usage_statistics.reset()

        try:
            async with self._get_client() as client:
                stream = await client.chat.completions.create(
                    messages=messages,
                    **self.llm_config,
                )
                async for chunk in stream:  # type: ignore[attr-defined]
                    # The final chunk, which contains the usage statistics,
                    # arrives with an empty `choices` list.
                    if not chunk.choices:
                        self.usage_statistics.update_from_stream_chunk(chunk)
                        # Nothing to yield – continue to the next chunk.
                        continue

                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content

        except openai.OpenAIError as e:
            structlogger.exception("copilot.stream_response.api_error", error=str(e))
            raise CopilotStreamError(
                "Failed to stream response from OpenAI API."
            ) from e
        except asyncio.TimeoutError as e:
            structlogger.exception(
                "copilot.stream_response.timeout_error", error=str(e)
            )
            raise CopilotStreamError("Request to OpenAI API timed out.") from e
        except Exception as e:
            structlogger.exception(
                "copilot.stream_response.unexpected_error", error=str(e)
            )
            raise

    async def _build_messages(
        self,
        context: CopilotContext,
        relevant_documents: List[Document],
    ) -> List[Dict[str, Any]]:
        """Build the complete message list for the OpenAI API.

        Args:
            context: The context of the copilot.
            relevant_documents: The relevant documents to use in the context.

        Returns:
            A list of messages in OpenAI format.
        """
        if not context.copilot_chat_history:
            return []

        past_messages = self._create_chat_history_messages(
            context.copilot_chat_history[:-1]
        )

        latest_message = self._process_latest_message(
            context.copilot_chat_history[-1], context, relevant_documents
        )
        system_message = self._create_system_message()

        return [system_message, *past_messages, latest_message]

    def _create_system_message(self) -> Dict[str, Any]:
        """Create the system message for the conversation.

        Returns:
            System message in OpenAI format with rendered prompt template.
        """
        rendered_prompt = _system_message_prompt_template().render()
        return CopilotSystemMessage().build_openai_message(prompt=rendered_prompt)

    @classmethod
    def _create_documentation_search_query(cls, context: CopilotContext) -> str:
        """Format chat messages between user and copilot for documentation search.

        Filters out guardrails policy violations and only includes messages with
        USER or COPILOT roles, then takes the last N relevant messages.
        """
        role_to_prefix = {
            ROLE_USER: "User",
            ROLE_COPILOT: "Assistant",
            ROLE_COPILOT_INTERNAL: "User",
        }
        allowed_message_types = (
            UserChatMessage,
            InternalCopilotRequestChatMessage,
            CopilotChatMessage,
        )

        query_chat_history: List[str] = []

        for message in reversed(context.copilot_chat_history):
            if (
                message.response_category
                == ResponseCategory.GUARDRAILS_POLICY_VIOLATION
                or not isinstance(message, allowed_message_types)
            ):
                continue

            if (
                len(query_chat_history)
                >= COPILOT_DOCUMENTATION_SEARCH_QUERY_HISTORY_MESSAGES
            ):
                break

            prefix = role_to_prefix[message.role]
            text = (
                cls._format_internal_message_for_query_chat_history(message)
                if isinstance(message, InternalCopilotRequestChatMessage)
                else cls._format_normal_message_for_query_chat_history(message)
            )
            query_chat_history.insert(0, f"{prefix}: {text}")

        return "\n".join(query_chat_history)
