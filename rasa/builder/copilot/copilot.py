import asyncio
import importlib
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import openai
import structlog
from jinja2 import Template
from typing_extensions import AsyncGenerator

from rasa.builder import config
from rasa.builder.config import COPILOT_DOCUMENTATION_SEARCH_QUERY_HISTORY_MESSAGES
from rasa.builder.copilot.constants import (
    COPILOT_LAST_USER_MESSAGE_CONTEXT_PROMPT_FILE,
    COPILOT_PROMPTS_DIR,
    COPILOT_PROMPTS_FILE,
    COPILOT_TRAINING_ERROR_HANDLER_PROMPT_FILE,
    ROLE_COPILOT,
    ROLE_COPILOT_INTERNAL,
    ROLE_USER,
)
from rasa.builder.copilot.exceptions import CopilotStreamError
from rasa.builder.copilot.models import (
    ChatMessage,
    CopilotChatMessage,
    CopilotContext,
    CopilotGenerationContext,
    CopilotSystemMessage,
    EventContent,
    FileContent,
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
from rasa.builder.shared.tracker_context import TrackerContext
from rasa.builder.telemetry.copilot_langfuse_telemetry import CopilotLangfuseTelemetry
from rasa.shared.constants import PACKAGE_NAME

structlogger = structlog.get_logger()


class Copilot:
    def __init__(self) -> None:
        self._inkeep_document_retrieval = InKeepDocumentRetrieval()

        # Load both prompt templates up-front
        self._system_message_prompt_template = Template(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{COPILOT_PROMPTS_DIR}",
                COPILOT_PROMPTS_FILE,
            )
        )
        self._last_user_message_context_prompt_template = Template(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{COPILOT_PROMPTS_DIR}",
                COPILOT_LAST_USER_MESSAGE_CONTEXT_PROMPT_FILE,
            )
        )
        self._training_error_handler_prompt_template = Template(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{COPILOT_PROMPTS_DIR}",
                COPILOT_TRAINING_ERROR_HANDLER_PROMPT_FILE,
            )
        )

        # The final stream chunk includes usage statistics.
        self.usage_statistics = UsageStatistics(
            input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
            output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
            cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
        )

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
            # TODO: Log documentation retrieval to Langfuse
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
    ) -> tuple[AsyncGenerator[str, None], CopilotGenerationContext]:
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

        # TODO: Delete this after Langfuse is implemented
        support_evidence = CopilotGenerationContext(
            relevant_documents=relevant_documents,
            system_message=messages[0],
            chat_history=messages[1:-1],
            last_user_message=messages[-1],
            tracker_event_attachments=tracker_event_attachments,
        )

        return (
            self._stream_response(messages),
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

    def _create_chat_history_messages(
        self, chat_history: List[Union[UserChatMessage, CopilotChatMessage]]
    ) -> List[Dict[str, Any]]:
        """Filter and convert past messages to OpenAI format.

        Excludes guardrails policy violations and non-user/copilot messages.

        Args:
            chat_history: List of chat messages to filter and convert.

        Returns:
            List of messages in OpenAI format
        """
        filtered_messages = []

        for message in chat_history:
            if (
                message.response_category
                != ResponseCategory.GUARDRAILS_POLICY_VIOLATION
                and message.role in [ROLE_USER, ROLE_COPILOT]
            ):
                filtered_messages.append(message)

        return [message.build_openai_message() for message in filtered_messages]

    def _process_latest_message(
        self,
        latest_message: Any,
        context: CopilotContext,
        relevant_documents: List[Document],
    ) -> Dict[str, Any]:
        """Process the latest message and convert it to OpenAI format.

        Args:
            latest_message: The most recent message from the chat history.
            context: The copilot context containing conversation state.
            relevant_documents: List of relevant documents for context.

        Returns:
            Message in OpenAI format.

        Raises:
            ValueError: If the message type is not supported.
        """
        if isinstance(latest_message, UserChatMessage):
            tracker_event_attachments = latest_message.get_content_blocks_by_type(
                EventContent
            )
            rendered_prompt = self._render_last_user_message_context_prompt(
                context, relevant_documents, tracker_event_attachments
            )
            return latest_message.build_openai_message(prompt=rendered_prompt)

        elif isinstance(latest_message, InternalCopilotRequestChatMessage):
            rendered_prompt = self._render_training_error_handler_prompt(
                latest_message, relevant_documents
            )
            return latest_message.build_openai_message(prompt=rendered_prompt)

        else:
            raise ValueError(f"Unexpected message type: {type(latest_message)}")

    def _create_system_message(self) -> Dict[str, Any]:
        """Create the system message for the conversation.

        Returns:
            System message in OpenAI format with rendered prompt template.
        """
        rendered_prompt = self._system_message_prompt_template.render()
        return CopilotSystemMessage().build_openai_message(prompt=rendered_prompt)

    def _render_last_user_message_context_prompt(
        self,
        context: CopilotContext,
        relevant_documents: List[Document],
        tracker_event_attachments: List[EventContent],
    ) -> str:
        # Format relevant documentation
        documents = [doc.model_dump() for doc in relevant_documents]
        # Format conversation history
        conversation = self._format_conversation_history(context.tracker_context)
        # Format current state
        current_state = self._format_current_state(context.tracker_context)
        # Format tracker events
        attachments = self._format_tracker_event_attachments(tracker_event_attachments)

        rendered_prompt = self._last_user_message_context_prompt_template.render(
            current_conversation=conversation,
            current_state=current_state,
            assistant_logs=context.assistant_logs,
            assistant_files=context.assistant_files,
            documentation_results=documents,
            attachments=attachments,
        )
        return rendered_prompt

    def _render_training_error_handler_prompt(
        self,
        internal_request_message: InternalCopilotRequestChatMessage,
        relevant_documents: List[Document],
    ) -> str:
        """Render the training error handler prompt with documentation and context.

        Args:
            internal_request_message: Internal request message.
            context: The copilot context.
            relevant_documents: List of relevant documents for context.

        Returns:
            Rendered prompt string for training error analysis.
        """
        modified_files_dicts: Dict[str, str] = {
            file.file_path: file.file_content
            for file in internal_request_message.get_content_blocks_by_type(FileContent)
        }
        rendered_prompt = self._training_error_handler_prompt_template.render(
            logs=internal_request_message.get_flattened_log_content(),
            modified_files=modified_files_dicts,
            documentation_results=self._format_documents(relevant_documents),
        )

        return rendered_prompt

    @staticmethod
    def _create_documentation_search_query(context: CopilotContext) -> str:
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
                Copilot._format_internal_message_for_query_chat_history(message)
                if isinstance(message, InternalCopilotRequestChatMessage)
                else Copilot._format_normal_message_for_query_chat_history(message)
            )
            query_chat_history.insert(0, f"{prefix}: {text}")

        return "\n".join(query_chat_history)

    @staticmethod
    def _format_documents(results: List[Document]) -> Optional[str]:
        """Format documentation search results as JSON dump to be used in the prompt."""
        # We want the special message that indicates no relevant documentation source
        # found if there are no results.
        if not results:
            return None

        formatted_results: Dict[str, Any] = {
            "sources": [
                {
                    # Start the reference from 1, not 0.
                    "idx": idx + 1,
                    "title": result.title,
                    "url": result.url,
                    "content": result.content,
                }
                for idx, result in enumerate(results)
            ]
        }
        return json.dumps(formatted_results, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_conversation_history(tracker_context: Optional[TrackerContext]) -> str:
        """Format conversation history from TrackerContext using nested turn structure.

        Args:
            tracker_context: The TrackerContext containing conversation data.

        Returns:
            A JSON string with turns containing user_input, assistant_response,
            and context.

        Example:
            ```json
            {
                "conversation_history": [
                    {
                        "turn_id": 1,
                        "USER": {
                            "text": "I want to transfer money",
                            "predicted_commands": ["start flow", "set slot", ...]
                        },
                        "BOT": [
                            {"text": "How much would you like to transfer?"}
                        ],
                        "other_tracker_events": [
                            {
                                "event": "action_executed",
                                "data": {"action_name": "action_ask_amount"}
                            },
                            {
                                "event": "slot_set",
                                "data": {
                                    "slot_name": "amount_of_money",
                                    "slot_value": 100,
                                },
                            }
                        ]
                    }
                ]
            }
            ```
        """
        conversation_history: Dict[str, Any] = {
            "conversation_history": [],
        }

        if not tracker_context or not tracker_context.conversation_turns:
            return json.dumps(conversation_history, ensure_ascii=False, indent=2)

        conversation_turns: List[Dict[str, Any]] = []
        user_prefix = "USER"
        assistant_prefix = "BOT"

        for turn_idx, turn in enumerate(tracker_context.conversation_turns, 1):
            turn_data: Dict[str, Any] = {"turn_id": turn_idx}

            # Add user if present
            if turn.user_message:
                turn_data[user_prefix] = {
                    "text": turn.user_message.text,
                    "predicted_commands": turn.user_message.predicted_commands,
                }

            # Add assistant messages if present
            if turn.assistant_messages:
                turn_data[assistant_prefix] = [
                    {"text": assistant_message.text}
                    for assistant_message in turn.assistant_messages
                ]

            # Add other tracker events
            if turn.context_events:
                other_events = [event.model_dump() for event in turn.context_events]
                turn_data["other_tracker_events"] = other_events

            conversation_turns.append(turn_data)

        conversation_history["conversation_history"] = conversation_turns
        return json.dumps(conversation_history, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_current_state(tracker_context: Optional[TrackerContext]) -> str:
        """Format current state from TrackerContext for LLM consumption.

        Args:
            tracker_context: The TrackerContext containing current state data.

        Returns:
            A JSON string containing the current state information.
        """
        if not tracker_context or not tracker_context.current_state:
            return json.dumps({}, ensure_ascii=False, indent=2)
        current_state = tracker_context.current_state.model_dump()
        return json.dumps(current_state, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_normal_message_for_query_chat_history(
        message: Union[UserChatMessage, CopilotChatMessage],
    ) -> str:
        """Format normal message for query chat history."""
        return f"{message.get_flattened_text_content()}"

    @staticmethod
    def _format_internal_message_for_query_chat_history(
        message: InternalCopilotRequestChatMessage,
    ) -> str:
        """Format internal copilot request message for query chat history."""
        text_content = message.get_flattened_text_content()
        log_content = message.get_flattened_log_content()
        if text_content and log_content:
            return f"{text_content}\nLogs: {log_content}"
        elif text_content:
            return text_content
        elif log_content:
            return f"Logs: {log_content}"
        else:
            return ""

    @staticmethod
    def _format_tracker_event_attachments(events: List[EventContent]) -> Optional[str]:
        """Format tracker events as JSON dump to be used in the prompt."""
        # We don't want to display the attachment sectin in the last user message
        # context prompt if there are no attachments.
        if not events:
            return None
        # If there are attachments, return the formatted JSON dump.
        return json.dumps(
            [event_content.model_dump() for event_content in events],
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def _extract_tracker_event_attachments(message: ChatMessage) -> List[EventContent]:
        """Extract the tracker event attachments from the message."""
        if not isinstance(message, UserChatMessage):
            return []
        # TODO: Log tracker event attachments to Langfuse only in the case of the
        #       User chat message.
        return message.get_content_blocks_by_type(EventContent)
