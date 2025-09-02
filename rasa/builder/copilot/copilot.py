import asyncio
import importlib
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import openai
import structlog
from jinja2 import Template
from typing_extensions import AsyncGenerator

from rasa.builder import config
from rasa.builder.copilot.constants import (
    COPILOT_PROMPTS_DIR,
    COPILOT_PROMPTS_FILE,
    ROLE_COPILOT,
    ROLE_COPILOT_INTERNAL,
    ROLE_SYSTEM,
    ROLE_USER,
)
from rasa.builder.copilot.exceptions import CopilotStreamError
from rasa.builder.copilot.models import (
    CopilotContext,
    ResponseCategory,
    UsageStatistics,
)
from rasa.builder.document_retrieval.inkeep_document_retrieval import (
    InKeepDocumentRetrieval,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.exceptions import (
    DocumentRetrievalError,
)
from rasa.builder.shared.tracker_context import TrackerContext
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

        # The final stream chunk includes usage statistics.
        self.usage_statistics = UsageStatistics()

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
            return await self._inkeep_document_retrieval.retrieve_documents(query)
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
    ) -> tuple[AsyncGenerator[str, None], list[Document], str]:
        """Generate a response from the copilot.

        This method performs document retrieval and response generation as a single
        atomic operation. The returned documents are the supporting evidence used
        to generate the response, ensuring consistency between the response content
        and its sources.

        Args:
            context: The context of the copilot.

        Returns:
            A tuple containing the async response stream, relevant documents used
            as supporting evidence for the generated response, and the prompt used.

        Raises:
            CopilotStreamError: If the stream fails.
            Exception: If an unexpected error occurs.
        """
        relevant_documents = await self.search_rasa_documentation(context)
        system_message = await self._create_system_message(context, relevant_documents)
        chat_history = self._create_chat_history_messages(context)
        messages = [system_message, *chat_history]

        return (
            self._stream_response(messages),
            relevant_documents,
            system_message.get("content", ""),
        )

    async def _stream_response(
        self, messages: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """Stream markdown chunks one by one."""
        self.usage_statistics.reset()

        try:
            async with self._get_client() as client:
                stream = await client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=messages,  # type: ignore
                    temperature=config.OPENAI_TEMPERATURE,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for chunk in stream:
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

    async def _create_system_message(
        self,
        context: CopilotContext,
        relevant_documents: List[Document],
    ) -> Dict[str, Any]:
        """Render the correct Jinja template based on desired output_type."""
        # Format relevant documentation
        documents = [doc.model_dump() for doc in relevant_documents]

        # Format conversation history
        conversation = self._format_conversation_history(context.tracker_context)

        # Format current state
        current_state = self._format_current_state(context.tracker_context)

        # Render template
        rendered_prompt = self._system_message_prompt_template.render(
            current_conversation=conversation,
            current_state=current_state,
            assistant_logs=context.assistant_logs,
            assistant_files=context.assistant_files,
            documentation_results=documents,
        )
        return {"role": ROLE_SYSTEM, "content": rendered_prompt}

    def _create_chat_history_messages(
        self,
        context: CopilotContext,
    ) -> List[Dict[str, Any]]:
        """Create the chat history messages for the copilot.

        Filter out messages with response_category of GUARDRAILS_POLICY_VIOLATION.
        This will filter out all the user messages that were flagged by guardrails, but
        also the copilot messages that were produced by guardrails.
        """
        # Filter out messages with response_category of GUARDRAILS_POLICY_VIOLATION.
        # This will filter out all the user messages flagged by guardrails, but also the
        # copilot messages that were produced.
        return [
            message.to_openai_format()
            for message in context.copilot_chat_history
            if message.response_category != ResponseCategory.GUARDRAILS_POLICY_VIOLATION
        ]

    @staticmethod
    def _create_documentation_search_query(context: CopilotContext) -> str:
        """Format chat messages between user and copilot for documentation search."""
        result = ""
        role_to_prefix = {
            ROLE_USER: "User",
            ROLE_COPILOT: "Assistant",
            ROLE_COPILOT_INTERNAL: "Copilot Internal Request",
        }
        for message in context.copilot_chat_history:
            prefix = role_to_prefix[message.role]
            text = message.get_text_content().strip()
            if text:
                result += f"{prefix}: {text}\n"
            log_content = message.get_log_content().strip()
            if log_content:
                result += f"{prefix}: {log_content}\n"

        return result

    @staticmethod
    def _format_documents(results: List[Document]) -> Optional[str]:
        """Format documentation search results as JSON dump to be used in the prompt."""
        if not results:
            return None

        formatted_results = {
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
