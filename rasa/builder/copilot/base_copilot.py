import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from rasa.builder.copilot.models import (
    ChatMessage,
    CopilotChatMessage,
    CopilotContext,
    CopilotGenerationContext,
    EventContent,
    InternalCopilotRequestChatMessage,
    ResponseCategory,
    UsageStatistics,
    UserChatMessage,
)
from rasa.builder.copilot.utils import filter_chat_history_messages
from rasa.builder.document_retrieval.models import Document
from rasa.builder.shared.tracker_context import TrackerContext

if TYPE_CHECKING:
    from rasa.builder.copilot.response_handling import (
        message_classifier_response_handler as mc_handler,
    )
    from rasa.builder.copilot.response_handling.agent_copilot_response_handler import (
        AgentCopilotResponseHandler,
    )
    from rasa.builder.copilot.response_handling.legacy_copilot_response_handler import (
        LegacyCopilotResponseHandler,
    )

    MessageClassifierResponseHandler = mc_handler.MessageClassifierResponseHandler

    CopilotResponseHandler = Union[
        AgentCopilotResponseHandler,
        LegacyCopilotResponseHandler,
        MessageClassifierResponseHandler,
    ]


class BaseCopilot(ABC):
    """Base class for copilot implementations."""

    @property
    @abstractmethod
    def usage_statistics(self) -> UsageStatistics:
        """Get usage statistics for the copilot.

        Returns:
            UsageStatistics object tracking token usage and costs.
        """
        pass

    @abstractmethod
    async def generate_response(
        self, context: CopilotContext
    ) -> Tuple["CopilotResponseHandler", CopilotGenerationContext]:
        """Generate a response from the copilot.

        Args:
            context: The context of the copilot

        Returns:
            A tuple containing the async response stream and a
            CopilotGenerationContext object with relevant documents and messages
        """
        pass

    @property
    @abstractmethod
    def llm_config(self) -> Dict[str, Any]:
        """The LLM config used to generate the response."""
        pass

    # HELPERS

    def _create_chat_history_messages(
        self, chat_history: List[UserChatMessage | CopilotChatMessage]
    ) -> List[Dict[str, Any]]:
        """Filter and convert past messages to OpenAI format.

        Excludes guardrails policy violations and non-user/copilot messages.

        Args:
            chat_history: List of chat messages to filter and convert.

        Returns:
            List of messages in OpenAI format
        """
        filtered_messages = filter_chat_history_messages(
            chat_history,
            excluded_response_categories=[ResponseCategory.GUARDRAILS_POLICY_VIOLATION],
        )
        return [message.build_openai_message() for message in filtered_messages]

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
        """Format conversation history from TrackerContext."""
        conversation_history: Dict[str, Any] = {
            "conversation_history": [],
        }

        if not tracker_context or not tracker_context.conversation_turns:
            return json.dumps(conversation_history, ensure_ascii=False, indent=2)

        conversation_history["conversation_history"] = (
            tracker_context.formatted_conversation_turns
        )
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
        message: UserChatMessage | CopilotChatMessage,
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
        # TODO: (agent-sdk) Log tracker event attachments to Langfuse
        #       only in the case of the User chat message.
        return message.get_content_blocks_by_type(EventContent)
