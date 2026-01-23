"""Shared utilities for copilot functionality."""

from typing import List, Optional

from rasa.builder.copilot.models import (
    ChatMessage,
    CopilotChatMessage,
    ResponseCategory,
    UserChatMessage,
)


def filter_chat_history_messages(
    chat_history: List[ChatMessage],
    excluded_response_categories: Optional[List[ResponseCategory]] = None,
) -> List[UserChatMessage | CopilotChatMessage]:
    """Filter and convert chat history messages.

    This method will only return UserChatMessage and CopilotChatMessage messages.

    Args:
        chat_history: List of chat messages to filter and convert.
        excluded_response_categories: List of response categories to exclude.
            Defaults to [ResponseCategory.GUARDRAILS_POLICY_VIOLATION].
        allowed_roles: List of roles to keep. Defaults to ["user", "copilot"].

    Returns:
        List of messages in OpenAI format.
    """
    filtered_messages: List[UserChatMessage | CopilotChatMessage] = []

    for message in chat_history:
        # Check if message should be excluded based on response category
        if (
            excluded_response_categories is not None
            and message.response_category in excluded_response_categories
        ):
            continue

        # Only process UserChatMessage and CopilotChatMessage
        if not isinstance(message, (UserChatMessage, CopilotChatMessage)):
            continue

        filtered_messages.append(message)

    return filtered_messages
