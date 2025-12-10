"""MCP tools for interacting with the built assistant.

These tools allow the copilot to test the assistant by sending messages
and verifying the bot's responses through the tracker context.
"""

import uuid
from typing import Any, Dict, List, Optional

import aiohttp
import structlog

from rasa.builder.copilot.mcp_server.models import (
    ConversationTurn,
    TalkToAssistantResponse,
    TrackerContextOutput,
)

structlogger = structlog.get_logger()


async def talk_to_assistant(messages: List[str]) -> TalkToAssistantResponse:
    """Send a sequence of messages to the built assistant and return the conversation.

    This tool sends each message to the assistant one after another,
    waits for the bot's response to each, and then returns the complete
    tracker context so you can verify the conversation flow.

    Args:
        messages: List of user messages to send to the assistant in order

    Returns:
        TalkToAssistantResponse with the conversation results and tracker context
    """
    from rasa.builder import config

    # Generate a unique session ID for this test conversation
    session_id = f"mcp-test-{uuid.uuid4().hex[:8]}"

    # Build the webhook URL - use localhost since MCP runs alongside Sanic
    webhook_url = (
        f"http://{config.BUILDER_SERVER_HOST}:{config.BUILDER_SERVER_PORT}"
        f"/webhooks/rest/webhook"
    )

    structlogger.info(
        "mcp_server.tools.bot_interaction.talk_to_assistant.start",
        event_info="Starting conversation with assistant",
        session_id=session_id,
        message_count=len(messages),
    )

    conversation: List[ConversationTurn] = []

    try:
        async with aiohttp.ClientSession() as session:
            for idx, message in enumerate(messages):
                structlogger.debug(
                    "mcp_server.tools.bot_interaction.sending_message",
                    event_info=f"Sending message {idx + 1}/{len(messages)}",
                    message=message[:100],  # Truncate for logging
                )

                payload = {"sender": session_id, "message": message}

                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        structlogger.error(
                            "mcp_server.tools.bot_interaction.request_failed",
                            status=resp.status,
                            error=error_text,
                        )
                        error_msg = (
                            f"Request failed with status {resp.status}: {error_text}"
                        )
                        return TalkToAssistantResponse(
                            success=False,
                            session_id=session_id,
                            message_count=idx,
                            conversation=conversation,
                            tracker_context=None,
                            error=error_msg,
                        )

                    bot_responses = await resp.json()

                    conversation.append(
                        ConversationTurn(
                            user_message=message,
                            bot_responses=bot_responses,
                        )
                    )

        # After all messages sent, fetch the tracker context
        tracker_context = await _get_tracker_context(session_id)

        structlogger.info(
            "mcp_server.tools.bot_interaction.talk_to_assistant.complete",
            event_info="Conversation completed successfully",
            session_id=session_id,
            message_count=len(messages),
        )

        return TalkToAssistantResponse(
            success=True,
            session_id=session_id,
            message_count=len(messages),
            conversation=conversation,
            tracker_context=tracker_context,
            error=None,
        )

    except aiohttp.ClientError as e:
        structlogger.error(
            "mcp_server.tools.bot_interaction.talk_to_assistant.connection_error",
            event_info="Failed to connect to assistant",
            error=str(e),
        )
        return TalkToAssistantResponse(
            success=False,
            session_id=session_id,
            message_count=len(conversation),
            conversation=conversation,
            tracker_context=None,
            error=f"Connection error: {e!s}. Is the assistant trained and running?",
        )

    except Exception as e:
        structlogger.error(
            "mcp_server.tools.bot_interaction.talk_to_assistant.error",
            event_info="Error during conversation",
            error=str(e),
        )
        return TalkToAssistantResponse(
            success=False,
            session_id=session_id,
            message_count=len(conversation),
            conversation=conversation,
            tracker_context=None,
            error=f"Error: {e!s}",
        )


async def _get_tracker_context(session_id: str) -> Optional[TrackerContextOutput]:
    """Fetch the tracker context for a session via internal API.

    Args:
        session_id: The session ID to fetch the tracker for

    Returns:
        TrackerContextOutput or None if not available
    """
    from rasa.builder import config

    # Use the internal tracker endpoint
    tracker_url = (
        f"http://{config.BUILDER_SERVER_HOST}:{config.BUILDER_SERVER_PORT}"
        f"/api/internal/tracker/{session_id}"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                tracker_url,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data: Dict[str, Any] = await resp.json()
                    return TrackerContextOutput(
                        conversation_turns=data.get("conversation_turns", []),
                        current_state=data.get("current_state", {}),
                    )
                else:
                    structlogger.warning(
                        "mcp_server.tools.bot_interaction.get_tracker_failed",
                        status=resp.status,
                        session_id=session_id,
                    )
                    return None
    except Exception as e:
        structlogger.warning(
            "mcp_server.tools.bot_interaction.get_tracker_error",
            error=str(e),
            session_id=session_id,
        )
        return None
