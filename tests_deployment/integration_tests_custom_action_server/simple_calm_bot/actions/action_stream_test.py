"""Streaming recent-transactions actions for CALM bot integration tests.

These actions exercise the streaming path (stream_chunk / stream_end) so that
integration tests can validate that:
  - partial text chunks arrive in order on the REST SSE stream, and
  - the graceful-fallback path (no ?stream=true) assembles the full response.

The "recent transactions" topic is intentionally distinct from the existing
``list_contacts`` flow so that the CALM LLMCommandGenerator routes the two
flows without ambiguity.
"""
import asyncio
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Tokens for a short transaction summary.  They concatenate to
# EXPECTED_STREAMED_TEXT without any joining character.
STREAMING_TOKENS: List[str] = [
    "Your",
    " last",
    " 3",
    " transactions:",
    " €50",
    " to",
    " John,",
    " €30",
    " to",
    " Jane,",
    " €20",
    " to",
    " Jack.",
]
EXPECTED_STREAMED_TEXT: str = "".join(STREAMING_TOKENS)

# Skill quick-replies used by ActionStreamAgentSkills after the intro text.
AGENT_SKILL_BUTTONS: List[Dict[str, Any]] = [
    {"title": "Transfer money", "payload": "transfer money"},
    {"title": "List my contacts", "payload": "list my contacts"},
    {"title": "List my reminders", "payload": "list my reminders"},
]


class ActionStreamRecentTransactions(Action):
    """Streams a summary of recent transactions token-by-token.

    On non-streaming transports stream_end() replays each accumulated token
    via utter_message() so the full response is still delivered.
    """

    def name(self) -> str:
        return "action_stream_recent_transactions"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[Text, Any]]:
        await dispatcher.stream_start()
        for token in STREAMING_TOKENS:
            await dispatcher.stream_chunk(text=token)
            # Small delay so SSE chunks are flushed incrementally during tests.
            await asyncio.sleep(0.01)

        # Flushes accumulated chunks as utter_message() calls on non-streaming
        # transports so the action works correctly on every channel.
        await dispatcher.stream_end()
        return []


class ActionStreamAgentSkills(Action):
    """Streams a short intro then emits quick-reply buttons for each skill.

    The action streams an intro text and a rich-content chunk (buttons)
    listing the available skills.
    """
    def name(self) -> str:
        return "action_stream_agent_skills"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[Text, Any]]:
        await dispatcher.stream_start()
        intro_text = "I can help you with:\n"

        # Rich-content chunk — routed via send_response() on the Rasa side
        # because dispatch_stream_chunk detects "buttons" in RICH_KEYS.
        await dispatcher.stream_chunk(text=intro_text, buttons=AGENT_SKILL_BUTTONS)

        await dispatcher.stream_end()
        return []
