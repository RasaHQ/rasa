import asyncio
import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, cast

import structlog

from rasa.builder.copilot.constants import ROLE_COPILOT, ROLE_USER
from rasa.builder.copilot.copilot_response_handler import CopilotResponseHandler
from rasa.builder.copilot.models import (
    CopilotChatMessage,
    CopilotContext,
    GeneratedContent,
    ResponseCategory,
)
from rasa.builder.guardrails.clients import GuardrailsClient
from rasa.builder.guardrails.models import (
    GuardrailRequestKey,
    GuardrailResponse,
)
from rasa.builder.guardrails.utils import (
    create_guardrail_request,
)
from rasa.builder.shared.tracker_context import (
    AssistantConversationTurn,
    TrackerContext,
)

if TYPE_CHECKING:
    pass


structlogger = structlog.get_logger()


class GuardrailsPolicyChecker:
    def __init__(self, guardrail_client: GuardrailsClient):
        self.guardrail_client = guardrail_client

    async def check_assistant_chat_for_policy_violations(
        self,
        tracker_context: TrackerContext,
        hello_rasa_user_id: Optional[str],
        hello_rasa_project_id: Optional[str],
        **kwargs: Any,
    ) -> TrackerContext:
        """Return a sanitised TrackerContext with unsafe turns removed.

        Only user messages are moderated - assistant messages are assumed safe.
        LRU cache is used, so each unique user text is checked once.

        Args:
            tracker_context: The TrackerContext containing conversation turns.
            hello_rasa_user_id: The user ID for the conversation.
            hello_rasa_project_id: The project ID for the conversation.
            **kwargs: Additional parameters for the guardrail request.

        Returns:
            TrackerContext with unsafe turns removed.
        """
        # Collect (turn_index, user_text) for all turns with a user message
        items = self._format_user_messages_to_assistant(
            tracker_context.conversation_turns
        )

        flagged_turns = await self._check_user_messages_for_violations(
            items,
            hello_rasa_user_id=hello_rasa_user_id,
            hello_rasa_project_id=hello_rasa_project_id,
            log_prefix="assistant_guardrails",
            **kwargs,
        )

        if not flagged_turns:
            return tracker_context

        structlogger.info(
            "guardrails_policy_checker.assistant_guardrails.turns_flagged",
            count=len(flagged_turns),
            turn_indices=sorted(flagged_turns),
        )

        # Build a TrackerContext with safe turns
        safe_turns: List[AssistantConversationTurn] = [
            turn
            for idx, turn in enumerate(tracker_context.conversation_turns)
            if idx not in flagged_turns
        ]

        new_tracker_context = copy.deepcopy(tracker_context)
        new_tracker_context.conversation_turns = safe_turns
        return new_tracker_context

    async def check_copilot_chat_for_policy_violations(
        self,
        context: CopilotContext,
        hello_rasa_user_id: Optional[str],
        hello_rasa_project_id: Optional[str],
        **kwargs: Any,
    ) -> Optional[GeneratedContent]:
        """Check the copilot chat history for guardrail policy violations.

        Only user messages are moderated – assistant messages are assumed safe.
        LRU cache is used, so each unique user text is checked once.

        Args:
            context: The CopilotContext containing the copilot chat history.
            hello_rasa_user_id: The user ID for the conversation.
            hello_rasa_project_id: The project ID for the conversation.
            **kwargs: Additional parameters for the guardrail request.

        Returns:
            Returns a default violation response if the system flags any user message,
            otherwise return None.
        """
        # Collect (index, text) for user messages; skip ones already marked as
        # violations
        items = self._format_user_messages_to_copilot(context.copilot_chat_history)

        flagged_user_indices = await self._check_user_messages_for_violations(
            items,
            hello_rasa_user_id=hello_rasa_user_id,
            hello_rasa_project_id=hello_rasa_project_id,
            log_prefix="copilot_guardrails",
            **kwargs,
        )

        self._annotate_flagged_user_messages(
            context.copilot_chat_history, flagged_user_indices
        )

        if not flagged_user_indices:
            return None

        # Identify the latest user message index in the current request
        last_user_idx: Optional[int] = None
        for i in range(len(context.copilot_chat_history) - 1, -1, -1):
            if getattr(context.copilot_chat_history[i], "role", None) == ROLE_USER:
                last_user_idx = i
                break

        # Remove flagged user messages and their next copilot messages
        indices_to_remove: Set[int] = set()
        total = len(context.copilot_chat_history)
        for uidx in flagged_user_indices:
            indices_to_remove.add(uidx)
            next_idx = uidx + 1
            if (
                next_idx < total
                and getattr(context.copilot_chat_history[next_idx], "role", None)
                == ROLE_COPILOT
            ):
                indices_to_remove.add(next_idx)

        # Apply sanitization
        filtered_copilot_chat_history = [
            msg
            for i, msg in enumerate(context.copilot_chat_history)
            if i not in indices_to_remove
        ]
        if len(filtered_copilot_chat_history) != len(context.copilot_chat_history):
            structlogger.info(
                "guardrails_policy_checker"
                ".copilot_guardrails"
                ".copilot_chat_history_sanitized",
                removed_indices=sorted(indices_to_remove),
                removed_messages=(
                    len(context.copilot_chat_history)
                    - len(filtered_copilot_chat_history)
                ),
                kept_messages=len(filtered_copilot_chat_history),
            )
            context.copilot_chat_history = filtered_copilot_chat_history

        # Block only if the latest user message in this request was flagged
        if last_user_idx is not None and last_user_idx in flagged_user_indices:
            return CopilotResponseHandler.respond_to_guardrail_policy_violations()

        # Otherwise proceed (following messages are respected)
        return None

    async def _check_user_messages_for_violations(
        self,
        items: List[Tuple[int, str]],
        hello_rasa_user_id: Optional[str],
        hello_rasa_project_id: Optional[str],
        log_prefix: str,
        **kwargs: Any,
    ) -> Set[int]:
        """Run guardrail checks for provided (index, user_text) pairs.

        Args:
            items: List of tuples containing (index, user_text) to check.
            hello_rasa_user_id: The user ID for the conversation.
            hello_rasa_project_id: The project ID for the conversation.
            log_prefix: Prefix for logging messages.
            **kwargs: Additional parameters for the guardrail request.

        Returns:
            A set of indices that were flagged by the guardrails.
        """
        if not items:
            return set()

        # 1) Group indices by logical request key (hashable by value)
        indices_by_key: Dict[GuardrailRequestKey, List[int]] = {}
        for idx, text in items:
            key = GuardrailRequestKey(
                user_text=text,
                hello_rasa_user_id=hello_rasa_user_id or "",
                hello_rasa_project_id=hello_rasa_project_id or "",
                # The client-specific request parameters go in the metadata
                # so that they can be used to create the request
                metadata=kwargs,
            )
            if not key.user_text:
                continue
            indices_by_key.setdefault(key, []).append(idx)

        if not indices_by_key:
            return set()

        # 2) Create one task per logical key
        tasks_by_key: Dict[GuardrailRequestKey, asyncio.Task[GuardrailResponse]] = {}
        for key in indices_by_key:
            request = create_guardrail_request(
                client_type=type(self.guardrail_client),
                user_text=key.user_text,
                hello_rasa_user_id=key.hello_rasa_user_id,
                hello_rasa_project_id=key.hello_rasa_project_id,
                **kwargs,
            )
            tasks_by_key[key] = self.guardrail_client.schedule_check(request)

        # 3) Await unique tasks once
        keys = list(tasks_by_key.keys())
        tasks = [tasks_by_key[k] for k in keys]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 4) Map results back to all corresponding indices
        flagged: Set[int] = set()
        for key, response in zip(keys, responses):
            if isinstance(response, Exception):
                structlogger.warning(
                    f"{log_prefix}.request_failed", error=str(response)
                )
                continue
            # At this point, response is guaranteed to be GuardrailResponse
            # Use typing.cast to explicitly cast the type
            guardrail_response = cast(GuardrailResponse, response)
            if guardrail_response.flagged:
                flagged.update(indices_by_key.get(key, []))

        return flagged

    def _annotate_flagged_user_messages(
        self,
        copilot_chat_history: List[CopilotChatMessage],
        flagged_user_indices: Set[int],
    ) -> None:
        """Mark flagged user messages in-place on the original copilot chat history.

        Args:
            copilot_chat_history: The copilot chat history containing messages.
            flagged_user_indices: Set of indices of user messages that were flagged.
        """
        if not flagged_user_indices:
            return

        total = len(copilot_chat_history)
        for uidx in flagged_user_indices:
            if 0 <= uidx < total and copilot_chat_history[uidx].role == ROLE_USER:
                copilot_chat_history[
                    uidx
                ].response_category = ResponseCategory.GUARDRAILS_POLICY_VIOLATION

    def _format_user_messages_to_assistant(
        self, conversation_turns: List[AssistantConversationTurn]
    ) -> List[Tuple[int, str]]:
        """Collect (turn_index, user_text) tuples for all turns with user messages.

        Args:
            conversation_turns: The list of conversation turns.

        Returns:
            List of tuples containing (turn_index, user_text) for valid user messages.
        """
        items: List[Tuple[int, str]] = []
        for idx, turn in enumerate(conversation_turns):
            user_message = turn.user_message
            if not user_message:
                continue

            text = (user_message.text or "").strip()
            if not text:
                continue

            items.append((idx, text))

        return items

    def _format_user_messages_to_copilot(
        self, copilot_chat_history: List[CopilotChatMessage]
    ) -> List[Tuple[int, str]]:
        """Collect (index, user_text) tuples for all messages with user messages.

        Args:
            copilot_chat_history: The list of messages.
        """
        items: List[Tuple[int, str]] = []
        for idx, message in enumerate(copilot_chat_history):
            if (
                message.response_category
                == ResponseCategory.GUARDRAILS_POLICY_VIOLATION
            ):
                continue
            if message.role != ROLE_USER:
                continue
            formatted_message = message.build_openai_message()
            text = (formatted_message.get("content") or "").strip()
            if not text:
                continue
            items.append((idx, text))

        return items
