import asyncio
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import structlog

from rasa.builder.config import (
    ASSISTANT_HISTORY_GUARDRAIL_PROJECT_ID,
    COPILOT_HISTORY_GUARDRAIL_PROJECT_ID,
)
from rasa.builder.copilot.constants import ROLE_COPILOT, ROLE_USER
from rasa.builder.copilot.copilot_response_handler import CopilotResponseHandler
from rasa.builder.copilot.models import (
    CopilotChatMessage,
    CopilotContext,
    GeneratedContent,
    ResponseCategory,
)
from rasa.builder.guardrails.models import (
    GuardrailRequestKey,
    GuardrailResponse,
    LakeraGuardrailRequest,
)
from rasa.builder.llm_service import llm_service
from rasa.builder.shared.tracker_context import (
    AssistantConversationTurn,
    TrackerContext,
)

if TYPE_CHECKING:
    from rasa.builder.guardrails.models import GuardrailType


structlogger = structlog.get_logger()


def map_lakera_detector_type_to_guardrail_type(
    lakera_detector_type: str,
) -> Optional["GuardrailType"]:
    """Map a Lakera detector type to a Rasa guardrail type."""
    from rasa.builder.guardrails.models import GuardrailType

    # Check for exact matches first
    LAKERA_DETECTOR_TYPES_2_RASA_GUARDRAIL_TYPES_MAPPING = {
        "prompt_attack": GuardrailType.PROMPT_ATTACK,
        "unknown_links": GuardrailType.MALICIOUS_LINKS,
        "custom": GuardrailType.CUSTOM,
    }

    # Check for exact match first
    if lakera_detector_type in LAKERA_DETECTOR_TYPES_2_RASA_GUARDRAIL_TYPES_MAPPING:
        return LAKERA_DETECTOR_TYPES_2_RASA_GUARDRAIL_TYPES_MAPPING[
            lakera_detector_type
        ]

    # Check for subtypes that start with specific prefixes
    # https://docs.lakera.ai/docs/policies/self-hosted-policies#detectors-section
    if lakera_detector_type.startswith("moderated_content"):
        return GuardrailType.CONTENT_VIOLATION
    if lakera_detector_type.startswith("pii"):
        return GuardrailType.DATA_LEAKAGE

    # If no match found, return OTHER
    return GuardrailType.OTHER


@lru_cache(maxsize=512)
def _schedule_guardrails_check(
    user_text: str,
    hello_rasa_user_id: str,
    hello_rasa_project_id: str,
    lakera_project_id: str,
) -> "asyncio.Task[GuardrailResponse]":
    """Return a cached asyncio.Task that resolves to Lakera's response.

    Args:
        user_text: The user message text to check for policy violations.
        hello_rasa_user_id: The user ID for the conversation.
        hello_rasa_project_id: The project ID for the conversation.
        lakera_project_id: The Lakera project ID to use for this check.

    Returns:
        An asyncio Task that resolves to a GuardrailResponse.
    """
    structlogger.debug("guardrails.cache_miss", text=user_text)

    loop = asyncio.get_running_loop()
    request = LakeraGuardrailRequest(
        lakera_project_id=lakera_project_id,
        hello_rasa_user_id=hello_rasa_user_id,
        hello_rasa_project_id=hello_rasa_project_id,
        messages=[{"role": ROLE_USER, "content": user_text}],
    )

    return loop.create_task(llm_service.guardrails.send_request(request))


async def _detect_flagged_user_indices(
    items: List[Tuple[int, str]],
    *,
    hello_rasa_user_id: Optional[str],
    hello_rasa_project_id: Optional[str],
    lakera_project_id: str,
    log_prefix: str,
) -> Set[int]:
    """Run guardrail checks for provided (index, user_text) pairs.

    Args:
        items: List of tuples containing (index, user_text) to check.
        hello_rasa_user_id: The user ID for the conversation.
        hello_rasa_project_id: The project ID for the conversation.
        lakera_project_id: The Lakera project ID to use for this check.
        log_prefix: Prefix for logging messages.

    Returns:
        A set of indices that were flagged by the guardrails.
    """
    if not items:
        return set()

    # 1) Group indices by logical request key (hashable by value)
    indices_by_key: Dict[GuardrailRequestKey, List[int]] = {}
    for idx, text in items:
        key = GuardrailRequestKey(
            user_text=(text or "").strip(),
            hello_rasa_user_id=hello_rasa_user_id or "",
            hello_rasa_project_id=hello_rasa_project_id or "",
            lakera_project_id=lakera_project_id,
        )
        if not key.user_text:
            continue
        indices_by_key.setdefault(key, []).append(idx)

    if not indices_by_key:
        return set()

    # 2) Create one task per logical key
    tasks_by_key: Dict[GuardrailRequestKey, asyncio.Task[GuardrailResponse]] = {}
    for key in indices_by_key:
        tasks_by_key[key] = _schedule_guardrails_check(
            user_text=key.user_text,
            hello_rasa_user_id=key.hello_rasa_user_id,
            hello_rasa_project_id=key.hello_rasa_project_id,
            lakera_project_id=key.lakera_project_id,
        )

    # 3) Await unique tasks once
    keys = list(tasks_by_key.keys())
    tasks = [tasks_by_key[k] for k in keys]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # 4) Map results back to all corresponding indices
    flagged: Set[int] = set()
    for key, response in zip(keys, responses):
        if isinstance(response, Exception):
            structlogger.warning(f"{log_prefix}.request_failed", error=str(response))
            continue
        if response.flagged:
            flagged.update(indices_by_key.get(key, []))

    return flagged


async def check_assistant_chat_for_policy_violations(
    tracker_context: TrackerContext,
    hello_rasa_user_id: Optional[str],
    hello_rasa_project_id: Optional[str],
) -> TrackerContext:
    """Return a sanitised TrackerContext with unsafe turns removed.

    Only user messages are moderated – assistant messages are assumed safe.
    LRU cache is used, so each unique user text is checked once.

    Args:
        tracker_context: The TrackerContext containing conversation turns.
        hello_rasa_user_id: The user ID for the conversation.
        hello_rasa_project_id: The project ID for the conversation.

    Returns:
        TrackerContext with unsafe turns removed.
    """
    # Collect (turn_index, user_text) for all turns with a user message
    items: List[Tuple[int, str]] = []
    for idx, turn in enumerate(tracker_context.conversation_turns):
        user_message = turn.user_message
        if not user_message:
            continue

        text = (user_message.text or "").strip()
        if not text:
            continue

        items.append((idx, text))

    flagged_turns = await _detect_flagged_user_indices(
        items,
        hello_rasa_user_id=hello_rasa_user_id,
        hello_rasa_project_id=hello_rasa_project_id,
        lakera_project_id=ASSISTANT_HISTORY_GUARDRAIL_PROJECT_ID,
        log_prefix="assistant_guardrails",
    )

    if not flagged_turns:
        return tracker_context

    structlogger.info(
        "assistant_guardrails.turns_flagged",
        count=len(flagged_turns),
        turn_indices=sorted(flagged_turns),
    )

    # Build a filtered TrackerContext
    safe_turns: List[AssistantConversationTurn] = [
        turn
        for idx, turn in enumerate(tracker_context.conversation_turns)
        if idx not in flagged_turns
    ]

    new_tracker_context = tracker_context.copy(deep=True)
    new_tracker_context.conversation_turns = safe_turns
    return new_tracker_context


def _annotate_flagged_user_messages(
    history: List[CopilotChatMessage], flagged_user_indices: Set[int]
) -> None:
    """Mark flagged user messages in-place on the original history.

    Args:
        history: The copilot chat history containing messages.
        flagged_user_indices: Set of indices of user messages that were flagged.
    """
    if not flagged_user_indices:
        return

    total = len(history)
    for uidx in flagged_user_indices:
        if 0 <= uidx < total and history[uidx].role == ROLE_USER:
            history[
                uidx
            ].response_category = ResponseCategory.GUARDRAILS_POLICY_VIOLATION


async def check_copilot_chat_for_policy_violations(
    context: CopilotContext,
    hello_rasa_user_id: Optional[str],
    hello_rasa_project_id: Optional[str],
) -> Optional[GeneratedContent]:
    """Check the copilot chat history for guardrail policy violations.

    Only user messages are moderated – assistant messages are assumed safe.
    LRU cache is used, so each unique user text is checked once.

    Args:
        context: The CopilotContext containing the copilot chat history.
        hello_rasa_user_id: The user ID for the conversation.
        hello_rasa_project_id: The project ID for the conversation.

    Returns:
        Returns a default violation response if the system flags any user message,
        otherwise return None.
    """
    history = context.copilot_chat_history

    # Collect (index, text) for user messages; skip ones already marked as violations
    items: List[Tuple[int, str]] = []
    for idx, message in enumerate(history):
        if message.response_category == ResponseCategory.GUARDRAILS_POLICY_VIOLATION:
            continue
        if message.role != ROLE_USER:
            continue
        formatted_message = message.to_openai_format()
        text = (formatted_message.get("content") or "").strip()
        if not text:
            continue
        items.append((idx, text))

    flagged_user_indices = await _detect_flagged_user_indices(
        items,
        hello_rasa_user_id=hello_rasa_user_id,
        hello_rasa_project_id=hello_rasa_project_id,
        lakera_project_id=COPILOT_HISTORY_GUARDRAIL_PROJECT_ID,
        log_prefix="copilot_guardrails",
    )

    _annotate_flagged_user_messages(history, flagged_user_indices)

    if not flagged_user_indices:
        return None

    # Identify the latest user message index in the current request
    last_user_idx: Optional[int] = None
    for i in range(len(history) - 1, -1, -1):
        if getattr(history[i], "role", None) == ROLE_USER:
            last_user_idx = i
            break

    # Remove flagged user messages and their next copilot messages
    indices_to_remove: Set[int] = set()
    total = len(history)
    for uidx in flagged_user_indices:
        indices_to_remove.add(uidx)
        next_idx = uidx + 1
        if (
            next_idx < total
            and getattr(history[next_idx], "role", None) == ROLE_COPILOT
        ):
            indices_to_remove.add(next_idx)

    # Apply sanitization
    filtered_history = [
        msg for i, msg in enumerate(history) if i not in indices_to_remove
    ]
    if len(filtered_history) != len(history):
        structlogger.info(
            "copilot_guardrails.history_sanitized",
            removed_indices=sorted(indices_to_remove),
            removed_messages=len(history) - len(filtered_history),
            kept_messages=len(filtered_history),
        )
        context.copilot_chat_history = filtered_history

    # Block only if the latest user message in this request was flagged
    if last_user_idx is not None and last_user_idx in flagged_user_indices:
        return CopilotResponseHandler.respond_to_guardrail_policy_violations()

    # Otherwise proceed (following messages are respected)
    return None
