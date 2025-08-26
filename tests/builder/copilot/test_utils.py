import asyncio
from typing import List, Tuple
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from rasa.builder.copilot.constants import ROLE_COPILOT, ROLE_USER
from rasa.builder.copilot.copilot_response_handler import CopilotResponseHandler
from rasa.builder.copilot.models import (
    CopilotChatMessage,
    CopilotContext,
    ResponseCategory,
    TextContent,
)
from rasa.builder.guardrails.models import GuardrailRequestKey, GuardrailResponse
from rasa.builder.guardrails.utils import (
    _detect_flagged_user_indices,
    _schedule_guardrails_check,
    check_assistant_chat_for_policy_violations,
    check_copilot_chat_for_policy_violations,
)
from rasa.builder.llm_service import llm_service
from rasa.builder.shared.tracker_context import (
    AssistantConversationTurn,
    CurrentState,
    TrackerContext,
    UserMessage,
)


@pytest.fixture(autouse=True)
def clear_schedule_guardrails_check_cache() -> None:
    """Ensure _schedule_guardrails_check LRU cache is clean between tests."""
    _schedule_guardrails_check.cache_clear()


@pytest.mark.asyncio
async def test_schedule_guardrails_check_caches_tasks(monkeypatch: pytest.MonkeyPatch):
    mock = AsyncMock(return_value=GuardrailResponse(flagged=False))
    monkeypatch.setattr(llm_service.guardrails, "send_request", mock)

    # Same arguments - same cached task
    t1 = _schedule_guardrails_check("hello", "user-1", "proj-1", "lakera-1")
    t2 = _schedule_guardrails_check("hello", "user-1", "proj-1", "lakera-1")
    assert t1 is t2

    # Different arguments - different task
    t3 = _schedule_guardrails_check("hello", "user-1", "proj-2", "lakera-1")
    t4 = _schedule_guardrails_check("hello2", "user-1", "proj-1", "lakera-1")
    assert t3 is not t1
    assert t4 is not t1

    # Await tasks to avoid warnings about pending tasks
    res1, res2, res3 = await asyncio.gather(t1, t3, t4)
    assert isinstance(res1, GuardrailResponse)
    assert isinstance(res2, GuardrailResponse)
    assert isinstance(res3, GuardrailResponse)
    assert not res1.flagged
    assert not res2.flagged
    assert not res3.flagged


@pytest.mark.asyncio
async def test_check_assistant_chat_filters_flagged_turns(
    monkeypatch: pytest.MonkeyPatch,
):
    # Build a TrackerContext with three turns
    turns = [
        AssistantConversationTurn(user_message=UserMessage(text="A")),  # flagged
        AssistantConversationTurn(user_message=UserMessage(text="B")),  # safe
        AssistantConversationTurn(user_message=UserMessage(text="C")),  # flagged
    ]
    context = TrackerContext(conversation_turns=turns, current_state=CurrentState())

    async def _fake_send_request(request) -> GuardrailResponse:
        # Requests come from _schedule_guardrails_check, with one message per user text
        user_text = (request.messages[0]["content"] or "").strip()
        return GuardrailResponse(flagged=user_text in {"A", "C"})

    monkeypatch.setattr(llm_service.guardrails, "send_request", _fake_send_request)

    new_context = await check_assistant_chat_for_policy_violations(
        tracker_context=context,
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
    )

    # We should get a new TrackerContext instance with only the safe turn ("B")
    assert new_context is not context
    assert len(new_context.conversation_turns) == 1
    assert new_context.conversation_turns[0].user_message is not None
    assert new_context.conversation_turns[0].user_message.text == "B"


@pytest.mark.asyncio
async def test_check_assistant_chat_returns_same_if_no_flags(
    monkeypatch: pytest.MonkeyPatch,
):
    turns = [
        AssistantConversationTurn(user_message=UserMessage(text="hello")),
        AssistantConversationTurn(user_message=UserMessage(text="world")),
    ]
    ctx = TrackerContext(conversation_turns=turns, current_state=CurrentState())

    mock_send_request = AsyncMock(return_value=GuardrailResponse(flagged=False))
    monkeypatch.setattr(llm_service.guardrails, "send_request", mock_send_request)

    same_ctx = await check_assistant_chat_for_policy_violations(
        tracker_context=ctx,
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
    )
    assert same_ctx is ctx
    assert len(same_ctx.conversation_turns) == 2


@pytest.mark.asyncio
async def test_check_copilot_chat_builds_request_and_redacts(
    monkeypatch: pytest.MonkeyPatch,
):
    history = [
        CopilotChatMessage(
            role=ROLE_COPILOT,
            content=[TextContent(type="text", text="copilot says hi")],
        ),
        CopilotChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="user says hello")],
        ),
        CopilotChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="should be skipped")],
            response_category=ResponseCategory.GUARDRAILS_POLICY_VIOLATION,
        ),
    ]

    mock = AsyncMock(return_value=GuardrailResponse(flagged=False))
    monkeypatch.setattr(llm_service.guardrails, "send_request", mock)

    response = await check_copilot_chat_for_policy_violations(
        context=CopilotContext(copilot_chat_history=history),
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
    )
    assert response is None

    request = mock.await_args.args[0]
    captured_messages = request.messages

    # Only one message should be sent (flagged one excluded)
    assert len(captured_messages) == 1
    assert captured_messages[0]["role"] == "user"
    assert captured_messages[0]["content"] == "user says hello"


@pytest.mark.asyncio
async def test_check_copilot_chat_returns_violation_response(
    monkeypatch: pytest.MonkeyPatch,
):
    history = [
        CopilotChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="unsafe prompt")],
        )
    ]

    mock_send_request = AsyncMock(return_value=GuardrailResponse(flagged=True))
    monkeypatch.setattr(llm_service.guardrails, "send_request", mock_send_request)

    response = await check_copilot_chat_for_policy_violations(
        context=CopilotContext(copilot_chat_history=history),
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
    )

    assert response is not None
    assert response.response_category == ResponseCategory.GUARDRAILS_POLICY_VIOLATION

    # Compare content with the helper's canonical response to avoid string drift
    expected = CopilotResponseHandler.respond_to_guardrail_policy_violations()
    assert response.content == expected.content


@pytest.mark.asyncio
async def test_check_copilot_sanitizes_non_latest_flagged_user_message(
    monkeypatch: pytest.MonkeyPatch,
):
    history = [
        CopilotChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="ok-1")],  # safe
        ),
        CopilotChatMessage(
            role=ROLE_COPILOT,
            content=[TextContent(type="text", text="copilot to ok-1")],
        ),
        CopilotChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="unsafe")],  # flagged
        ),
        CopilotChatMessage(
            role=ROLE_COPILOT,
            content=[TextContent(type="text", text="copilot to unsafe")],
        ),
        CopilotChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="ok-2")],  # safe
        ),
    ]

    async def _fake_send_request(request) -> GuardrailResponse:
        user_text = (request.messages[0]["content"] or "").strip()
        return GuardrailResponse(flagged=(user_text == "unsafe"))

    monkeypatch.setattr(llm_service.guardrails, "send_request", _fake_send_request)

    ctx = CopilotContext(copilot_chat_history=history)
    result = await check_copilot_chat_for_policy_violations(
        context=ctx,
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
    )

    # Not blocked because the latest user message ("ok-2") is safe
    assert result is None

    # History should be sanitized: remove "unsafe" message
    remaining = ctx.copilot_chat_history
    assert [m.get_text_content() for m in remaining if m.role == ROLE_USER] == [
        "ok-1",
        "ok-2",
    ]


@pytest.mark.asyncio
async def test_check_copilot_chat_blocks_when_latest_user_flagged_and_sanitizes(
    monkeypatch: pytest.MonkeyPatch,
):
    # Latest user message is unsafe
    history = [
        CopilotChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="ok")],  # safe
        ),
        CopilotChatMessage(
            role=ROLE_COPILOT,
            content=[TextContent(type="text", text="copilot to ok")],
        ),
        CopilotChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="unsafe")],  # flagged
        ),
    ]

    async def _fake_send_request(request) -> GuardrailResponse:
        user_text = (request.messages[0]["content"] or "").strip()
        return GuardrailResponse(flagged=(user_text == "unsafe"))

    monkeypatch.setattr(llm_service.guardrails, "send_request", _fake_send_request)

    ctx = CopilotContext(copilot_chat_history=history)
    result = await check_copilot_chat_for_policy_violations(
        context=ctx,
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
    )

    # Blocked with default violation response
    assert result == CopilotResponseHandler.respond_to_guardrail_policy_violations()
    assert result.response_category == ResponseCategory.GUARDRAILS_POLICY_VIOLATION

    # History sanitized - latest "unsafe" user message removed
    remaining_user_texts = [
        m.get_text_content() for m in ctx.copilot_chat_history if m.role == ROLE_USER
    ]
    assert remaining_user_texts == ["ok"]


@pytest.mark.asyncio
async def test__detect_flagged_user_indices_maps_back_to_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items: List[Tuple[int, str]] = [
        (0, "  A  "),  # flagged (after strip)
        (1, "B"),  # safe
        (2, "   "),  # ignored
        (3, "C"),  # flagged
        (4, "A"),  # flagged (duplicate text)
    ]

    async def _fake_send_request(request) -> GuardrailResponse:
        text = (request.messages[0]["content"] or "").strip()
        return GuardrailResponse(flagged=text in {"A", "C"})

    monkeypatch.setattr(llm_service.guardrails, "send_request", _fake_send_request)

    flagged = await _detect_flagged_user_indices(
        items,
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
        lakera_project_id="lakera-1",
        log_prefix="test_guardrails",
    )

    assert flagged == {0, 3, 4}


@pytest.mark.asyncio
async def test__detect_flagged_user_indices_handles_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items: List[Tuple[int, str]] = [
        (0, "ERR"),
        (1, "SAFE"),
    ]

    async def _fake_send_request(request) -> GuardrailResponse:
        text = (request.messages[0]["content"] or "").strip()
        if text == "ERR":
            raise RuntimeError("provider failure")
        return GuardrailResponse(flagged=False)

    monkeypatch.setattr(llm_service.guardrails, "send_request", _fake_send_request)

    flagged = await _detect_flagged_user_indices(
        items,
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
        lakera_project_id="lakera-1",
        log_prefix="test_guardrails",
    )
    assert flagged == set()


def test_guardrail_request_key_value_equality_and_hash() -> None:
    k1 = GuardrailRequestKey(
        user_text="hi",
        hello_rasa_user_id="u1",
        hello_rasa_project_id="p1",
        lakera_project_id="lk1",
    )
    k2 = GuardrailRequestKey(
        user_text="hi",
        hello_rasa_user_id="u1",
        hello_rasa_project_id="p1",
        lakera_project_id="lk1",
    )

    # Different instances, equal by value, same hash
    assert k1 is not k2
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_guardrail_request_key_as_dict_key() -> None:
    k1 = GuardrailRequestKey(
        user_text="hello",
        hello_rasa_user_id="u1",
        hello_rasa_project_id="p1",
        lakera_project_id="lk1",
    )
    k2 = GuardrailRequestKey(
        user_text="hello",
        hello_rasa_user_id="u1",
        hello_rasa_project_id="p1",
        lakera_project_id="lk1",
    )

    mapping = {k1: "value"}
    assert mapping[k2] == "value"
    assert len(mapping) == 1


def test_guardrail_request_key_inequality_by_field() -> None:
    base = GuardrailRequestKey(
        user_text="hello",
        hello_rasa_user_id="u1",
        hello_rasa_project_id="p1",
        lakera_project_id="lk1",
    )
    diff_text = GuardrailRequestKey(
        user_text="hello!",
        hello_rasa_user_id="u1",
        hello_rasa_project_id="p1",
        lakera_project_id="lk1",
    )
    diff_user = GuardrailRequestKey(
        user_text="hello",
        hello_rasa_user_id="u2",
        hello_rasa_project_id="p1",
        lakera_project_id="lk1",
    )
    diff_project = GuardrailRequestKey(
        user_text="hello",
        hello_rasa_user_id="u1",
        hello_rasa_project_id="p2",
        lakera_project_id="lk1",
    )
    diff_policy = GuardrailRequestKey(
        user_text="hello",
        hello_rasa_user_id="u1",
        hello_rasa_project_id="p1",
        lakera_project_id="lk2",
    )

    assert base != diff_text
    assert base != diff_user
    assert base != diff_project
    assert base != diff_policy

    # As dict keys, they should create distinct entries
    mapping = {base: 1, diff_text: 2, diff_user: 3, diff_project: 4, diff_policy: 5}
    assert len(mapping) == 5


def test_guardrail_request_key_is_frozen_immutable() -> None:
    key = GuardrailRequestKey(
        user_text="immutable",
        hello_rasa_user_id="u1",
        hello_rasa_project_id="p1",
        lakera_project_id="lk1",
    )
    with pytest.raises(ValidationError):
        key.user_text = "mutated"
