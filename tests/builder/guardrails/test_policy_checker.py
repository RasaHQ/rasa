"""Unit tests for GuardrailsPolicyChecker."""

from typing import List, Tuple
from unittest.mock import AsyncMock

import pytest

from rasa.builder.copilot.constants import ROLE_COPILOT, ROLE_USER
from rasa.builder.copilot.copilot_response_handler import CopilotResponseHandler
from rasa.builder.copilot.models import (
    CopilotChatMessage,
    CopilotContext,
    ResponseCategory,
    TextContent,
    UserChatMessage,
)
from rasa.builder.guardrails.clients import LakeraAIGuardrails
from rasa.builder.guardrails.models import GuardrailResponse
from rasa.builder.guardrails.policy_checker import GuardrailsPolicyChecker
from rasa.builder.shared.tracker_context import (
    AssistantConversationTurn,
    CurrentState,
    TrackerContext,
    UserMessage,
)


@pytest.mark.asyncio
async def test_check_assistant_chat_filters_flagged_turns(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the policy checker filters out flagged turns."""
    # Build a TrackerContext with three turns
    turns = [
        AssistantConversationTurn(user_message=UserMessage(text="A")),  # flagged
        AssistantConversationTurn(user_message=UserMessage(text="B")),  # safe
        AssistantConversationTurn(user_message=UserMessage(text="C")),  # flagged
    ]
    context = TrackerContext(conversation_turns=turns, current_state=CurrentState())

    async def _fake_send_request(request) -> GuardrailResponse:
        # Requests come from schedule_check, with one message per user text
        user_text = (request.messages[0]["content"] or "").strip()
        return GuardrailResponse(flagged=user_text in {"A", "C"})

    # Create policy checker and mock the client
    client = LakeraAIGuardrails(api_key="test_key")
    policy_checker = GuardrailsPolicyChecker(client)
    monkeypatch.setattr(client, "send_request", _fake_send_request)

    new_context = await policy_checker.check_assistant_chat_for_policy_violations(
        tracker_context=context,
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
        lakera_project_id="lakera-1",
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
    """Test that the policy checker returns same context if no flags."""
    turns = [
        AssistantConversationTurn(user_message=UserMessage(text="hello")),
        AssistantConversationTurn(user_message=UserMessage(text="world")),
    ]
    ctx = TrackerContext(conversation_turns=turns, current_state=CurrentState())

    # Create policy checker and mock the client
    client = LakeraAIGuardrails(api_key="test_key")
    policy_checker = GuardrailsPolicyChecker(client)
    mock_send_request = AsyncMock(return_value=GuardrailResponse(flagged=False))
    monkeypatch.setattr(client, "send_request", mock_send_request)

    same_ctx = await policy_checker.check_assistant_chat_for_policy_violations(
        tracker_context=ctx,
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
        lakera_project_id="lakera-1",
    )
    assert same_ctx is ctx
    assert len(same_ctx.conversation_turns) == 2


@pytest.mark.asyncio
async def test_check_copilot_chat_builds_request_and_redacts(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the policy checker builds request and redacts flagged messages."""
    history = [
        CopilotChatMessage(
            role=ROLE_COPILOT,
            content=[TextContent(type="text", text="copilot says hi")],
        ),
        UserChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="user says hello")],
        ),
        UserChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="should be skipped")],
            response_category=ResponseCategory.GUARDRAILS_POLICY_VIOLATION,
        ),
    ]

    # Create policy checker and mock the client
    client = LakeraAIGuardrails(api_key="test_key")
    policy_checker = GuardrailsPolicyChecker(client)
    mock = AsyncMock(return_value=GuardrailResponse(flagged=False))
    monkeypatch.setattr(client, "send_request", mock)

    response = await policy_checker.check_copilot_chat_for_policy_violations(
        context=CopilotContext(copilot_chat_history=history),
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
        lakera_project_id="lakera-1",
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
    """Test that the policy checker returns violation response when flagged."""
    history = [
        UserChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="unsafe prompt")],
        )
    ]

    # Create policy checker and mock the client
    client = LakeraAIGuardrails(api_key="test_key")
    policy_checker = GuardrailsPolicyChecker(client)
    mock_send_request = AsyncMock(return_value=GuardrailResponse(flagged=True))
    monkeypatch.setattr(client, "send_request", mock_send_request)

    response = await policy_checker.check_copilot_chat_for_policy_violations(
        context=CopilotContext(copilot_chat_history=history),
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
        lakera_project_id="lakera-1",
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
    """Test that the policy checker sanitizes non-latest flagged messages."""
    history = [
        UserChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="ok-1")],  # safe
        ),
        CopilotChatMessage(
            role=ROLE_COPILOT,
            content=[TextContent(type="text", text="copilot to ok-1")],
        ),
        UserChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="unsafe")],  # flagged
        ),
        CopilotChatMessage(
            role=ROLE_COPILOT,
            content=[TextContent(type="text", text="copilot to unsafe")],
        ),
        UserChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="ok-2")],  # safe
        ),
    ]

    async def _fake_send_request(request) -> GuardrailResponse:
        user_text = (request.messages[0]["content"] or "").strip()
        return GuardrailResponse(flagged=(user_text == "unsafe"))

    # Create policy checker and mock the client
    client = LakeraAIGuardrails(api_key="test_key")
    policy_checker = GuardrailsPolicyChecker(client)
    monkeypatch.setattr(client, "send_request", _fake_send_request)

    ctx = CopilotContext(copilot_chat_history=history)
    result = await policy_checker.check_copilot_chat_for_policy_violations(
        context=ctx,
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
        lakera_project_id="lakera-1",
    )

    # Not blocked because the latest user message ("ok-2") is safe
    assert result is None

    # History should be sanitized: remove "unsafe" message
    remaining = ctx.copilot_chat_history
    assert [
        m.get_flattened_text_content() for m in remaining if m.role == ROLE_USER
    ] == [
        "ok-1",
        "ok-2",
    ]


@pytest.mark.asyncio
async def test_check_copilot_chat_blocks_when_latest_user_flagged_and_sanitizes(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the policy checker blocks when latest user message is flagged."""
    # Latest user message is unsafe
    history = [
        UserChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="ok")],  # safe
        ),
        CopilotChatMessage(
            role=ROLE_COPILOT,
            content=[TextContent(type="text", text="copilot to ok")],
        ),
        UserChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="unsafe")],  # flagged
        ),
    ]

    async def _fake_send_request(request) -> GuardrailResponse:
        user_text = (request.messages[0]["content"] or "").strip()
        return GuardrailResponse(flagged=(user_text == "unsafe"))

    # Create policy checker and mock the client
    client = LakeraAIGuardrails(api_key="test_key")
    policy_checker = GuardrailsPolicyChecker(client)
    monkeypatch.setattr(client, "send_request", _fake_send_request)

    ctx = CopilotContext(copilot_chat_history=history)
    result = await policy_checker.check_copilot_chat_for_policy_violations(
        context=ctx,
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
        lakera_project_id="lakera-1",
    )

    # Blocked with default violation response
    assert result == CopilotResponseHandler.respond_to_guardrail_policy_violations()
    assert result.response_category == ResponseCategory.GUARDRAILS_POLICY_VIOLATION

    # History sanitized - latest "unsafe" user message removed
    remaining_user_texts = [
        m.get_flattened_text_content()
        for m in ctx.copilot_chat_history
        if m.role == ROLE_USER
    ]
    assert remaining_user_texts == ["ok"]


@pytest.mark.asyncio
async def test_check_user_messages_for_violations_maps_back_to_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the policy checker correctly maps results back to indices."""
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

    # Create policy checker and mock the client
    client = LakeraAIGuardrails(api_key="test_key")
    policy_checker = GuardrailsPolicyChecker(client)
    monkeypatch.setattr(client, "send_request", _fake_send_request)

    flagged = await policy_checker._check_user_messages_for_violations(
        items,
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
        log_prefix="test_guardrails",
        lakera_project_id="lakera-1",
    )

    assert flagged == {0, 3, 4}


@pytest.mark.asyncio
async def test_check_user_messages_for_violations_handles_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the policy checker handles exceptions gracefully."""
    items: List[Tuple[int, str]] = [
        (0, "ERR"),
        (1, "SAFE"),
    ]

    async def _fake_send_request(request) -> GuardrailResponse:
        text = (request.messages[0]["content"] or "").strip()
        if text == "ERR":
            raise RuntimeError("provider failure")
        return GuardrailResponse(flagged=False)

    # Create policy checker and mock the client
    client = LakeraAIGuardrails(api_key="test_key")
    policy_checker = GuardrailsPolicyChecker(client)
    monkeypatch.setattr(client, "send_request", _fake_send_request)

    flagged = await policy_checker._check_user_messages_for_violations(
        items,
        hello_rasa_user_id="user-1",
        hello_rasa_project_id="proj-1",
        log_prefix="test_guardrails",
        lakera_project_id="lakera-1",
    )
    assert flagged == set()
