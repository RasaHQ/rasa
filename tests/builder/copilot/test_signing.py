from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

from rasa.builder import config
from rasa.builder.copilot.constants import SIGNATURE_VERSION_V1
from rasa.builder.copilot.exceptions import (
    InvalidCopilotChatHistorySignature,
    MissingCopilotChatHistorySignature,
)
from rasa.builder.copilot.models import (
    CopilotChatMessage,
    GeneratedContent,
    ResponseCategory,
    TextContent,
)
from rasa.builder.copilot.signing import (
    _b64url_no_pad,
    _canonicalize_messages,
    compute_history_signature,
    extract_full_text_and_category,
    verify_history_signature,
    verify_signature,
)


def make_user_message(text: str) -> CopilotChatMessage:
    return CopilotChatMessage(
        role="user", content=[TextContent(type="text", text=text)]
    )


def make_copilot_message(
    text: str, category: Optional[ResponseCategory] = None
) -> CopilotChatMessage:
    return CopilotChatMessage(
        role="copilot",
        content=[TextContent(type="text", text=text)],
        response_category=category,
    )


class RequestObject:
    """Lightweight request-like object with the fields signing expects."""

    def __init__(
        self,
        *,
        session_id: str,
        copilot_chat_history: List[CopilotChatMessage],
        signature_version: Optional[str] = None,
        history_signature: Optional[str] = None,
    ) -> None:
        self.session_id = session_id
        self.copilot_chat_history = copilot_chat_history
        self.signature_version = signature_version
        self.history_signature = history_signature


@pytest.fixture()
def sse():
    stream = MagicMock()
    stream.send = AsyncMock()
    stream.eof = AsyncMock()
    return stream


@pytest.fixture(autouse=True)
def set_secret(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        config, "COPILOT_HISTORY_SIGNING_SECRET", "devsecret", raising=False
    )


def test_b64url_no_pad():
    # 0xff ia "_w==", without padding it is "_w"
    assert _b64url_no_pad(b"\xff") == "_w"


def test_canonicalize_messages_provides_canonical_form():
    messages = [
        make_user_message("hello"),
        make_copilot_message("world"),
    ]
    canonical = _canonicalize_messages(messages)
    assert isinstance(canonical, (bytes, bytearray))

    canonical_string = canonical.decode("utf-8")
    assert '"role":"user"' in canonical_string
    assert '"role":"copilot"' in canonical_string
    assert "hello" in canonical_string and "world" in canonical_string


def test_canonicalize_messages_is_deterministic():
    messages1 = [make_user_message("a"), make_copilot_message("b")]
    messages2 = [make_user_message("a"), make_copilot_message("b")]
    assert _canonicalize_messages(messages1) == _canonicalize_messages(messages2)


def test_compute_history_signature_changes_with_text():
    session_id = "s1"
    messages1 = [make_user_message("hello")]
    messages2 = [make_user_message("hello!")]

    signature1 = compute_history_signature(
        messages1, session_id, config.COPILOT_HISTORY_SIGNING_SECRET
    )
    signature2 = compute_history_signature(
        messages2, session_id, config.COPILOT_HISTORY_SIGNING_SECRET
    )
    assert signature1 != signature2


def test_compute_history_signature_changes_with_session_id():
    messages = [make_user_message("hello")]
    signature1 = compute_history_signature(
        messages, "s1", config.COPILOT_HISTORY_SIGNING_SECRET
    )
    signature2 = compute_history_signature(
        messages, "s2", config.COPILOT_HISTORY_SIGNING_SECRET
    )
    assert signature1 != signature2


def test_compute_history_signature_changes_with_version():
    messages = [make_user_message("hello")]
    signature1 = compute_history_signature(
        messages, "s", config.COPILOT_HISTORY_SIGNING_SECRET, version="v1"
    )
    signature2 = compute_history_signature(
        messages, "s", config.COPILOT_HISTORY_SIGNING_SECRET, version="v2"
    )
    assert signature1 != signature2


def test_verify_signature_ok():
    messages = [make_user_message("hello"), make_copilot_message("welcome!")]

    signature = compute_history_signature(
        messages, "s", config.COPILOT_HISTORY_SIGNING_SECRET
    )
    assert verify_history_signature(
        signature,
        messages,
        "s",
        config.COPILOT_HISTORY_SIGNING_SECRET,
        SIGNATURE_VERSION_V1,
    )


def test_verify_signature_fail_on_alteration():
    messages = [make_user_message("hello"), make_copilot_message("welcome!")]
    signature = compute_history_signature(
        messages, "s", config.COPILOT_HISTORY_SIGNING_SECRET
    )

    # Tamper text
    messages_altered = [make_user_message("hello"), make_copilot_message("welcome?")]
    assert not verify_history_signature(
        signature,
        messages_altered,
        "s",
        config.COPILOT_HISTORY_SIGNING_SECRET,
        SIGNATURE_VERSION_V1,
    )


@pytest.mark.asyncio
async def test_verify_signature_no_secret_works(
    monkeypatch: MonkeyPatch, sse: MagicMock
):
    monkeypatch.setattr(config, "COPILOT_HISTORY_SIGNING_SECRET", None, raising=False)

    req = RequestObject(
        session_id="s",
        copilot_chat_history=[make_user_message("hi")],
        signature_version="v1",
    )
    assert await verify_signature(req)
    assert sse.send.await_count == 0


@pytest.mark.asyncio
async def test_verify_signature_no_opt_in_works(sse: MagicMock):
    req = RequestObject(
        session_id="s",
        copilot_chat_history=[make_user_message("hi")],
        signature_version=None,
    )
    assert await verify_signature(req)
    assert sse.send.await_count == 0


@pytest.mark.asyncio
async def test_verify_signature_first_turn_missing_signature_works(sse: MagicMock):
    req = RequestObject(
        session_id="s",
        copilot_chat_history=[make_user_message("hi")],
        signature_version="v1",
    )
    assert await verify_signature(req)
    assert sse.send.await_count == 0


@pytest.mark.asyncio
async def test_verify_signature_user_multi_turn_missing_signature_fails(sse: MagicMock):
    history = [
        make_user_message("hi"),
        make_copilot_message("hello there"),
        make_user_message("how are you?"),
    ]
    req = RequestObject(
        session_id="s", copilot_chat_history=history, signature_version="v1"
    )

    with pytest.raises(MissingCopilotChatHistorySignature):
        await verify_signature(req)


@pytest.mark.asyncio
async def test_verify_signature_multi_turn_with_valid_signature_succeeds(
    sse: MagicMock,
):
    history = [make_user_message("hi"), make_copilot_message("hello there")]
    req = RequestObject(
        session_id="s",
        copilot_chat_history=history,
        signature_version="v1",
        history_signature=compute_history_signature(
            history, "s", config.COPILOT_HISTORY_SIGNING_SECRET
        ),
    )

    assert await verify_signature(req)

    # No error has been sent
    assert sse.send.await_count == 0


@pytest.mark.asyncio
async def test_verify_signature_multi_turn_with_invalid_signature_fails(sse: MagicMock):
    history = [make_user_message("hi"), make_copilot_message("hello there")]
    signature = compute_history_signature(
        history, "s", config.COPILOT_HISTORY_SIGNING_SECRET
    )

    # Alter the original signature
    altered_signature = signature + "A"
    req = RequestObject(
        session_id="s",
        copilot_chat_history=history,
        signature_version="v1",
        history_signature=altered_signature,
    )

    with pytest.raises(InvalidCopilotChatHistorySignature):
        await verify_signature(req)


def test_extract_full_text_and_category_simple():
    class Handler:
        def __init__(self) -> None:
            self.generated_responses: List[Any] = []

    handler = Handler()
    handler.generated_responses.append(
        GeneratedContent(content="Hello ", response_category=ResponseCategory.COPILOT)
    )
    handler.generated_responses.append(
        GeneratedContent(content="world!", response_category=ResponseCategory.COPILOT)
    )
    text, cat = extract_full_text_and_category(handler)
    assert text == "Hello world!"
    assert cat == ResponseCategory.COPILOT


def test_extract_full_text_and_category_ignores_reference_for_category() -> None:
    class Handler:
        def __init__(self) -> None:
            self.generated_responses: List[Any] = []

    handler = Handler()
    handler.generated_responses.append(
        GeneratedContent(content="A", response_category=ResponseCategory.COPILOT)
    )
    handler.generated_responses.append(
        GeneratedContent(content="B", response_category=ResponseCategory.REFERENCE)
    )
    handler.generated_responses.append(
        GeneratedContent(content="C", response_category=ResponseCategory.COPILOT)
    )

    text, cat = extract_full_text_and_category(handler)
    assert text == "ABC"
    assert cat == ResponseCategory.COPILOT


@pytest.mark.asyncio
async def test_conversation_opt_in_success_and_tamper_detection(sse: MagicMock):
    """Test the conversation opt-in flow with signature verification"""
    session_id = "session-123"

    # 1) First turn - user sends a message, no signature provided
    req1 = RequestObject(
        session_id=session_id,
        copilot_chat_history=[make_user_message("Hi")],
        signature_version="v1",
    )
    # Signature verification should pass and not emit errors
    assert await verify_signature(req1)
    assert sse.send.await_count == 0

    # Server generates assistant message and computes signature
    assistant_message = make_copilot_message("Hello! How can I help?")
    signature = compute_history_signature(
        [*req1.copilot_chat_history, assistant_message],
        session_id,
        config.COPILOT_HISTORY_SIGNING_SECRET,
        version="v1",
    )

    # 2) Assistant responds with message and signature
    sse2 = MagicMock()
    sse2.send = AsyncMock()
    sse2.eof = AsyncMock()

    history = [make_user_message("Hi"), assistant_message]
    req2 = RequestObject(
        session_id=session_id,
        copilot_chat_history=history,
        signature_version="v1",
        history_signature=signature,
    )
    assert await verify_signature(req2)
    assert sse2.send.await_count == 0

    # 3) User alters the conversation history and sends it back
    sse3 = MagicMock()
    sse3.send = AsyncMock()
    sse3.eof = AsyncMock()

    altered_assistant_message = make_copilot_message("Hello? How can I help?")
    req3 = RequestObject(
        session_id=session_id,
        copilot_chat_history=[make_user_message("Hi"), altered_assistant_message],
        signature_version="v1",
        history_signature=signature,
    )
    with pytest.raises(InvalidCopilotChatHistorySignature):
        await verify_signature(req3)

    # 4) User continues the conversation with a new message appended to the history
    sse4 = MagicMock()
    sse4.send = AsyncMock()
    sse4.eof = AsyncMock()

    next_user_message = make_user_message("I'd like to check my balance.")
    extended_history = [make_user_message("Hi"), assistant_message, next_user_message]
    req4 = RequestObject(
        session_id=session_id,
        copilot_chat_history=extended_history,
        signature_version="v1",
        history_signature=signature,
    )
    assert await verify_signature(req4)
    assert sse4.send.await_count == 0


@pytest.mark.asyncio
async def test_first_turn_uses_user_count_allows_multiple_copilot_turns(sse: MagicMock):
    session_id = "session-multi-copilot"

    # Only copilot turns, no user yet -> should be allowed without signature
    copilot_messages = [
        make_copilot_message("Welcome!"),
        make_copilot_message("How can I assist you today?"),
    ]
    copilot_messages_req = RequestObject(
        session_id=session_id,
        copilot_chat_history=copilot_messages,
        signature_version="v1",
        history_signature=None,
    )
    assert await verify_signature(copilot_messages_req)
    assert sse.send.await_count == 0

    # First user turn appended - still allowed without signature
    user_message_history = [
        *copilot_messages,
        make_user_message("Hi, I need help with my bot."),
    ]
    user_message_req = RequestObject(
        session_id=session_id,
        copilot_chat_history=user_message_history,
        signature_version="v1",
        history_signature=None,
    )
    assert await verify_signature(user_message_req)
    assert sse.send.await_count == 0

    # 2) Second user turn appended without providing a signature - must fail
    history = [
        *user_message_history,
        make_copilot_message("Sure, what would you like to work on?"),
        make_user_message("Tell me about slots."),
    ]
    req = RequestObject(
        session_id=session_id,
        copilot_chat_history=history,
        signature_version="v1",
        history_signature=None,
    )
    with pytest.raises(MissingCopilotChatHistorySignature):
        await verify_signature(req)


@pytest.mark.asyncio
async def test_signature_flow_guardrails_response_allows_next_turn(sse: MagicMock):
    session_id = "sess-guardrails"

    # 1) First user turn, no signature yet -> allowed
    req1 = RequestObject(
        session_id=session_id,
        copilot_chat_history=[make_user_message("unsafe message")],
        signature_version="v1",
    )
    assert await verify_signature(req1)
    assert sse.send.await_count == 0

    # 2) Server responds with guardrails default message and sends signature
    guardrails_text = "Policy violation response"
    assistant_msg = make_copilot_message(
        guardrails_text, category=ResponseCategory.GUARDRAILS_POLICY_VIOLATION
    )
    signature = compute_history_signature(
        [*req1.copilot_chat_history, assistant_msg],
        session_id,
        config.COPILOT_HISTORY_SIGNING_SECRET,
        version="v1",
    )

    # 3) Next client request includes assistant message + new user message and signature
    sse_next = MagicMock()
    sse_next.send = AsyncMock()
    sse_next.eof = AsyncMock()

    req2 = RequestObject(
        session_id=session_id,
        copilot_chat_history=[
            *req1.copilot_chat_history,
            assistant_msg,
            make_user_message("ok then"),
        ],
        signature_version="v1",
        history_signature=signature,
    )
    assert await verify_signature(req2)
    assert sse_next.send.await_count == 0


@pytest.mark.asyncio
async def test_signature_flow_out_of_scope_response_allows_next_turn(sse: MagicMock):
    session_id = "sess-oos"

    # 1) First user turn, no signature yet -> allowed
    req1 = RequestObject(
        session_id=session_id,
        copilot_chat_history=[make_user_message("what's the weather on mars?")],
        signature_version="v1",
    )
    assert await verify_signature(req1)
    assert sse.send.await_count == 0

    # 2) Server responds with out-of-scope message and sends signature
    oos_text = "Out of scope response"
    assistant_msg = make_copilot_message(
        oos_text, category=ResponseCategory.OUT_OF_SCOPE_DETECTION
    )
    signature = compute_history_signature(
        [*req1.copilot_chat_history, assistant_msg],
        session_id,
        config.COPILOT_HISTORY_SIGNING_SECRET,
        version="v1",
    )

    # 3) Next client request includes assistant message + new user message and signature
    sse_next = MagicMock()
    sse_next.send = AsyncMock()
    sse_next.eof = AsyncMock()

    req2 = RequestObject(
        session_id=session_id,
        copilot_chat_history=[
            *req1.copilot_chat_history,
            assistant_msg,
            make_user_message("ok, let's proceed then"),
        ],
        signature_version="v1",
        history_signature=signature,
    )
    assert await verify_signature(req2)
    assert sse_next.send.await_count == 0


@pytest.mark.asyncio
async def test_signature_flow_roleplay_response_allows_next_turn(sse: MagicMock):
    session_id = "sess-roleplay"

    # 1) First user turn, no signature yet -> allowed
    req1 = RequestObject(
        session_id=session_id,
        copilot_chat_history=[make_user_message("add contact")],
        signature_version="v1",
    )
    assert await verify_signature(req1)
    assert sse.send.await_count == 0

    # 2) Server responds with roleplay-detection message and sends signature
    roleplay_text = "Roleplay detected response"
    assistant_msg = make_copilot_message(
        roleplay_text, category=ResponseCategory.ROLEPLAY_DETECTION
    )
    signature = compute_history_signature(
        [*req1.copilot_chat_history, assistant_msg],
        session_id,
        config.COPILOT_HISTORY_SIGNING_SECRET,
        version="v1",
    )

    # 3) Next client request includes assistant message + new user message and signature
    sse_next = MagicMock()
    sse_next.send = AsyncMock()
    sse_next.eof = AsyncMock()

    req2 = RequestObject(
        session_id=session_id,
        copilot_chat_history=[
            *req1.copilot_chat_history,
            assistant_msg,
            make_user_message("got it, help me with flows then"),
        ],
        signature_version="v1",
        history_signature=signature,
    )
    assert await verify_signature(req2)
    assert sse_next.send.await_count == 0
