import base64
import hashlib
import hmac
import json
from typing import Any, List, Optional, Tuple

import structlog

from rasa.builder import config
from rasa.builder.copilot.constants import ROLE_COPILOT, ROLE_USER, SIGNATURE_VERSION_V1
from rasa.builder.copilot.copilot_response_handler import CopilotResponseHandler
from rasa.builder.copilot.exceptions import (
    InvalidCopilotChatHistorySignature,
    MissingCopilotChatHistorySignature,
)
from rasa.builder.copilot.models import (
    CopilotChatMessage,
    CopilotRequest,
    GeneratedContent,
    ResponseCategory,
    ResponseCompleteness,
    ServerSentEvent,
    SigningContext,
    TextContent,
)

structlogger = structlog.get_logger()


def _b64url_no_pad(bytes_to_convert: bytes) -> str:
    """Convert bytes to a base64 URL-safe string without padding.

    Args:
        bytes_to_convert: Bytes to convert.

    Returns:
        str: Base64 URL-safe encoded string without padding.
    """
    return base64.urlsafe_b64encode(bytes_to_convert).rstrip(b"=").decode("ascii")


def _canonicalize_messages(messages: List[CopilotChatMessage]) -> bytes:
    """Canonicalize messages to a deterministic JSON format suitable for signing.

    Args:
        messages: List of CopilotChatMessage objects to normalize.

    Returns:
        bytes: Canonicalized JSON string of messages.
    """
    canonical: List[dict] = []

    for message in messages:
        # Preserve internal roles exactly as sent by the client (e.g. "user",
        # "copilot", or any future internal roles) and keep structured content blocks.
        message_dict = message.model_dump(
            include={"role", "content"}, exclude_none=True
        )

        role = message_dict.get("role", "")
        content = message_dict.get("content", [])

        canonical.append({"role": role, "content": content})

    # Dump to JSON with consistent formatting
    return json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def compute_history_signature(
    messages: List[CopilotChatMessage],
    session_id: str,
    secret: str,
    version: str = SIGNATURE_VERSION_V1,
) -> str:
    """Compute a signature for the chat history messages.

    Args:
        messages: List of CopilotChatMessage objects representing the chat history.
        session_id: Unique identifier for the session.
        secret: Secret key used for signing.
        version: Version of the signature format (default is SIGNATURE_VERSION_V1).

    Returns:
        str: Base64 URL-safe encoded signature.
    """
    canonical = _canonicalize_messages(messages)
    mac_input = (
        version.encode("utf-8")
        + b"\x00"
        + session_id.encode("utf-8")
        + b"\x00"
        + canonical
    )
    digest = hmac.new(
        key=secret.encode("utf-8"), msg=mac_input, digestmod=hashlib.sha256
    ).digest()
    signature = _b64url_no_pad(digest)
    return signature


def verify_history_signature(
    history_signature: str,
    messages: List[CopilotChatMessage],
    session_id: str,
    secret: str,
    version: Optional[str],
) -> bool:
    """Verify the history signature against the provided messages and session ID.

    Args:
        history_signature: Base64 URL-safe encoded signature to verify.
        messages: List of CopilotChatMessage objects representing the chat history.
        session_id: Unique identifier for the session.
        secret: Secret key used for signing.
        version: Version of the signature format (default is SIGNATURE_VERSION_V1).

    Returns:
        bool: True if the signature is valid, False otherwise.
    """
    version = version or SIGNATURE_VERSION_V1
    expected_signature = compute_history_signature(
        messages=messages, session_id=session_id, secret=secret, version=version
    )
    return hmac.compare_digest(expected_signature, history_signature)


def _signing_context() -> Optional[SigningContext]:
    """Get the signing context for history signatures.

    Returns:
        SigningContext if a secret is present, otherwise None.
    """
    if not (secret := config.COPILOT_HISTORY_SIGNING_SECRET):
        return None

    return SigningContext(
        secret=secret,
        default_version=SIGNATURE_VERSION_V1,
    )


async def verify_signature(req: CopilotRequest) -> bool:
    """Verify the history signature for a request.

    Args:
        req: Request object containing copilot chat history and session ID.

    Returns:
        True if verification passes.

    Raises:
        InvalidCopilotChatHistorySignature: when the provided signature doesn't match.
        MissingCopilotChatHistorySignature: when a required signature is missing.
    """
    context = _signing_context()

    # If no server-side secret, signature verification is not enforced
    if context is None:
        return True

    # If the client didn't opt in by providing the signature version, skip verification
    if not (signature_version := getattr(req, "signature_version", None)):
        return True

    # Allow missing signature on first user turn
    user_message_count = sum(
        1 for m in req.copilot_chat_history if getattr(m, "role", None) == ROLE_USER
    )
    allow_missing = user_message_count <= 1
    version = signature_version or context.default_version

    history_signature = getattr(req, "history_signature", None)
    if history_signature and context.secret:
        # We generate the signature against the conversation history,
        # excluding the user message if it's the last one in the list
        messages = req.copilot_chat_history
        if user_message_count and req.copilot_chat_history[-1].role == ROLE_USER:
            messages = req.copilot_chat_history[:-1]

        is_valid = verify_history_signature(
            history_signature=history_signature,
            messages=messages,
            session_id=req.session_id,
            secret=context.secret,
            version=version,
        )
        if not is_valid:
            structlogger.error(
                "copilot.signing.invalid_history_signature",
                version=version,
                session_id=req.session_id,
            )
            raise InvalidCopilotChatHistorySignature("invalid_history_signature")
        return True

    # Client opted in and this is not the first turn → signature is required
    if not allow_missing:
        structlogger.error(
            "copilot.signing.missing_history_signature",
            version=version,
            session_id=req.session_id,
        )
        raise MissingCopilotChatHistorySignature("missing_history_signature")

    # Soft mode for first turn without signature
    return True


async def create_signature_envelope_for_text(
    req: CopilotRequest,
    text: str,
    category: ResponseCategory,
) -> Optional[ServerSentEvent]:
    """Create a signature SSE envelope for the assistant message.

    Args:
        req: Request object containing copilot chat history and session ID.
        text: Text content of the assistant message.
        category: Response category of the assistant message.

    Returns:
        Optional[ServerSentEvent]: SSE envelope with signature data or None if
        signing is not applicable.
    """
    context = _signing_context()

    # If no server-side secret, signature verification is not enforced
    if context is None:
        return None

    if not getattr(req, "signature_version", None):
        return None

    assistant_message = CopilotChatMessage(
        role=ROLE_COPILOT,
        content=[TextContent(type="text", text=text)],
        response_category=category,
    )
    next_history = [*req.copilot_chat_history, assistant_message]
    version = getattr(req, "signature_version", None) or context.default_version
    signature = compute_history_signature(
        messages=next_history,
        session_id=req.session_id,
        secret=context.secret,
        version=version,
    )

    return ServerSentEvent(
        event="copilot_response",
        data={
            "response_category": "signature",
            "completeness": ResponseCompleteness.COMPLETE.value,
            "version": version,
            "signature": signature,
            "assistant_message": assistant_message.model_dump(),
        },
    )


async def create_signature_envelope_for_handler(
    req: CopilotRequest, handler: CopilotResponseHandler
) -> Optional[ServerSentEvent]:
    """Create a signature SSE envelope from the accumulated handler output.

    Args:
        req: Request object containing copilot chat history and session ID.
        handler: The response handler containing generated responses.

    Returns:
        Optional[ServerSentEvent]: SSE envelope with signature data or None if
        signing is not applicable.
    """
    text, category = extract_full_text_and_category(handler)
    return await create_signature_envelope_for_text(req, text, category)


def extract_full_text_and_category(handler: Any) -> Tuple[str, ResponseCategory]:
    """Extract full text and response category from the handler's generated responses.

    Args:
        handler: The response handler containing generated responses.

    Returns:
        Tuple[str, ResponseCategory]: Concatenated text and the last response category.
    """
    text_parts: List[str] = []
    last_category: Optional[ResponseCategory] = None

    for response in getattr(handler, "generated_responses", []) or []:
        content = getattr(response, "content", None)
        if isinstance(response, GeneratedContent) and content:
            text_parts.append(content)
            category = getattr(response, "response_category", None)
            if category and category not in {
                ResponseCategory.REFERENCE,
                ResponseCategory.REFERENCE_ENTRY,
            }:
                last_category = category

    return "".join(text_parts), (last_category or ResponseCategory.COPILOT)
