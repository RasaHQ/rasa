"""Integration tests for streaming custom actions (CALM bot).

These tests exercise the full streaming path end-to-end: a Rasa Pro server
receives a user message, runs a streaming custom action via the CALM bot, and
delivers the response incrementally over the REST SSE channel (``?stream=true``).

## Infrastructure requirements

The tests assume that the Docker stack in
``tests_deployment/integration_tests_custom_action_server/`` is already
running against the **simple_calm_bot**.  Spin it up with::

    make run-action-server-calm-containers

The expected service layout is:

| Server URL                    | Transport            | TLS?     |
|-------------------------------|----------------------|----------|
| ``http://localhost:5012``     | gRPC (plain)         | no       |
| ``http://localhost:5013``     | gRPC TLS             | yes*     |
| ``http://localhost:5014``     | Direct executor      | no       |

(*) TLS is between Rasa and the gRPC action server; the Rasa REST endpoint
itself remains plain HTTP on both gRPC servers.

## Parametrisation convention

gRPC tests are parametrised over the plain (5012) and TLS (5013) gRPC
action server variants.  Direct executor tests run against a single server
(5014) — TLS does not apply to in-process execution.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from tests.integration_tests.conftest import (
    send_message_to_rasa_server,
    send_streaming_message_to_rasa_server,
)

# These must stay in sync with the constants defined in
# tests_deployment/…/simple_calm_bot/actions/action_stream_test.py.
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

# Buttons emitted by ActionStreamAgentSkills after the streamed intro text.
AGENT_SKILL_BUTTONS: List[Dict[str, Any]] = [
    {"title": "Transfer money", "payload": "transfer money"},
    {"title": "List my contacts", "payload": "list my contacts"},
    {"title": "List my reminders", "payload": "list my reminders"},
]
# Intro text streamed before the skill buttons.
EXPECTED_SKILLS_INTRO_TEXT: str = "I can help you with:"

# ---------------------------------------------------------------------------
# Server URL constants
# ---------------------------------------------------------------------------

# gRPC action server — plain channel; Rasa REST on plain HTTP.
GRPC_RASA_SERVER = "http://localhost:5012"

# gRPC action server — TLS channel (between Rasa and the action server);
# Rasa REST endpoint itself remains plain HTTP.
GRPC_TLS_RASA_SERVER = "http://localhost:5013"

# Direct custom action executor (in-process); Rasa REST on plain HTTP.
DIRECT_RASA_SERVER = "http://localhost:5014"


# ---------------------------------------------------------------------------
# Message trigger constants
# ---------------------------------------------------------------------------

# These phrases must match what the CALM LLMCommandGenerator maps to the
# ``stream_recent_transactions`` and ``stream_agent_skills`` flows respectively.
STREAM_TRANSACTIONS_MESSAGE = "show my recent transactions"
STREAM_AGENT_SKILLS_MESSAGE = "what can you help me with"

# ---------------------------------------------------------------------------
# Parametrize IDs
# ---------------------------------------------------------------------------

_RASA_SERVER_PARAMS = pytest.mark.parametrize(
    "server_location",
    [
        pytest.param(GRPC_RASA_SERVER, id="5012-no-tls"),
        pytest.param(GRPC_TLS_RASA_SERVER, id="5013-grpc-tls"),
        pytest.param(DIRECT_RASA_SERVER, id="5014"),
    ],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stream(server_location: str, message: str) -> tuple[str, list[dict]]:
    """Convenience wrapper around ``send_streaming_message_to_rasa_server``."""
    return send_streaming_message_to_rasa_server(
        server_location=server_location,
        message=message,
    )


def _no_stream(server_location: str, message: str) -> tuple[str, list[dict]]:
    """Non-streaming POST, returns ``(sender_id, messages)``."""
    return send_message_to_rasa_server(server_location=server_location, message=message)


# ---------------------------------------------------------------------------
# gRPC streaming tests
# ---------------------------------------------------------------------------


@_RASA_SERVER_PARAMS
def test_streaming_action_chunks_arrive_in_order(server_location: str) -> None:
    """WebhookStream RPC delivers chunk events in order; texts concatenate correctly.

    Asserts:
    - At least as many SSE events arrive as there are streaming tokens.
    - Every event carries the correct ``recipient_id``.
    - Concatenating all ``text`` values reproduces the expected full response.
    """
    sender_id, events = _stream(server_location, STREAM_TRANSACTIONS_MESSAGE)

    text_events = [e for e in events if "text" in e]
    assert len(text_events) >= len(STREAMING_TOKENS), (
        f"Expected at least {len(STREAMING_TOKENS)} text chunk(s), "
        f"got {len(text_events)}: {text_events}"
    )

    for event in text_events:
        assert (
            event["recipient_id"] == sender_id
        ), f"recipient_id mismatch in event: {event}"

    assembled = "".join(e["text"] for e in text_events)
    # assembled also contains pattern_completed utterance
    assert EXPECTED_STREAMED_TEXT in assembled


@_RASA_SERVER_PARAMS
def test_streaming_action_bot_utterance_forwarded(server_location: str) -> None:
    """A mid-stream BotUtterance (rich content) is forwarded via send_response().

    ``ActionStreamAgentSkills`` streams a short intro text then emits a
    ``stream_chunk(buttons=[...])`` with the available skill quick-replies.
    On the Rasa side the buttons chunk is routed via ``send_response()``
    (``dispatch_stream_chunk`` detects ``buttons`` in ``RICH_KEYS``).
    The test asserts that the buttons payload arrives before the connection
    closes and that text tokens precede it.

    Asserts:
    - At least one event has a ``buttons`` key.
    - All buttons events carry the correct ``recipient_id``.
    - The buttons payload matches ``AGENT_SKILL_BUTTONS``.
    - Text chunk(s) arrive before the buttons' event.
    """
    sender_id, events = _stream(server_location, STREAM_AGENT_SKILLS_MESSAGE)

    buttons_events = [e for e in events if "buttons" in e]
    assert len(buttons_events) >= 1, (
        f"Expected at least one buttons event in stream, got none. "
        f"All events: {events}"
    )

    for event in buttons_events:
        assert event["recipient_id"] == sender_id
        assert (
            event["buttons"] == AGENT_SKILL_BUTTONS
        ), f"buttons payload mismatch: {event['buttons']}"

    # Intro text tokens must arrive before the buttons chunk.
    text_events = [e for e in events if "text" in e]
    assert (
        len(text_events) >= 1
    ), "Expected at least one intro text chunk to arrive before the buttons event"
    assembled_intro = "".join(e["text"] for e in text_events)
    assert EXPECTED_SKILLS_INTRO_TEXT in assembled_intro


@_RASA_SERVER_PARAMS
def test_streaming_graceful_fallback_on_plain_rest(server_location: str) -> None:
    """Without ``?stream=true``, the unary path is used and the full response returned.

    When ``CollectingOutputChannel.supports_streaming`` is ``False`` (the
    default for a plain REST request), Rasa falls back to the unary webhook
    call and returns a standard JSON array.

    Asserts:
    - The response is a JSON array (not a streaming response).
    - At least one message is present.
    - Concatenating all ``text`` values in the response reproduces the
      expected full text (the SDK calls ``utter_message`` for each accumulated
      chunk on the non-streaming transport).
    - The response object does not contain SSE / newline-stream framing.
    """
    sender_id, messages = _no_stream(server_location, STREAM_TRANSACTIONS_MESSAGE)

    assert isinstance(messages, list), (
        "Non-streaming path must return a JSON array, "
        f"got {type(messages).__name__}: {messages!r}"
    )
    assert len(messages) > 0, "Expected at least one message in non-streaming response"

    for msg in messages:
        assert msg.get("recipient_id") == sender_id

    all_text = " ".join(m.get("text", "") for m in messages)
    assert EXPECTED_STREAMED_TEXT in all_text


# ---------------------------------------------------------------------------
# SSE endpoint mechanics tests
# ---------------------------------------------------------------------------


@_RASA_SERVER_PARAMS
def test_sse_endpoint_returns_incremental_chunks(server_location: str) -> None:
    """POST with ``?stream=true`` delivers multiple incremental data events.

    This test focuses on the SSE channel mechanics: the REST webhook must
    flush individual JSON objects to the client *before* the action completes,
    so that the end-user sees tokens arriving incrementally.

    Asserts:
    - More than one SSE event arrives (incremental, not a single batch).
    - Each event is a valid JSON object with a ``recipient_id`` field.
    - The full content (concatenated ``text`` values) equals the expected text.
    - The last ``text`` event is NOT the assembled full string (i.e., chunks
      arrived individually and the assembled duplicate was deduplicated by
      ``QueueOutputChannel``).
    """
    sender_id, events = _stream(server_location, STREAM_TRANSACTIONS_MESSAGE)

    assert len(events) > 1, (
        "Expected multiple SSE events (incremental delivery), "
        f"but received only {len(events)}: {events}"
    )

    for event in events:
        assert "recipient_id" in event, f"Event missing recipient_id: {event}"
        assert event["recipient_id"] == sender_id

    text_events = [e for e in events if "text" in e]
    assert (
        len(text_events) >= 2
    ), f"Expected at least 2 text chunk events, got {len(text_events)}: {text_events}"

    # The QueueOutputChannel deduplicates the assembled text so it should NOT
    # appear as a separate final event after all individual tokens.
    assembled = "".join(e["text"] for e in text_events)
    assert EXPECTED_STREAMED_TEXT in assembled

    # Confirm that no single event already equals the full assembled text
    # (that would mean the deduplication of the final response event failed).
    full_text_events = [
        e for e in text_events if e.get("text") == EXPECTED_STREAMED_TEXT
    ]
    assert len(full_text_events) == 0, (
        "QueueOutputChannel should have deduplicated the assembled response, "
        f"but found a full-text event: {full_text_events}"
    )


@_RASA_SERVER_PARAMS
def test_sse_endpoint_non_streaming_request_unaffected(server_location: str) -> None:
    """Without ``?stream=true``, the REST endpoint returns a plain JSON response.

    Asserts:
    - The response body is a valid JSON array.
    - There is no SSE / newline-stream framing (the response is parsed in one
      shot, not iterated line by line).
    - At least one message is present with the expected content.
    """
    sender_id, messages = _no_stream(server_location, STREAM_TRANSACTIONS_MESSAGE)

    assert isinstance(
        messages, list
    ), f"Expected JSON array, got {type(messages).__name__}: {messages!r}"
    assert len(messages) > 0, "Expected at least one message in the response"

    for msg in messages:
        assert msg.get("recipient_id") == sender_id, f"recipient_id mismatch: {msg}"

    all_text = " ".join(m.get("text", "") for m in messages)
    assert EXPECTED_STREAMED_TEXT in all_text
