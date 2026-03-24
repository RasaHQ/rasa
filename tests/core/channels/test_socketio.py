from unittest.mock import AsyncMock, call

import pytest

from rasa.core.channels.socketio import SocketIOInput, SocketIOOutput
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.fixture
def socketio_output(default_tracker: DialogueStateTracker):
    sio_server = AsyncMock()
    mock_input_channel = AsyncMock()
    mock_input_channel.enable_silence_timeout = False
    output_channel = SocketIOOutput(mock_input_channel, sio_server, "bot")
    output_channel.attach_tracker_state(default_tracker)
    return output_channel


async def test_socketio_handles_buttons_without_payload(
    socketio_output: SocketIOOutput,
):
    message = {
        "text": "hello world",
        "buttons": [{"title": "Button1"}],
    }

    # Send the message
    await socketio_output.send_response("recipient_id", message)

    # Check if the socketio object was called with the correct arguments
    expected_calls = [
        call(
            "bot",
            {
                "text": "hello world",
                "quick_replies": [
                    {"content_type": "text", "title": "Button1", "payload": "Button1"}
                ],
            },
            room="recipient_id",
        ),
    ]
    socketio_output.sio_server.emit.assert_has_calls(expected_calls, any_order=False)


async def test_socketio_handles_buttons_with_payload(socketio_output: SocketIOOutput):
    message = {
        "text": "hello world",
        "buttons": [{"title": "Button1", "payload": "/example_intent"}],
    }

    # Send the message
    await socketio_output.send_response("recipient_id", message)

    # Check if the socketio object was called with the correct arguments
    expected_calls = [
        call(
            "bot",
            {
                "text": "hello world",
                "quick_replies": [
                    {
                        "content_type": "text",
                        "title": "Button1",
                        "payload": "/example_intent",
                    }
                ],
            },
            room="recipient_id",
        ),
    ]
    socketio_output.sio_server.emit.assert_has_calls(expected_calls, any_order=False)


async def test_socketio_send_text_message_skips_empty_paragraphs(
    socketio_output: SocketIOOutput,
):
    """Consecutive blank lines yield an empty split segment; it must not be emitted."""
    await socketio_output.send_text_message("recipient_id", "alpha\n\n\n\nbeta")

    assert socketio_output.sio_server.emit.call_count == 2
    socketio_output.sio_server.emit.assert_has_calls(
        [
            call("bot", {"text": "alpha"}, room="recipient_id"),
            call("bot", {"text": "beta"}, room="recipient_id"),
        ],
        any_order=False,
    )


async def test_socketio_send_text_message_whitespace_only_sends_nothing(
    socketio_output: SocketIOOutput,
):
    await socketio_output.send_text_message("recipient_id", "   \n\n   ")

    socketio_output.sio_server.emit.assert_not_called()


async def test_socketio_send_text_message_preserves_paragraph_whitespace(
    socketio_output: SocketIOOutput,
):
    """Strip only detects empty segments; non-empty parts are sent unsplit."""
    await socketio_output.send_text_message("recipient_id", "a \n\n b \n\nc")

    assert socketio_output.sio_server.emit.call_count == 3
    socketio_output.sio_server.emit.assert_has_calls(
        [
            call("bot", {"text": "a "}, room="recipient_id"),
            call("bot", {"text": " b "}, room="recipient_id"),
            call("bot", {"text": "c"}, room="recipient_id"),
        ],
        any_order=False,
    )


async def test_socketio_send_text_with_buttons_whitespace_only_no_emit(
    socketio_output: SocketIOOutput,
):
    await socketio_output.send_text_with_buttons(
        "recipient_id",
        "   \n\n   ",
        [],
    )
    socketio_output.sio_server.emit.assert_not_called()


async def test_socketio_send_text_with_buttons_skips_empty_paragraphs_keeps_buttons(
    socketio_output: SocketIOOutput,
):
    await socketio_output.send_text_with_buttons(
        "recipient_id",
        "first\n\n\n\nsecond",
        [{"title": "Pick", "payload": "/intent"}],
    )
    assert socketio_output.sio_server.emit.call_count == 2
    socketio_output.sio_server.emit.assert_has_calls(
        [
            call("bot", {"text": "first", "quick_replies": []}, room="recipient_id"),
            call(
                "bot",
                {
                    "text": "second",
                    "quick_replies": [
                        {
                            "content_type": "text",
                            "title": "Pick",
                            "payload": "/intent",
                        }
                    ],
                },
                room="recipient_id",
            ),
        ],
        any_order=False,
    )


# =============================================================================
# on_disconnect_callback tests
# =============================================================================


def test_on_disconnect_callback_defaults_to_none():
    channel = SocketIOInput()
    assert channel.on_disconnect_callback is None


def test_on_disconnect_callback_can_be_set():
    channel = SocketIOInput()

    def _cb(sender_id: str) -> None:
        return None

    channel.on_disconnect_callback = _cb
    assert channel.on_disconnect_callback is not None
