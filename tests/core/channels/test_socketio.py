from unittest.mock import AsyncMock, call

import pytest

from rasa.core.channels.socketio import SocketIOOutput
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
