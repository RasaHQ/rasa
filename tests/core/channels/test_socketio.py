from unittest.mock import AsyncMock
from rasa.core.channels.socketio import SocketIOOutput


async def test_socketio_handles_buttons_without_payload():
    message = {
        "text": "hello world",
        "buttons": [{"title": "Button1"}],
    }

    # Create a socketio output channel with a dummy socketio object
    socket_io_mock = AsyncMock()
    socket_io_output = SocketIOOutput(socket_io_mock, "bot")

    # Send the message
    await socket_io_output.send_response("recipient_id", message)

    # Check if the socketio object was called with the correct arguments
    socket_io_mock.emit.assert_called_once_with(
        "bot",
        {
            "text": "hello world",
            "quick_replies": [
                {"content_type": "text", "title": "Button1", "payload": "Button1"}
            ],
        },
        room="recipient_id",
    )


async def test_socketio_handles_buttons_with_payload():
    message = {
        "text": "hello world",
        "buttons": [{"title": "Button1", "payload": "/example_intent"}],
    }

    # Create a socketio output channel with a dummy socketio object
    socket_io_mock = AsyncMock()
    socket_io_output = SocketIOOutput(socket_io_mock, "bot")

    # Send the message
    await socket_io_output.send_response("recipient_id", message)

    # Check if the socketio object was called with the correct arguments
    socket_io_mock.emit.assert_called_once_with(
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
    )
