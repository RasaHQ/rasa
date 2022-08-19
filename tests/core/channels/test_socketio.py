import json
import logging

import pytest
from unittest import mock

from rasa.core import utils
from rasa.core.channels.channel import UserMessage

logger = logging.getLogger(__name__)


def test_socketio_channel():

    from rasa.core.channels.socketio import SocketIOInput
    import rasa.core

    input_channel = SocketIOInput(
        user_message_evt="user_uttered",
        bot_message_evt="bot_uttered",
        session_persistence=False,
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)

    routes_list = utils.list_routes(s)
    print(f"routes: {routes_list.get('socketio_webhook.health')}")

    assert routes_list.get("socketio_webhook.health").startswith("/webhooks/socketio")


def AsyncMock(*args, **kwargs):
    """Return a mock asynchronous function."""
    m = mock.MagicMock(*args, **kwargs)

    async def mock_coro(*args, **kwargs):
        return m(*args, **kwargs)

    mock_coro.mock = m
    return mock_coro


@pytest.mark.asyncio
async def test_socketio_output_channel_functions():

    from rasa.core.channels.socketio import SocketIOOutput

    # handle_request = AsyncMock()
    mgr = mock.MagicMock()
    mgr.emit = AsyncMock()

    output_channel = SocketIOOutput(sio=mgr, bot_message_evt="bot_uttered")

    await output_channel.send_text_message(recipient_id="Greg Stephens", text="Test:")
    output_channel.messages = {"text": "Test:"}

    assert len(output_channel.messages) == 1
    assert output_channel.messages["text"] == "Test:"
