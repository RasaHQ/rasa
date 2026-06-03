import logging
from http import HTTPStatus
from typing import Any, Dict, List, Text

import pytest
from sanic import Sanic

from rasa.core.channels.channel import UserMessage
from rasa.core.channels.telnyx import TelnyxInput, TelnyxOutput

logger = logging.getLogger(__name__)


def telnyx_message_webhook(text: Text = "Hello") -> Dict[Text, Any]:
    """Create a Telnyx inbound message webhook payload."""
    return {
        "data": {
            "event_type": "message.received",
            "payload": {
                "id": "message-id",
                "from": {"phone_number": "+15551234567"},
                "to": [{"phone_number": "+15557654321"}],
                "text": text,
            },
        }
    }


@pytest.mark.asyncio
async def test_telnyx_health():
    """Telnyx channel exposes a health route."""
    input_channel = TelnyxInput(
        api_key="TELNYX_API_KEY",
        from_number="+15557654321",
    )

    async def on_new_message(message: UserMessage) -> None:
        pass

    app = Sanic("telnyx_health_test_app")
    app.blueprint(
        input_channel.blueprint(on_new_message), url_prefix="/webhooks/telnyx"
    )

    _, res = await app.asgi_client.get("/webhooks/telnyx/")

    assert res.status == HTTPStatus.OK
    assert res.json == {"status": "ok"}


@pytest.mark.asyncio
async def test_telnyx_receive_message():
    """Telnyx message webhooks are passed to Rasa."""
    input_channel = TelnyxInput(
        api_key="TELNYX_API_KEY",
        from_number="+15557654321",
    )
    messages: List[UserMessage] = []

    async def on_new_message(message: UserMessage) -> None:
        messages.append(message)

    app = Sanic("telnyx_test_app")
    app.blueprint(
        input_channel.blueprint(on_new_message), url_prefix="/webhooks/telnyx"
    )

    _, res = await app.asgi_client.post(
        "/webhooks/telnyx/webhook",
        json=telnyx_message_webhook(),
    )

    assert res.status == HTTPStatus.NO_CONTENT
    assert len(messages) == 1
    assert messages[0].text == "Hello"
    assert messages[0].sender_id == "+15551234567"
    assert messages[0].input_channel == "telnyx"
    assert messages[0].message_id == "message-id"
    assert isinstance(messages[0].output_channel, TelnyxOutput)
    assert messages[0].metadata == telnyx_message_webhook()["data"]["payload"]


@pytest.mark.asyncio
async def test_telnyx_ignores_non_message_received_webhook():
    """Telnyx webhooks for other events are ignored."""
    input_channel = TelnyxInput(
        api_key="TELNYX_API_KEY",
        from_number="+15557654321",
    )
    messages: List[UserMessage] = []

    async def on_new_message(message: UserMessage) -> None:
        messages.append(message)

    app = Sanic("telnyx_ignored_event_test_app")
    app.blueprint(
        input_channel.blueprint(on_new_message), url_prefix="/webhooks/telnyx"
    )

    _, res = await app.asgi_client.post(
        "/webhooks/telnyx/webhook",
        json={
            "data": {
                "event_type": "message.finalized",
                "payload": {"id": "message-id"},
            }
        },
    )

    assert res.status == HTTPStatus.NO_CONTENT
    assert messages == []


@pytest.mark.asyncio
async def test_telnyx_send_text_message(monkeypatch):
    """Telnyx text messages are sent through the Messages API payload."""
    sent_messages = []
    output_channel = TelnyxOutput("TELNYX_API_KEY", "+15557654321")

    async def mock_send_message(message_data: Dict[Text, Any]) -> None:
        sent_messages.append(message_data)

    monkeypatch.setattr(output_channel, "_send_message", mock_send_message)

    await output_channel.send_text_message("+15551234567", "hello\n\nagain")

    assert sent_messages == [
        {
            "from": "+15557654321",
            "to": "+15551234567",
            "text": "hello",
        },
        {
            "from": "+15557654321",
            "to": "+15551234567",
            "text": "again",
        },
    ]


@pytest.mark.asyncio
async def test_telnyx_send_image_url(monkeypatch):
    """Telnyx image messages are sent as MMS media URLs."""
    sent_messages = []
    output_channel = TelnyxOutput("TELNYX_API_KEY", "+15557654321")

    async def mock_send_message(message_data: Dict[Text, Any]) -> None:
        sent_messages.append(message_data)

    monkeypatch.setattr(output_channel, "_send_message", mock_send_message)

    await output_channel.send_image_url("+15551234567", "https://example.com/image.png")

    assert sent_messages == [
        {
            "from": "+15557654321",
            "to": "+15551234567",
            "text": "",
            "media_urls": ["https://example.com/image.png"],
        }
    ]
