import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from sanic import Sanic

from rasa.core import run, utils
from rasa.core.channels.channel import UserMessage
from rasa.core.channels.voice_ready.audiocodes import (
    AudiocodesInput,
    AudiocodesOutput,
    Conversation,
    HttpUnauthorized,
)
from rasa.shared.exceptions import RasaException


@pytest.mark.parametrize(
    "credentials",
    [
        ({"token": "abc", "keep_alive": "123"}),
        ({"token": 123}),
        ({"token": "123", "use_websocket": "true"}),
        ({"use_websocket": True}),
        ({"token": "123", "keep_alive_expiration_factor": 0.5}),
    ],
)
def test_from_credentials_invalid_format(credentials: Any) -> None:
    with pytest.raises(RasaException):
        AudiocodesInput.from_credentials(credentials)


@pytest.mark.parametrize(
    "credentials",
    [
        ({}),
    ],
)
def test_from_credentials_empty(credentials: Any) -> None:
    with pytest.raises(RasaException):
        AudiocodesInput.from_credentials(credentials)


@pytest.mark.parametrize(
    "credentials",
    [
        ({"token": "abc", "use_websocket": False}),
        ({"token": "abc", "keep_alive": 123}),
    ],
)
def test_from_credentials(credentials: Any) -> None:
    input_channel = AudiocodesInput.from_credentials(credentials)
    assert input_channel is not None
    assert isinstance(input_channel, AudiocodesInput)


async def test_attachment_messages_raise_exceptions() -> None:
    with pytest.raises(RasaException):
        output_channel = AudiocodesOutput()
        await output_channel.send_attachment(recipient_id="123", attachment="xxx")


async def test_image_messages_raise_exceptions() -> None:
    with pytest.raises(RasaException):
        output_channel = AudiocodesOutput()
        await output_channel.send_image_url(recipient_id="123", image="xxx")


def test_audiocodes_input_channel() -> None:
    input_channel = AudiocodesInput(
        token="TOKEN",
        use_websocket=True,
        keep_alive=120,
        keep_alive_expiration_factor=1.5,
    )

    s = run.configure_app([input_channel], port=5004)
    routes_list = utils.list_routes(s)
    print(routes_list)
    assert routes_list["ac_webhook.health"].startswith("/webhooks/audiocodes")
    assert routes_list["ac_webhook.receive"].startswith("/webhooks/audiocodes/webhook")
    assert routes_list["ac_webhook.on_activities"].startswith(
        "/webhooks/audiocodes/conversation/<conversation_id:str>/activities"
    )
    assert routes_list["ac_webhook.disconnect"].startswith(
        "/webhooks/audiocodes/conversation/<conversation_id:str>/disconnect"
    )
    assert routes_list["ac_webhook.keepalive"].startswith(
        "/webhooks/audiocodes/conversation/<conversation_id:str>/keepalive"
    )


async def test_send_text_message() -> None:
    output_channel = AudiocodesOutput()

    await output_channel.send_text_message(recipient_id="123", text="hey")
    assert len(output_channel.messages) == 1
    message = output_channel.messages[0]
    assert "id" in message
    assert "timestamp" in message
    assert "type" in message and message.get("type") == "message"
    assert "text" in message and message.get("text") == "hey"


async def test_send_custom_json_message() -> None:
    from rasa.core.channels.voice_ready.audiocodes import AudiocodesOutput

    output_channel = AudiocodesOutput()

    await output_channel.send_custom_json(
        recipient_id="123", json_message={"key": "val"}
    )
    assert len(output_channel.messages) == 1
    message = output_channel.messages[0]
    assert "id" in message
    assert "timestamp" in message
    assert "key" in message and message.get("key") == "val"


async def test_conversation_handle_event_invalid_payload(
    capsys: CaptureFixture,
) -> None:
    conversation = Conversation(conversation_id="123")
    event_payload = {}

    text = conversation._handle_event(event_payload)

    # assert that warning was raised and text is ""
    assert text == ""
    captured = capsys.readouterr()
    assert "audiocodes.handle.event.no_name_key" in captured.out


async def test_conversation_handle_event_invalid_name(capsys: CaptureFixture) -> None:
    conversation = Conversation(conversation_id="123")
    event_payload = {"name": "invalid"}

    text = conversation._handle_event(event_payload)

    # assert that warning was raised and text is ""
    assert text == ""
    captured = capsys.readouterr()
    assert "audiocodes.handle.event.unknown_event" in captured.out


async def test_handle_startup() -> None:
    """Audiocodes sends this message at the beginning of conversation"""
    # Setup
    conversation = Conversation("test_id")
    on_new_message = AsyncMock()
    output_channel = AudiocodesOutput()

    activities = {
        "conversation": "f010e998-4499-4ddb-80d4-fea137fd7b4d",
        "activities": [
            {
                "id": "e54d4dfe-e1ff-4272-8c3d-4ec4f4294681",
                "timestamp": "2024-12-04T15:07:55.145Z",
                "language": "en-US",
                "type": "event",
                "name": "start",
                "parameters": {
                    "callee": "+493040739365",
                    "calleeHost": "20.113.51.15",
                    "caller": "+491604697810",
                    "callerHost": "sip.telnyx.eu",
                    "callerDisplayName": "+491604697810",
                    "vaigConversationId": "f010e998-4499-4ddb-80d4-fea137fd7b4d",
                },
            }
        ],
    }

    # Execute
    await conversation.handle_activities(activities, output_channel, on_new_message)

    on_new_message.assert_called_once()
    user_msg = on_new_message.call_args[0][0]
    assert isinstance(user_msg, UserMessage)
    assert user_msg.text == "/session_start"
    assert user_msg.sender_id == "test_id"
    assert user_msg.metadata == {
        "bot_host": "20.113.51.15",
        "bot_phone": "+493040739365",
        "call_id": "f010e998-4499-4ddb-80d4-fea137fd7b4d",
        "direction": None,
        "stream_id": None,
        "user_host": "sip.telnyx.eu",
        "user_name": "+491604697810",
        "user_phone": "+491604697810",
    }


async def test_on_activities_returns_immediately(monkeypatch: MonkeyPatch) -> None:
    """
    Test that on_activities endpoint returns immediately without
    waiting for activity processing.
    """

    # Setup a slow activity handler
    async def slow_on_new_message(message: UserMessage) -> None:
        await asyncio.sleep(1.0)  # Simulate slow processing

    input_channel = AudiocodesInput(
        token="test_token",
        use_websocket=False,
        keep_alive=120,
        keep_alive_expiration_factor=1.5,
    )

    conversation_id = "test_conv"
    input_channel.conversations[conversation_id] = Conversation(conversation_id)

    # Create Sanic test client
    app = Sanic("test_app")
    blueprint = input_channel.blueprint(slow_on_new_message)
    app.blueprint(blueprint)
    test_client = app.asgi_client

    # Prepare request data
    url_prefix = "rasa.core.channels.voice_ready.audiocodes"
    url = f"{url_prefix}/conversation/{conversation_id}/activities"
    data = {
        "activities": [
            {
                "id": "test_id",
                "type": "message",
                "text": "hello",
                "parameters": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
    }
    headers = {"Authorization": "test_token"}

    # Measure response time
    start_time = datetime.now()
    _, response = await test_client.post(url, json=data, headers=headers)
    elapsed_time = (datetime.now() - start_time).total_seconds()

    # Response should return immediately
    assert elapsed_time < 0.5  # Much less than the 1.0s sleep
    assert response.status == 200
    assert "activities" in response.json


async def test_background_task_completes(monkeypatch: MonkeyPatch) -> None:
    """
    Test that background task created for activity handling
    completes successfully.
    """

    processed_messages = []

    async def tracking_on_new_message(message: UserMessage) -> None:
        processed_messages.append(message.text)
        await asyncio.sleep(0.1)  # Small delay to ensure it's running async

    input_channel = AudiocodesInput(
        token="test_token",
        use_websocket=False,
        keep_alive=120,
        keep_alive_expiration_factor=1.5,
    )

    conversation_id = "test_conv"
    input_channel.conversations[conversation_id] = Conversation(conversation_id)

    # Create Sanic test client
    app = Sanic("test_app")
    blueprint = input_channel.blueprint(tracking_on_new_message)
    app.blueprint(blueprint)
    test_client = app.asgi_client

    # Prepare request data
    url_prefix = "rasa.core.channels.voice_ready.audiocodes"
    url = f"{url_prefix}/conversation/{conversation_id}/activities"
    data = {
        "activities": [
            {
                "id": "test_id",
                "type": "message",
                "text": "test message",
                "parameters": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
    }
    headers = {"Authorization": "Bearer test_token"}

    # Make request
    _, response = await test_client.post(url, json=data, headers=headers)
    assert response.status == 200

    # Wait for background task to complete
    await asyncio.sleep(0.2)

    # Verify message was processed
    assert len(processed_messages) == 1
    assert processed_messages[0] == "test message"

    # Verify task cleanup
    assert len(input_channel.background_tasks[conversation_id]) == 0


async def test_invalid_token_raises_error() -> None:
    """Test that requests with invalid tokens are rejected."""
    input_channel = AudiocodesInput(
        token="correct_token",
        use_websocket=False,
        keep_alive=120,
        keep_alive_expiration_factor=1.5,
    )

    # Create Sanic test client
    app = Sanic("test_app")
    blueprint = input_channel.blueprint(AsyncMock())
    app.blueprint(blueprint)
    test_client = app.asgi_client

    url_prefix = "rasa.core.channels.voice_ready.audiocodes"
    url = f"{url_prefix}/webhook"

    # Test with correct token
    headers = {"Authorization": "Bearer correct_token"}
    _, response = await test_client.get(url, headers=headers)

    assert response.status == 200

    # Test with wrong token
    headers = {"Authorization": "wrong_token"}
    _, response = await test_client.get(url, headers=headers)

    assert response.status == 401

    # Test with missing token
    _, response = await test_client.get(url)

    assert response.status == 401


def test_check_token() -> None:
    input_channel = AudiocodesInput(
        token="correct_token",
        use_websocket=False,
        keep_alive=120,
        keep_alive_expiration_factor=1.5,
    )

    # Test with correct token
    # assert no exception is raised
    try:
        input_channel._check_token("correct_token")
    except HttpUnauthorized:
        pytest.fail("HttpUnauthorized raised unexpectedly!")

    # Test with wrong token, Expect HttpUnauthorized exception
    with pytest.raises(HttpUnauthorized):
        input_channel._check_token("wrong_token")

    # Test with missing token
    with pytest.raises(HttpUnauthorized):
        input_channel._check_token(None)
