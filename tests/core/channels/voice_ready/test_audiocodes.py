import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from sanic import Request, Sanic

from rasa.core import run, utils
from rasa.core.channels.channel import (
    CollectingOutputChannel,
    OutputChannel,
    UserMessage,
)
from rasa.core.channels.voice_ready.audiocodes import (
    CHANNEL_NAME,
    AudiocodesInput,
    AudiocodesOutput,
    Conversation,
    HttpUnauthorized,
    map_call_params,
)
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.shared.exceptions import RasaException
from tests.utilities import filter_logs


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

    text, metadata = conversation._handle_event(event_payload)

    # assert that warning was raised and text is ""
    assert text == ""
    assert metadata == {}
    captured = capsys.readouterr()
    assert "audiocodes.handle.event.no_name_key" in captured.out


async def test_conversation_handle_event_invalid_name() -> None:
    """Test that every event creates an intent in format /vaig_event_<name>"""
    conversation = Conversation(conversation_id="123")
    event_payload = {"name": "invalid"}

    text, metadata = conversation._handle_event(event_payload)

    # assert that an intent was created
    assert text == "/vaig_event_invalid"
    assert metadata == {}


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
    await conversation.handle_activities(
        activities, CHANNEL_NAME, output_channel, on_new_message
    )

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


@pytest.mark.parametrize(
    "message, expected_intent, expected_metadata",
    [
        (
            {
                "conversation": "f010e998-4499-4ddb-80d4-fea137fd7b4d",
                "activities": [
                    {
                        "id": "e54d4dfe-e1ff-4272-8c3d-4ec4f4294681",
                        "timestamp": "2024-12-04T15:07:55.145Z",
                        "type": "event",
                        "name": "noUserInput",
                        "value": 1,
                        "parameters": {
                            "vaigConversationId": "f010e998-4499-4ddb-80d4-fea137fd7b4d",  # noqa: E501
                        },
                    }
                ],
            },
            "/vaig_event_noUserInput",
            {"vaigConversationId": "f010e998-4499-4ddb-80d4-fea137fd7b4d", "value": 1},
        ),
        (
            {
                "conversation": "f010e998-4499-4ddb-80d4-fea137fd7b4d",
                "activities": [
                    {
                        "id": "e54d4dfe-e1ff-4272-8c3d-4ec4f4294681",
                        "timestamp": "2024-12-04T15:07:55.145Z",
                        "type": "event",
                        "name": "noUserInput",
                        "value": 1,
                    }
                ],
            },
            "/vaig_event_noUserInput",
            {"value": 1},
        ),
    ],
)
async def test_handle_no_user_input_event(
    message: Dict[str, Any],
    expected_intent: str,
    expected_metadata: Dict[str, Any],
) -> None:
    """Test handling of noUserInput event from Audiocodes"""
    # Setup
    conversation = Conversation("test_id")
    on_new_message = AsyncMock()
    output_channel = AudiocodesOutput()

    # Execute
    await conversation.handle_activities(
        message, CHANNEL_NAME, output_channel, on_new_message
    )

    # Verify
    on_new_message.assert_called_once()
    user_msg = on_new_message.call_args[0][0]
    assert isinstance(user_msg, UserMessage)
    assert user_msg.text == expected_intent
    assert user_msg.metadata == expected_metadata


async def test_on_activities_returns_immediately(monkeypatch: MonkeyPatch) -> None:
    """Test that on_activities endpoint returns immediately without
    waiting for activity processing.
    """

    # Setup a slow activity handler
    async def slow_on_new_message(message: UserMessage) -> None:
        await asyncio.sleep(1.0)  # Simulate slow processing

    input_channel = AudiocodesInput(
        token="test_token",
        use_websocket=True,
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


async def test_background_task_completes(monkeypatch: MonkeyPatch) -> None:
    """Test that background task created for activity handling
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


@pytest.fixture
def audiocodes_activity_metadata() -> Dict[str, Any]:
    return {
        "vaigConversationId": "some id",
        "caller": "some caller",
        "callee": "some callee",
        "callerDisplayName": "some caller display name",
        "callerHost": "some caller host",
        "calleeHost": "some callee host",
    }


@pytest.fixture
def user_message_metadata(
    audiocodes_activity_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return asdict(map_call_params(audiocodes_activity_metadata))


@pytest.fixture
def audiocodes_message_text() -> str:
    return "Hello, world!"


@pytest.fixture
def audiocodes_message(
    audiocodes_activity_metadata: Dict[str, Any], audiocodes_message_text: str
) -> Dict[str, Any]:
    return {
        "activities": [
            {
                "id": "123",
                "timestamp": "2023-10-01T12:00:00Z",
                "type": "message",
                "text": audiocodes_message_text,
                "parameters": audiocodes_activity_metadata,
            }
        ]
    }


@pytest.fixture
def conversation_id() -> str:
    """Conversation ID fixture."""
    return "123"


@pytest.fixture
def conversation(conversation_id: str) -> Conversation:
    """Conversation fixture."""
    return Conversation(
        conversation_id=conversation_id,
    )


@pytest.fixture
def on_new_message_mock() -> Callable[[UserMessage], Awaitable[Any]]:
    # Mock the on_new_message function to simulate the behavior of the
    # function that handles new messages in the conversation
    _async_mock = AsyncMock()
    return _async_mock


@pytest.fixture
def output_channel_mock() -> MagicMock:
    """Mock of the OutputChannel class."""
    return MagicMock(spec=OutputChannel)


@pytest.fixture
def audiocodes_token() -> str:
    """Fixture for a mock token."""
    return "123"


@pytest.fixture
def audiocodes_disconnect_request(
    audiocodes_token: str,
) -> Request:
    """Fixture for a mock disconnect request."""
    request = MagicMock(spec=Request)
    request.token = audiocodes_token
    request.json = {
        "reason": "user_requested_to_disconnect",
    }
    return request


@pytest.fixture
def audiocodes_input(
    audiocodes_token: str,
) -> AudiocodesInput:
    """Fixture for AudiocodesInput."""
    return AudiocodesInput(
        token=audiocodes_token,
        use_websocket=True,
        keep_alive=120,
        keep_alive_expiration_factor=1.0,
    )


async def test_handle_activities_in_conversation(
    audiocodes_message: Dict[str, Any],
    audiocodes_message_text: str,
    audiocodes_activity_metadata: Dict[str, Any],
    conversation: Conversation,
    conversation_id: str,
    on_new_message_mock: Callable[[UserMessage], Awaitable[Any]],
    output_channel_mock: MagicMock,
) -> None:
    """Tests that handle_activities method correctly creates the UserMessage object"""
    channel_name = "audiocodes"

    await conversation.handle_activities(
        message=audiocodes_message,
        input_channel_name=channel_name,
        output_channel=output_channel_mock,
        on_new_message=on_new_message_mock,
    )

    # Check that the on_new_message function was called with the correct arguments
    on_new_message_call_args = on_new_message_mock.call_args.args
    assert len(on_new_message_call_args) == 1
    assert isinstance(on_new_message_call_args[0], UserMessage)
    assert on_new_message_call_args[0].text == audiocodes_message_text
    assert on_new_message_call_args[0].input_channel == channel_name
    assert on_new_message_call_args[0].output_channel == output_channel_mock
    assert on_new_message_call_args[0].sender_id == conversation_id
    assert on_new_message_call_args[0].metadata == audiocodes_activity_metadata


async def test_handle_activities_with_empty_input_channel_name(
    audiocodes_message: Dict[str, Any],
    audiocodes_message_text: str,
    audiocodes_activity_metadata: Dict[str, Any],
    conversation: Conversation,
    conversation_id: str,
    on_new_message_mock: Callable[[UserMessage], Awaitable[Any]],
    output_channel_mock: MagicMock,
) -> None:
    """Tests that handle_activities method correctly handles empty input channel name."""  # noqa: E501
    channel_name = ""

    with structlog.testing.capture_logs() as caplog:
        await conversation.handle_activities(
            message=audiocodes_message,
            input_channel_name=channel_name,
            output_channel=output_channel_mock,
            on_new_message=on_new_message_mock,
        )

        msg = (
            f"Audiocodes input channel name is empty for conversation {conversation_id}"
        )

        logs = filter_logs(
            caplog,
            "audiocodes.handle.activities.empty_input_channel_name",
            "warning",
            [msg],
        )
        assert len(logs) == 1

    # Check that the on_new_message function was called with the correct arguments
    on_new_message_call_args = on_new_message_mock.call_args.args
    assert len(on_new_message_call_args) == 1
    assert isinstance(on_new_message_call_args[0], UserMessage)
    assert on_new_message_call_args[0].text == audiocodes_message_text
    assert on_new_message_call_args[0].input_channel == channel_name
    assert on_new_message_call_args[0].output_channel == output_channel_mock
    assert on_new_message_call_args[0].sender_id == conversation_id
    assert on_new_message_call_args[0].metadata == audiocodes_activity_metadata


async def test_handle_disconnect(
    audiocodes_disconnect_request: Request,
    conversation: Conversation,
    conversation_id: str,
    on_new_message_mock: Callable[[UserMessage], Awaitable[Any]],
    audiocodes_input: AudiocodesInput,
) -> None:
    audiocodes_input.conversations[conversation_id] = conversation
    """Tests that handle_disconnect method correctly handles disconnect requests."""
    await audiocodes_input._handle_disconnect(
        request=audiocodes_disconnect_request,
        conversation_id=conversation_id,
        on_new_message=on_new_message_mock,
    )

    # Check that the on_new_message function was called with the correct arguments
    on_new_message_call_args = on_new_message_mock.call_args.args
    assert len(on_new_message_call_args) == 1
    assert isinstance(on_new_message_call_args[0], UserMessage)
    assert on_new_message_call_args[0].text == f"{INTENT_MESSAGE_PREFIX}session_end"
    assert on_new_message_call_args[0].input_channel == audiocodes_input.name()
    assert isinstance(
        on_new_message_call_args[0].output_channel, CollectingOutputChannel
    )
    assert on_new_message_call_args[0].sender_id == conversation_id
    assert on_new_message_call_args[0].metadata == {
        "reason": "user_requested_to_disconnect"
    }
