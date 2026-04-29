import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from sanic import Request, Sanic
from sanic.exceptions import NotFound, ServerError

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
    WebsocketOutput,
    map_call_params,
)
from rasa.core.channels.voice_ready.utils import CallParameters
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


def test_from_credentials_validation_error_message() -> None:
    """Invalid schema raises RasaException with 'Invalid credentials'."""
    with pytest.raises(RasaException) as exc_info:
        AudiocodesInput.from_credentials(
            {"token": "abc", "keep_alive_expiration_factor": 0.5}
        )
    assert "Invalid credentials" in str(exc_info.value)


def test_map_call_params() -> None:
    """map_call_params maps Audiocodes parameters to CallParameters."""
    parameters = {
        "vaigConversationId": "conv-1",
        "caller": "+123",
        "callee": "+456",
        "callerDisplayName": "Alice",
        "callerHost": "host1",
        "calleeHost": "host2",
    }
    result = map_call_params(parameters)
    assert isinstance(result, CallParameters)
    assert result.call_id == "conv-1"
    assert result.user_phone == "+123"
    assert result.bot_phone == "+456"
    assert result.user_name == "Alice"
    assert result.user_host == "host1"
    assert result.bot_host == "host2"


def test_map_call_params_partial() -> None:
    """map_call_params handles missing keys with None."""
    result = map_call_params({})
    assert result.call_id is None
    assert result.user_phone is None
    assert result.bot_phone is None
    assert result.user_name is None
    assert result.user_host is None
    assert result.bot_host is None


def test_conversation_get_metadata() -> None:
    """Conversation.get_metadata returns activity parameters."""
    assert Conversation.get_metadata({"parameters": {"key": "val"}}) == {"key": "val"}
    assert Conversation.get_metadata({}) is None
    assert Conversation.get_metadata({"type": "message"}) is None


def test_conversation_update_sets_last_activity() -> None:
    """Conversation.update sets last_activity to current time."""
    conv = Conversation(conversation_id="c1")
    assert hasattr(conv, "last_activity")
    before = datetime.now(timezone.utc)
    conv.update()
    after = datetime.now(timezone.utc)
    assert before <= conv.last_activity <= after


async def test_handle_event_dtmf() -> None:
    """_handle_event returns DTMF intent and value metadata."""
    conversation = Conversation(conversation_id="123")
    event = {"name": "DTMF", "value": "5"}
    text, metadata = conversation._handle_event(event)
    assert text == f"{INTENT_MESSAGE_PREFIX}vaig_event_DTMF"
    assert metadata == {"value": "5"}


async def test_handle_event_other_with_parameters_and_value() -> None:
    """_handle_event for other events includes parameters and value."""
    conversation = Conversation(conversation_id="123")
    event = {
        "name": "noUserInput",
        "value": 1,
        "parameters": {"vaigConversationId": "id1"},
    }
    text, metadata = conversation._handle_event(event)
    assert text == f"{INTENT_MESSAGE_PREFIX}vaig_event_noUserInput"
    assert metadata == {"vaigConversationId": "id1", "value": 1}


def test_conversation_is_active_conversation_active() -> None:
    """is_active_conversation returns True when within delta."""
    conv = Conversation(conversation_id="c1")
    now = datetime.now(timezone.utc)
    delta = timedelta(seconds=60)
    assert conv.is_active_conversation(now, delta) is True


def test_conversation_is_active_conversation_inactive(
    capsys: CaptureFixture,
) -> None:
    """is_active_conversation returns False and logs when past delta."""
    conv = Conversation(conversation_id="c1")
    conv.last_activity = datetime.now(timezone.utc) - timedelta(seconds=200)
    now = datetime.now(timezone.utc)
    delta = timedelta(seconds=60)
    assert conv.is_active_conversation(now, delta) is False
    captured = capsys.readouterr()
    assert "audiocodes.conversation.inactive" in captured.out


async def test_handle_activities_duplicate_activity_logs_and_skips(
    capsys: CaptureFixture,
) -> None:
    """Duplicate activity id is skipped and warning is logged."""
    conversation = Conversation(conversation_id="c1")
    on_new_message = AsyncMock()
    output_channel = MagicMock(spec=OutputChannel)
    message = {
        "activities": [
            {"id": "dup-id", "type": "message", "text": "first", "parameters": {}},
            {"id": "dup-id", "type": "message", "text": "second", "parameters": {}},
        ]
    }
    await conversation.handle_activities(
        message, CHANNEL_NAME, output_channel, on_new_message
    )
    on_new_message.assert_called_once()
    assert on_new_message.call_args[0][0].text == "first"
    captured = capsys.readouterr()
    assert "audiocodes.handle.activities.duplicate_activity" in captured.out


async def test_handle_activities_unknown_activity_type_logs_and_skips(
    capsys: CaptureFixture,
) -> None:
    """Unknown activity type is skipped and warning is logged."""
    conversation = Conversation(conversation_id="c1")
    on_new_message = AsyncMock()
    output_channel = MagicMock(spec=OutputChannel)
    message = {
        "activities": [
            {"id": "id1", "type": "unknown_type", "text": "ignored", "parameters": {}},
        ]
    }
    await conversation.handle_activities(
        message, CHANNEL_NAME, output_channel, on_new_message
    )
    on_new_message.assert_not_called()
    captured = capsys.readouterr()
    assert "audiocodes.handle.activities.unknown_activity_type" in captured.out


async def test_handle_activities_empty_text_skipped() -> None:
    """Activity that yields empty text (e.g. event with no name) is skipped."""
    conversation = Conversation(conversation_id="c1")
    on_new_message = AsyncMock()
    output_channel = MagicMock(spec=OutputChannel)
    message = {
        "activities": [
            {"id": "id1", "type": "event"},  # no "name" -> _handle_event returns "", {}
        ]
    }
    await conversation.handle_activities(
        message, CHANNEL_NAME, output_channel, on_new_message
    )
    on_new_message.assert_not_called()


async def test_handle_activities_on_new_message_raises_sends_hangup() -> None:
    """When on_new_message raises, hangup event is sent via output channel."""
    conversation = Conversation(conversation_id="c1")
    output_channel = MagicMock(spec=OutputChannel)
    output_channel.send_custom_json = AsyncMock()

    async def failing_handler(_: UserMessage) -> None:
        raise RuntimeError("simulated failure")

    message = {
        "activities": [
            {"id": "id1", "type": "message", "text": "hi", "parameters": {}},
        ]
    }
    await conversation.handle_activities(
        message, CHANNEL_NAME, output_channel, failing_handler
    )
    output_channel.send_custom_json.assert_called_once()
    call_args = output_channel.send_custom_json.call_args
    assert call_args[0][0] == "c1"
    assert call_args[0][1]["type"] == "event"
    assert call_args[0][1]["name"] == "hangup"
    assert "An error occurred" in call_args[0][1]["text"]


async def test_attachment_messages_raise_exceptions() -> None:
    with pytest.raises(RasaException):
        output_channel = AudiocodesOutput()
        await output_channel.send_attachment(recipient_id="123", attachment="xxx")


async def test_image_messages_raise_exceptions() -> None:
    with pytest.raises(RasaException):
        output_channel = AudiocodesOutput()
        await output_channel.send_image_url(recipient_id="123", image="xxx")


def test_audiocodes_input_channel(
    audiocodes_input_token_channel_factory: Callable[[bool], AudiocodesInput],
) -> None:
    input_channel = audiocodes_input_token_channel_factory(use_websocket=True)

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
        "language": None,
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


async def test_on_activities_returns_immediately(
    audiocodes_input_token_channel_factory: Callable[[bool], AudiocodesInput],
) -> None:
    """Test that on_activities endpoint returns immediately without
    waiting for activity processing.
    """

    # Setup a slow activity handler
    async def slow_on_new_message(message: UserMessage) -> None:
        await asyncio.sleep(1.0)  # Simulate slow processing

    input_channel = audiocodes_input_token_channel_factory(use_websocket=True)

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
    headers = {"Authorization": "Bearer TOKEN"}

    # Measure response time
    start_time = datetime.now()
    _, response = await test_client.post(url, json=data, headers=headers)
    elapsed_time = (datetime.now() - start_time).total_seconds()

    # Response should return immediately
    assert elapsed_time < 0.5  # Much less than the 1.0s sleep
    assert response.status == 200


async def test_background_task_completes(
    audiocodes_input_token_channel_factory: Callable[[bool], AudiocodesInput],
) -> None:
    """Test that background task created for activity handling
    completes successfully.
    """
    processed_messages = []

    async def tracking_on_new_message(message: UserMessage) -> None:
        processed_messages.append(message.text)
        await asyncio.sleep(0.1)  # Small delay to ensure it's running async

    input_channel = audiocodes_input_token_channel_factory(use_websocket=False)

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
    headers = {"Authorization": "Bearer TOKEN"}

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


async def test_invalid_token_raises_error(
    audiocodes_input_token_channel_factory: Callable[[bool], AudiocodesInput],
) -> None:
    """Test that requests with invalid tokens are rejected."""
    input_channel = audiocodes_input_token_channel_factory(use_websocket=False)

    # Create Sanic test client
    app = Sanic("test_app")
    blueprint = input_channel.blueprint(AsyncMock())
    app.blueprint(blueprint)
    test_client = app.asgi_client

    url_prefix = "rasa.core.channels.voice_ready.audiocodes"
    url = f"{url_prefix}/webhook"

    # Test with correct token
    headers = {"Authorization": "Bearer TOKEN"}
    _, response = await test_client.get(url, headers=headers)

    assert response.status == 200

    # Test with wrong token
    headers = {"Authorization": "wrong_token"}
    _, response = await test_client.get(url, headers=headers)

    assert response.status == 401

    # Test with missing token
    _, response = await test_client.get(url)

    assert response.status == 401


def test_check_token(
    audiocodes_input_token_channel_factory: Callable[[bool], AudiocodesInput],
) -> None:
    input_channel = audiocodes_input_token_channel_factory(use_websocket=False)

    # Test with correct token
    # assert no exception is raised
    try:
        input_channel._check_token("TOKEN")
    except HttpUnauthorized:
        pytest.fail("HttpUnauthorized raised unexpectedly!")

    # Test with wrong token, Expect HttpUnauthorized exception
    with pytest.raises(HttpUnauthorized):
        input_channel._check_token("wrong_token")

    # Test with missing token
    with pytest.raises(HttpUnauthorized):
        input_channel._check_token(None)


@pytest.mark.asyncio
async def test_create_task_tracks_and_cleans_up(
    audiocodes_input_token_channel_factory: Callable[[bool], AudiocodesInput],
) -> None:
    """_create_task adds task to background_tasks and callback removes it."""
    input_channel = audiocodes_input_token_channel_factory(use_websocket=True)
    conv_id = "conv-1"
    input_channel.conversations[conv_id] = Conversation(conv_id)

    async def dummy_coro() -> None:
        pass

    task = input_channel._create_task(conv_id, dummy_coro())
    assert task in input_channel.background_tasks[conv_id]
    await task
    assert len(input_channel.background_tasks[conv_id]) == 0


@pytest.mark.asyncio
async def test_set_scheduler_job(
    audiocodes_input_token_channel_factory: Callable[[bool], AudiocodesInput],
    monkeypatch: MonkeyPatch,
) -> None:
    """_set_scheduler_job adds interval job for clean_old_conversations."""
    input_channel = audiocodes_input_token_channel_factory(use_websocket=True)
    mock_job = MagicMock()
    mock_scheduler = MagicMock()
    mock_scheduler.add_job = MagicMock(return_value=mock_job)

    async def return_scheduler() -> MagicMock:
        return mock_scheduler

    monkeypatch.setattr(
        "rasa.core.channels.voice_ready.audiocodes.jobs.scheduler",
        return_scheduler,
    )
    await input_channel._set_scheduler_job()
    assert input_channel.scheduler_job is mock_job
    mock_scheduler.add_job.assert_called_once_with(
        input_channel.clean_old_conversations, "interval", minutes=10
    )


def test_get_conversation_not_found(
    audiocodes_input_token_channel_factory: Callable[[bool], AudiocodesInput],
) -> None:
    """_get_conversation raises NotFound when conversation does not exist."""
    input_channel = audiocodes_input_token_channel_factory(use_websocket=False)
    with pytest.raises(NotFound, match="Conversation not found"):
        input_channel._get_conversation("TOKEN", "nonexistent-conv")


def test_clean_old_conversations_removes_inactive(
    audiocodes_input_token_channel_factory: Callable[[bool], AudiocodesInput],
) -> None:
    """clean_old_conversations removes conversations past keep_alive * factor."""
    input_channel = audiocodes_input_token_channel_factory(use_websocket=True)
    conv1 = Conversation("c1")
    conv1.last_activity = datetime.now(timezone.utc) - timedelta(seconds=200)
    conv2 = Conversation("c2")
    input_channel.conversations["c1"] = conv1
    input_channel.conversations["c2"] = conv2

    input_channel.clean_old_conversations()

    assert "c1" not in input_channel.conversations
    assert "c2" in input_channel.conversations


def test_handle_start_conversation_success_with_websocket(
    audiocodes_input_token_channel_factory: Callable[[bool], AudiocodesInput],
) -> None:
    """handle_start_conversation returns urls including websocketURL."""
    input_channel = audiocodes_input_token_channel_factory(use_websocket=True)
    body = {"conversation": "conv-123"}
    result = input_channel.handle_start_conversation(body)
    assert result["activitiesURL"] == "conversation/conv-123/activities"
    assert result["disconnectURL"] == "conversation/conv-123/disconnect"
    assert result["refreshURL"] == "conversation/conv-123/keepalive"
    assert result["expiresSeconds"] == 120
    assert result["websocketURL"] == "conversation/conv-123/websocket"
    assert "conv-123" in input_channel.conversations


def test_handle_start_conversation_success_without_websocket(
    audiocodes_input_token_channel_factory: Callable[[bool], AudiocodesInput],
) -> None:
    """handle_start_conversation omits websocketURL when use_websocket False."""
    input_channel = audiocodes_input_token_channel_factory(use_websocket=False)
    body = {"conversation": "conv-456"}
    result = input_channel.handle_start_conversation(body)
    assert "websocketURL" not in result


def test_handle_start_conversation_already_exists_raises(
    audiocodes_input_token_channel_factory: Callable[[bool], AudiocodesInput],
) -> None:
    """handle_start_conversation raises ServerError when conversation already exists."""
    input_channel = audiocodes_input_token_channel_factory(use_websocket=False)
    input_channel.conversations["existing"] = Conversation("existing")
    with pytest.raises(ServerError, match="Conversation already exists"):
        input_channel.handle_start_conversation({"conversation": "existing"})


# URL prefix for audiocodes blueprint (matches existing tests)
_AC_BLUEPRINT_URL_PREFIX = "rasa.core.channels.voice_ready.audiocodes"


def _ac_path(path: str) -> str:
    """Build request path for audiocodes blueprint (asgi_client, no leading slash)."""
    return f"{_AC_BLUEPRINT_URL_PREFIX}{path}"


@pytest.fixture
def audiocodes_input_token_channel_factory(
    monkeypatch: MonkeyPatch,
) -> Callable[[bool], AudiocodesInput]:
    """Callable(use_websocket) -> AudiocodesInput(
    token=TOKEN, keep_alive=120, factor=1.5).

    License validation is mocked before construction.
    """
    monkeypatch.setattr(
        "rasa.core.channels.voice_ready.audiocodes.validate_voice_license_scope",
        MagicMock(),
    )

    def _make(use_websocket: bool) -> AudiocodesInput:
        return AudiocodesInput(
            token="TOKEN",
            use_websocket=use_websocket,
            keep_alive=120,
            keep_alive_expiration_factor=1.5,
        )

    return _make


@pytest.fixture
def audiocodes_blueprint_input(monkeypatch: MonkeyPatch) -> AudiocodesInput:
    """AudiocodesInput for Sanic blueprint tests; token matches Bearer in requests.

    License validation is mocked before construction (fixtures run before @patch).
    """
    monkeypatch.setattr(
        "rasa.core.channels.voice_ready.audiocodes.validate_voice_license_scope",
        MagicMock(),
    )
    return AudiocodesInput(
        token="token",
        use_websocket=False,
        keep_alive=120,
        keep_alive_expiration_factor=1.5,
    )


@pytest.fixture
def audiocodes_blueprint_app(audiocodes_blueprint_input: AudiocodesInput) -> Sanic:
    """Sanic app with audiocodes blueprint registered (asgi_client route tests)."""
    app = Sanic("test")
    app.blueprint(
        audiocodes_blueprint_input.blueprint(AsyncMock()),
        url_prefix=_AC_BLUEPRINT_URL_PREFIX,
    )
    return app


@pytest.mark.asyncio
async def test_health_route(audiocodes_blueprint_app: Sanic) -> None:
    """Health route returns status ok."""
    _, response = await audiocodes_blueprint_app.asgi_client.get(_ac_path("/"))
    assert response.status == 200
    assert response.json == {"status": "ok"}


@pytest.mark.asyncio
async def test_receive_get_returns_ac_bot_api(
    audiocodes_blueprint_app: Sanic,
) -> None:
    """Receive GET returns ac-bot-api type and success."""
    _, response = await audiocodes_blueprint_app.asgi_client.get(
        _ac_path("/webhook"),
        headers={"Authorization": "Bearer token"},
    )
    assert response.status == 200
    assert response.json.get("type") == "ac-bot-api"
    assert response.json.get("success") is True


@patch(
    "rasa.core.channels.voice_ready.audiocodes.jobs.scheduler",
    new_callable=AsyncMock,
)
@pytest.mark.asyncio
async def test_receive_post_starts_conversation(
    mock_scheduler: AsyncMock,
    audiocodes_blueprint_app: Sanic,
) -> None:
    """Receive POST with conversation body returns urls."""
    mock_scheduler.return_value = MagicMock(add_job=MagicMock(return_value=MagicMock()))
    _, response = await audiocodes_blueprint_app.asgi_client.post(
        _ac_path("/webhook"),
        json={"conversation": "new-conv-1"},
        headers={"Authorization": "Bearer token"},
    )
    assert response.status == 200
    data = response.json
    assert "activitiesURL" in data
    assert "new-conv-1" in data["activitiesURL"]


@pytest.mark.asyncio
async def test_keepalive_route(
    audiocodes_blueprint_input: AudiocodesInput,
    audiocodes_blueprint_app: Sanic,
) -> None:
    """Keepalive route validates token and returns empty json."""
    audiocodes_blueprint_input.conversations["conv-1"] = Conversation("conv-1")
    _, response = await audiocodes_blueprint_app.asgi_client.post(
        _ac_path("/conversation/conv-1/keepalive"),
        headers={"Authorization": "Bearer token"},
    )
    assert response.status == 200
    assert response.json == {}


async def test_audiocodes_output_hangup() -> None:
    """hangup adds hangup event to messages."""
    output = AudiocodesOutput()
    await output.hangup(recipient_id="conv-1")
    assert len(output.messages) == 1
    assert output.messages[0]["type"] == "event"
    assert output.messages[0]["name"] == "hangup"


async def test_audiocodes_output_send_text_with_buttons() -> None:
    """send_text_with_buttons uses concise format (text + button titles)."""
    output = AudiocodesOutput()
    await output.send_text_with_buttons(
        recipient_id="conv-1",
        text="Choose one",
        buttons=[{"title": "A", "payload": "/a"}, {"title": "B", "payload": "/b"}],
    )
    assert len(output.messages) >= 1
    # Concise format appends ". A, B" to text (see send_text_with_buttons_concise)
    assert any("Choose one" in str(m.get("text", "")) for m in output.messages)


@pytest.mark.asyncio
async def test_websocket_output_do_add_message_sends_via_ws() -> None:
    """WebsocketOutput.do_add_message sends JSON via websocket."""
    mock_ws = MagicMock()
    mock_ws.send = AsyncMock()
    output = WebsocketOutput(ws=mock_ws, conversation_id="conv-1")
    await output.do_add_message({"type": "message", "text": "hello"})
    mock_ws.send.assert_called_once()
    payload = mock_ws.send.call_args[0][0]
    import json as json_module

    data = json_module.loads(payload)
    assert data["conversation"] == "conv-1"
    assert len(data["activities"]) == 1
    assert data["activities"][0]["text"] == "hello"


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
