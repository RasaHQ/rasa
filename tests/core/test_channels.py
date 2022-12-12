import logging

import jwt
from typing import Dict
from unittest.mock import patch, MagicMock

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from aiogram.utils.exceptions import TelegramAPIError
from aiohttp import ClientTimeout
from aioresponses import aioresponses
from sanic import Sanic

import rasa.core.run
import rasa.core.channels.channel
from rasa.core import utils
from rasa.core.channels import RasaChatInput, console
from rasa.core.channels.channel import UserMessage
from rasa.core.channels.rasa_chat import (
    JWT_USERNAME_KEY,
    CONVERSATION_ID_KEY,
    INTERACTIVE_LEARNING_PERMISSION,
)
from rasa.core.channels.telegram import TelegramOutput
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig
from tests.core import utilities

# this is needed so that the tests included as code examples look better
from tests.utilities import json_of_latest_request, latest_request

logger = logging.getLogger(__name__)


async def noop(*args, **kwargs):
    """Just do nothing."""
    pass


async def test_send_response(default_channel, default_tracker):
    text_only_message = {"text": "hey"}
    multiline_text_message = {
        "text": "This message should come first:  \n\n"
        "This is message two  \nThis as well\n\n"
    }
    image_only_message = {"image": "https://i.imgur.com/nGF1K8f.jpg"}
    text_and_image_message = {
        "text": "look at this",
        "image": "https://i.imgur.com/T5xVo.jpg",
    }
    custom_json_message = {
        "text": "look at this",
        "custom": {"some_random_arg": "value", "another_arg": "value2"},
    }

    await default_channel.send_response(default_tracker.sender_id, text_only_message)
    await default_channel.send_response(
        default_tracker.sender_id, multiline_text_message
    )
    await default_channel.send_response(default_tracker.sender_id, image_only_message)
    await default_channel.send_response(
        default_tracker.sender_id, text_and_image_message
    )
    await default_channel.send_response(default_tracker.sender_id, custom_json_message)
    collected = default_channel.messages

    assert len(collected) == 8

    # text only message
    assert collected[0] == {"recipient_id": "my-sender", "text": "hey"}

    # multiline text message, should split on '\n\n'
    assert collected[1] == {
        "recipient_id": "my-sender",
        "text": "This message should come first:  ",
    }
    assert collected[2] == {
        "recipient_id": "my-sender",
        "text": "This is message two  \nThis as well",
    }

    # image only message
    assert collected[3] == {
        "recipient_id": "my-sender",
        "image": "https://i.imgur.com/nGF1K8f.jpg",
    }

    # text & image combined - will result in two messages
    assert collected[4] == {"recipient_id": "my-sender", "text": "look at this"}
    assert collected[5] == {
        "recipient_id": "my-sender",
        "image": "https://i.imgur.com/T5xVo.jpg",
    }
    assert collected[6] == {"recipient_id": "my-sender", "text": "look at this"}
    assert collected[7] == {
        "recipient_id": "my-sender",
        "custom": {"some_random_arg": "value", "another_arg": "value2"},
    }


async def test_console_input():
    from rasa.core.channels import console

    # Overwrites the input() function and when someone else tries to read
    # something from the command line this function gets called.
    with utilities.mocked_cmd_input(console, text="Test Input"):
        with aioresponses() as mocked:
            mocked.post(
                "https://example.com/webhooks/rest/webhook?stream=true",
                repeat=True,
                payload={},
            )

            await console.record_messages(
                server_url="https://example.com",
                max_message_limit=3,
                sender_id="default",
            )

            r = latest_request(
                mocked, "POST", "https://example.com/webhooks/rest/webhook?stream=true"
            )

            assert r

            b = json_of_latest_request(r)

            assert b == {"message": "Test Input", "sender": "default"}


# USED FOR DOCS - don't rename without changing in the docs
def test_webexteams_channel():
    # START DOC INCLUDE
    from rasa.core.channels.webexteams import WebexTeamsInput

    input_channel = WebexTeamsInput(
        access_token="YOUR_ACCESS_TOKEN",
        # this is the `bot access token`
        room="YOUR_WEBEX_ROOM"
        # the name of your channel to which the bot posts (optional)
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)
    assert routes_list["webexteams_webhook.health"].startswith("/webhooks/webexteams")
    assert routes_list["webexteams_webhook.webhook"].startswith(
        "/webhooks/webexteams/webhook"
    )


# USED FOR DOCS - don't rename without changing in the docs
def test_slack_channel():
    # START DOC INCLUDE
    from rasa.core.channels.slack import SlackInput

    input_channel = SlackInput(
        # this is the Slack Bot Token
        slack_token="YOUR_SLACK_TOKEN",
        # the name of your channel to which the bot posts (optional)
        slack_channel="YOUR_SLACK_CHANNEL",
        # signing secret from slack to verify incoming webhook messages
        slack_signing_secret="YOUR_SIGNING_SECRET",
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)
    assert routes_list["slack_webhook.health"].startswith("/webhooks/slack")
    assert routes_list["slack_webhook.webhook"].startswith("/webhooks/slack/webhook")


# USED FOR DOCS - don't rename without changing in the docs
def test_mattermost_channel():
    # START DOC INCLUDE
    from rasa.core.channels.mattermost import MattermostInput

    input_channel = MattermostInput(
        # this is the url of the api for your mattermost instance
        url="http://chat.example.com/api/v4",
        # the bot token of the bot account that will post messages
        token="xxxxx",
        # the password of your bot user that will post messages
        # the webhook-url your bot should listen for messages
        webhook_url="YOUR_WEBHOOK_URL",
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)
    assert routes_list["mattermost_webhook.health"].startswith("/webhooks/mattermost")
    assert routes_list["mattermost_webhook.webhook"].startswith(
        "/webhooks/mattermost/webhook"
    )


# USED FOR DOCS - don't rename without changing in the docs
def test_botframework_channel():
    # START DOC INCLUDE
    from rasa.core.channels.botframework import BotFrameworkInput

    input_channel = BotFrameworkInput(
        # you get this from your Bot Framework account
        app_id="MICROSOFT_APP_ID",
        # also from your Bot Framework account
        app_password="MICROSOFT_APP_PASSWORD",
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)
    assert routes_list["botframework_webhook.health"].startswith(
        "/webhooks/botframework"
    )
    assert routes_list["botframework_webhook.webhook"].startswith(
        "/webhooks/botframework/webhook"
    )


# USED FOR DOCS - don't rename without changing in the docs
def test_rocketchat_channel():
    # START DOC INCLUDE
    from rasa.core.channels.rocketchat import RocketChatInput

    input_channel = RocketChatInput(
        # your bots rocket chat user name
        user="yourbotname",
        # the password for your rocket chat bots account
        password="YOUR_PASSWORD",
        # url where your rocket chat instance is running
        server_url="https://demo.rocket.chat",
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)
    assert routes_list["rocketchat_webhook.health"].startswith("/webhooks/rocketchat")
    assert routes_list["rocketchat_webhook.webhook"].startswith(
        "/webhooks/rocketchat/webhook"
    )


# USED FOR DOCS - don't rename without changing in the docs
@pytest.mark.filterwarnings("ignore:unclosed file.*:ResourceWarning")
# telegram channel will try to set a webhook, so we need to mock the api
@patch.object(TelegramOutput, "set_webhook", noop)
def test_telegram_channel():
    # START DOC INCLUDE
    from rasa.core.channels.telegram import TelegramInput

    input_channel = TelegramInput(
        # you get this when setting up a bot
        access_token="123:YOUR_ACCESS_TOKEN",
        # this is your bots username
        verify="YOUR_TELEGRAM_BOT",
        # the url your bot should listen for messages
        webhook_url="YOUR_WEBHOOK_URL",
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)
    assert routes_list["telegram_webhook.health"].startswith("/webhooks/telegram")
    assert routes_list["telegram_webhook.message"].startswith(
        "/webhooks/telegram/webhook"
    )


def test_telegram_channel_raise_rasa_exception_webhook_not_set(
    monkeypatch: MonkeyPatch,
):
    from rasa.core.channels.telegram import TelegramInput

    input_channel = TelegramInput(
        # you get this when setting up a bot
        access_token="123:YOUR_ACCESS_TOKEN",
        # this is your bots username
        verify="YOUR_TELEGRAM_BOT",
        # the url your bot should listen for messages
        webhook_url="",
    )

    monkeypatch.setattr(
        rasa.core.channels.telegram.TelegramOutput,
        "set_webhook",
        MagicMock(side_effect=TelegramAPIError("Error from Telegram.")),
    )

    with pytest.raises(RasaException) as e:
        rasa.core.run.configure_app([input_channel], port=5004)

    assert "Failed to set channel webhook:" in str(e.value)


async def test_handling_of_integer_user_id():
    # needed for telegram to work properly as this channel sends integer ids,
    # but we expect the sender_id to be a string everywhere else

    assert UserMessage("hello", sender_id=123).sender_id == "123"


# USED FOR DOCS - don't rename without changing in the docs
def test_twilio_channel():
    # START DOC INCLUDE
    from rasa.core.channels.twilio import TwilioInput

    input_channel = TwilioInput(
        # you get this from your twilio account
        account_sid="YOUR_ACCOUNT_SID",
        # also from your twilio account
        auth_token="YOUR_AUTH_TOKEN",
        # a number associated with your twilio account
        twilio_number="YOUR_TWILIO_NUMBER",
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)
    assert routes_list["twilio_webhook.health"].startswith("/webhooks/twilio")
    assert routes_list["twilio_webhook.message"].startswith("/webhooks/twilio/webhook")


# USED FOR DOCS - don't rename without changing in the docs
def test_callback_channel():
    # START DOC INCLUDE
    from rasa.core.channels.callback import CallbackInput

    input_channel = CallbackInput(
        # URL Core will call to send the bot responses
        endpoint=EndpointConfig("http://localhost:5004")
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)
    assert routes_list["callback_webhook.health"].startswith("/webhooks/callback")
    assert routes_list["callback_webhook.webhook"].startswith(
        "/webhooks/callback/webhook"
    )


# USED FOR DOCS - don't rename without changing in the docs
def test_socketio_channel():
    # START DOC INCLUDE
    from rasa.core.channels.socketio import SocketIOInput

    input_channel = SocketIOInput(
        # event name for messages sent from the user
        user_message_evt="user_uttered",
        # event name for messages sent from the bot
        bot_message_evt="bot_uttered",
        # socket.io namespace to use for the messages
        namespace=None,
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)
    assert routes_list["socketio_webhook.health"].startswith("/webhooks/socketio")
    assert routes_list["handle_request"].startswith("/socket.io")


def test_socketio_channel_metadata():
    from rasa.core.channels.socketio import SocketIOInput

    input_channel = SocketIOInput(
        # event name for messages sent from the user
        user_message_evt="user_uttered",
        # event name for messages sent from the bot
        bot_message_evt="bot_uttered",
        # optional metadata key name
        metadata_key="customData",
        # socket.io namespace to use for the messages
        namespace=None,
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)
    assert routes_list["socketio_webhook.health"].startswith("/webhooks/socketio")
    assert routes_list["handle_request"].startswith("/socket.io")


async def test_socketio_channel_jwt_authentication():
    from rasa.core.channels.socketio import SocketIOInput

    public_key = "random_key123"
    jwt_algorithm = "HS256"
    auth_token = jwt.encode({"payload": "value"}, public_key, algorithm=jwt_algorithm)

    input_channel = SocketIOInput(
        # event name for messages sent from the user
        user_message_evt="user_uttered",
        # event name for messages sent from the bot
        bot_message_evt="bot_uttered",
        # socket.io namespace to use for the messages
        namespace=None,
        # public key for JWT methods
        jwt_key=public_key,
        # method used for the signature of the JWT authentication payload
        jwt_method=jwt_algorithm,
    )

    assert input_channel.jwt_key == public_key
    assert input_channel.jwt_algorithm == jwt_algorithm
    assert rasa.core.channels.channel.decode_bearer_token(
        auth_token, input_channel.jwt_key, input_channel.jwt_algorithm
    )


async def test_socketio_channel_jwt_authentication_invalid_key(
    caplog: LogCaptureFixture,
):
    from rasa.core.channels.socketio import SocketIOInput

    public_key = "random_key123"
    invalid_public_key = "my_invalid_key"
    jwt_algorithm = "HS256"
    invalid_auth_token = jwt.encode(
        {"payload": "value"}, invalid_public_key, algorithm=jwt_algorithm
    )

    input_channel = SocketIOInput(
        # event name for messages sent from the user
        user_message_evt="user_uttered",
        # event name for messages sent from the bot
        bot_message_evt="bot_uttered",
        # socket.io namespace to use for the messages
        namespace=None,
        # public key for JWT methods
        jwt_key=public_key,
        # method used for the signature of the JWT authentication payload
        jwt_method=jwt_algorithm,
    )

    assert input_channel.jwt_key == public_key
    assert input_channel.jwt_algorithm == jwt_algorithm

    with caplog.at_level(logging.ERROR):
        rasa.core.channels.channel.decode_bearer_token(
            invalid_auth_token, input_channel.jwt_key, input_channel.jwt_algorithm
        )

    assert any("JWT public key invalid." in message for message in caplog.messages)


async def test_callback_calls_endpoint():
    from rasa.core.channels.callback import CallbackOutput

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/callback",
            repeat=True,
            headers={"Content-Type": "application/json"},
        )

        output = CallbackOutput(EndpointConfig("https://example.com/callback"))

        await output.send_response(
            "test-id", {"text": "Hi there!", "image": "https://example.com/image.jpg"}
        )

        r = latest_request(mocked, "post", "https://example.com/callback")

        assert r

        image = r[-1].kwargs["json"]
        text = r[-2].kwargs["json"]

        assert image["recipient_id"] == "test-id"
        assert image["image"] == "https://example.com/image.jpg"

        assert text["recipient_id"] == "test-id"
        assert text["text"] == "Hi there!"


def test_botframework_attachments():
    from rasa.core.channels.botframework import BotFrameworkInput
    from copy import deepcopy

    ch = BotFrameworkInput("app_id", "app_pass")

    payload = {
        "type": "message",
        "id": "123",
        "channelId": "msteams",
        "serviceUrl": "https://smba.trafficmanager.net/emea/",
        "from": {"id": "12:123", "name": "Rasa", "aadObjectId": "123"},
        "conversation": {
            "conversationType": "personal",
            "tenantId": "123",
            "id": "a:123",
        },
        "recipient": {"id": "12:123", "name": "Rasa chat"},
    }
    assert ch.add_attachments_to_metadata(payload, None) is None

    attachments = [
        {
            "contentType": "application/vnd.microsoft.teams.file.download.info",
            "content": {
                "downloadUrl": "https://test.sharepoint.com/personal/rasa/123",
                "uniqueId": "123",
                "fileType": "csv",
            },
            "contentUrl": "https://test.sharepoint.com/personal/rasa/123",
            "name": "rasa-test.csv",
        }
    ]
    payload["attachments"] = attachments

    assert ch.add_attachments_to_metadata(payload, None) == {"attachments": attachments}

    metadata = {"test": 1, "bigger_test": {"key": "value"}}
    updated_metadata = deepcopy(metadata)
    updated_metadata.update({"attachments": attachments})

    assert ch.add_attachments_to_metadata(payload, metadata) == updated_metadata


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
def test_channel_inheritance():
    from rasa.core.channels import RestInput
    from rasa.core.channels.rasa_chat import RasaChatInput

    rasa_input = RasaChatInput("https://example.com")

    s = rasa.core.run.configure_app([RestInput(), rasa_input], port=5004)

    routes_list = utils.list_routes(s)
    assert routes_list["custom_webhook_RasaChatInput.health"].startswith(
        "/webhooks/rasa"
    )
    assert routes_list["custom_webhook_RasaChatInput.receive"].startswith(
        "/webhooks/rasa/webhook"
    )


def test_int_sender_id_in_user_message():
    from rasa.core.channels.channel import UserMessage

    # noinspection PyTypeChecker
    message = UserMessage("A text", sender_id=1234567890)

    assert message.sender_id == "1234567890"


def test_int_message_id_in_user_message():
    from rasa.core.channels.channel import UserMessage

    # noinspection PyTypeChecker
    message = UserMessage("B text", message_id=987654321)

    assert message.message_id == "987654321"


async def test_send_elements_without_buttons():
    from rasa.core.channels.channel import OutputChannel

    async def test_message(sender, message):
        assert sender == "user"
        assert message == "a : b"

    channel = OutputChannel()
    channel.send_text_message = test_message
    await channel.send_elements("user", [{"title": "a", "subtitle": "b"}])


def test_newsline_strip():
    from rasa.core.channels.channel import UserMessage

    message = UserMessage("\n/restart\n")

    assert message.text == "/restart"


def test_register_channel_without_route():
    """Check we properly connect the input channel blueprint if route is None"""
    from rasa.core.channels import RestInput
    import rasa.core

    input_channel = RestInput()

    app = Sanic("test_channels")
    rasa.core.channels.channel.register([input_channel], app, route=None)

    routes_list = utils.list_routes(app)
    assert routes_list["test_channels.custom_webhook_RestInput.receive"].startswith(
        "/webhook"
    )


def test_channel_registration_with_absolute_url_prefix_overwrites_route():
    from rasa.core.channels import RestInput
    import rasa.core

    input_channel = RestInput()
    test_route = "/absolute_route"
    input_channel.url_prefix = lambda: test_route

    app = Sanic("test_channels")
    ignored_base_route = "/should_be_ignored"
    rasa.core.channels.channel.register(
        [input_channel], app, route="/should_be_ignored"
    )

    # Assure that an absolute url returned by `url_prefix` overwrites route parameter
    # given in `register`.
    routes_list = utils.list_routes(app)
    assert routes_list["test_channels.custom_webhook_RestInput.health"].startswith(
        test_route
    )
    assert ignored_base_route not in routes_list.get(
        "test_channels.custom_webhook_RestInput.health"
    )


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ({}, "rest"),
        ({"input_channel": None}, "rest"),
        ({"input_channel": "custom"}, "custom"),
    ],
)
def test_extract_input_channel(test_input, expected):
    from rasa.core.channels import RestInput

    input_channel = RestInput()

    fake_request = MagicMock()
    fake_request.json = test_input

    assert input_channel._extract_input_channel(fake_request) == expected


async def test_rasa_chat_input():
    from rasa.core.channels import RasaChatInput

    rasa_x_api_url = "https://rasa-x.com:5002"
    rasa_chat_input = RasaChatInput(rasa_x_api_url)
    public_key = "random_key123"
    jwt_algorithm = "RS256"
    with aioresponses() as mocked:
        mocked.get(
            rasa_x_api_url + "/version",
            payload={"keys": [{"key": public_key, "alg": jwt_algorithm}]},
            repeat=True,
            status=200,
        )
        await rasa_chat_input._fetch_public_key()
        assert rasa_chat_input.jwt_key == public_key
        assert rasa_chat_input.jwt_algorithm == jwt_algorithm


@pytest.mark.parametrize(
    "jwt, message",
    [
        ({JWT_USERNAME_KEY: "abc"}, {CONVERSATION_ID_KEY: "abc"}),
        (
            {
                JWT_USERNAME_KEY: "abc",
                "scopes": ["a", "b", INTERACTIVE_LEARNING_PERMISSION],
            },
            {CONVERSATION_ID_KEY: "test"},
        ),
    ],
)
def test_has_user_permission_to_send_messages_to_conversation(jwt: Dict, message: Dict):
    assert RasaChatInput._has_user_permission_to_send_messages_to_conversation(
        jwt, message
    )


@pytest.mark.parametrize(
    "jwt, message",
    [
        ({JWT_USERNAME_KEY: "abc"}, {CONVERSATION_ID_KEY: "xyz"}),
        (
            {JWT_USERNAME_KEY: "abc", "scopes": ["a", "b"]},
            {CONVERSATION_ID_KEY: "test"},
        ),
    ],
)
def test_has_user_permission_to_send_messages_to_conversation_without_permission(
    jwt: Dict, message: Dict
):
    assert not RasaChatInput._has_user_permission_to_send_messages_to_conversation(
        jwt, message
    )


def test_set_console_stream_reading_timeout(monkeypatch: MonkeyPatch):
    expected = 100
    monkeypatch.setenv(console.STREAM_READING_TIMEOUT_ENV, str(100))

    assert console._get_stream_reading_timeout() == ClientTimeout(expected)
