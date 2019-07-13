import json
import logging
from unittest.mock import patch, MagicMock

import pytest
import responses
from aioresponses import aioresponses
from sanic import Sanic

import rasa.core.run
from rasa.core import utils
from rasa.core.channels.channel import UserMessage
from rasa.core.channels.telegram import TelegramOutput
from rasa.utils.endpoints import EndpointConfig
from tests.core import utilities
from tests.core.conftest import MOODBOT_MODEL_PATH

# this is needed so that the tests included as code examples look better
from tests.utilities import json_of_latest_request, latest_request

MODEL_PATH = MOODBOT_MODEL_PATH

logger = logging.getLogger(__name__)


def fake_sanic_run(*args, **kwargs):
    """Used to replace `run` method of a Sanic server to avoid hanging."""
    logger.info("Rabatnic: Take this and find Sanic! I want him here by supper time.")


def noop(*args, **kwargs):
    """Just do nothing."""
    pass


def fake_telegram_me(*args, **kwargs):
    """Return a fake telegram user."""
    return {
        "id": 0,
        "first_name": "Test",
        "is_bot": True,
        "username": "YOUR_TELEGRAM_BOT",
    }


def fake_send_message(*args, **kwargs):
    """Fake sending a message."""
    return {"ok": True, "result": {}}


async def test_send_response(default_channel, default_tracker):
    text_only_message = {"text": "hey"}
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
    await default_channel.send_response(default_tracker.sender_id, image_only_message)
    await default_channel.send_response(
        default_tracker.sender_id, text_and_image_message
    )
    await default_channel.send_response(default_tracker.sender_id, custom_json_message)
    collected = default_channel.messages

    assert len(collected) == 6

    # text only message
    assert collected[0] == {"recipient_id": "my-sender", "text": "hey"}

    # image only message
    assert collected[1] == {
        "recipient_id": "my-sender",
        "image": "https://i.imgur.com/nGF1K8f.jpg",
    }

    # text & image combined - will result in two messages
    assert collected[2] == {"recipient_id": "my-sender", "text": "look at this"}
    assert collected[3] == {
        "recipient_id": "my-sender",
        "image": "https://i.imgur.com/T5xVo.jpg",
    }
    assert collected[4] == {"recipient_id": "my-sender", "text": "look at this"}
    assert collected[5] == {
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
                server_url="https://example.com", max_message_limit=3
            )

            r = latest_request(
                mocked, "POST", "https://example.com/webhooks/rest/webhook?stream=true"
            )

            assert r

            b = json_of_latest_request(r)

            assert b == {"message": "Test Input", "sender": "default"}


# USED FOR DOCS - don't rename without changing in the docs
def test_facebook_channel():
    # START DOC INCLUDE
    from rasa.core.channels.facebook import FacebookInput

    input_channel = FacebookInput(
        fb_verify="YOUR_FB_VERIFY",
        # you need tell facebook this token, to confirm your URL
        fb_secret="YOUR_FB_SECRET",  # your app secret
        fb_access_token="YOUR_FB_PAGE_ACCESS_TOKEN"
        # token for the page you subscribed to
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)

    assert routes_list.get("fb_webhook.health").startswith("/webhooks/facebook")
    assert routes_list.get("fb_webhook.webhook").startswith(
        "/webhooks/facebook/webhook"
    )


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
    assert routes_list.get("webexteams_webhook.health").startswith(
        "/webhooks/webexteams"
    )
    assert routes_list.get("webexteams_webhook.webhook").startswith(
        "/webhooks/webexteams/webhook"
    )


# USED FOR DOCS - don't rename without changing in the docs
def test_slack_channel():
    # START DOC INCLUDE
    from rasa.core.channels.slack import SlackInput

    input_channel = SlackInput(
        slack_token="YOUR_SLACK_TOKEN",
        # this is the `bot_user_o_auth_access_token`
        slack_channel="YOUR_SLACK_CHANNEL"
        # the name of your channel to which the bot posts (optional)
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)
    assert routes_list.get("slack_webhook.health").startswith("/webhooks/slack")
    assert routes_list.get("slack_webhook.webhook").startswith(
        "/webhooks/slack/webhook"
    )


# USED FOR DOCS - don't rename without changing in the docs
def test_mattermost_channel():
    # START DOC INCLUDE
    from rasa.core.channels.mattermost import MattermostInput

    input_channel = MattermostInput(
        # this is the url of the api for your mattermost instance
        url="http://chat.example.com/api/v4",
        # the name of your team for mattermost
        team="community",
        # the username of your bot user that will post
        user="user@email.com",
        # messages
        pw="password"
        # the password of your bot user that will post messages
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    routes_list = utils.list_routes(s)
    assert routes_list.get("mattermost_webhook.health").startswith(
        "/webhooks/mattermost"
    )
    assert routes_list.get("mattermost_webhook.webhook").startswith(
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
    assert routes_list.get("botframework_webhook.health").startswith(
        "/webhooks/botframework"
    )
    assert routes_list.get("botframework_webhook.webhook").startswith(
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
    assert routes_list.get("rocketchat_webhook.health").startswith(
        "/webhooks/rocketchat"
    )
    assert routes_list.get("rocketchat_webhook.webhook").startswith(
        "/webhooks/rocketchat/webhook"
    )


# USED FOR DOCS - don't rename without changing in the docs
@pytest.mark.filterwarnings("ignore:unclosed file.*:ResourceWarning")
# telegram channel will try to set a webhook, so we need to mock the api
@patch.object(TelegramOutput, "setWebhook", noop)
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
    assert routes_list.get("telegram_webhook.health").startswith("/webhooks/telegram")
    assert routes_list.get("telegram_webhook.message").startswith(
        "/webhooks/telegram/webhook"
    )


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
    assert routes_list.get("twilio_webhook.health").startswith("/webhooks/twilio")
    assert routes_list.get("twilio_webhook.message").startswith(
        "/webhooks/twilio/webhook"
    )


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
    assert routes_list.get("callback_webhook.health").startswith("/webhooks/callback")
    assert routes_list.get("callback_webhook.webhook").startswith(
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
    assert routes_list.get("socketio_webhook.health").startswith("/webhooks/socketio")
    assert routes_list.get("handle_request").startswith("/socket.io")


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


def test_slack_message_sanitization():
    from rasa.core.channels.slack import SlackInput

    test_uid = 17213535
    target_message_1 = "You can sit here if you want"
    target_message_2 = "Hey, you can sit here if you want !"
    target_message_3 = "Hey, you can sit here if you want!"

    uid_token = "<@{}>".format(test_uid)
    raw_messages = [
        test.format(uid=uid_token)
        for test in [
            "You can sit here {uid} if you want{uid}",
            "{uid} You can sit here if you want{uid} ",
            "{uid}You can sit here if you want {uid}",
            # those last cases may be disputable
            # as we're virtually altering the entered text,
            # but this seem to be the correct course of action
            # (to be decided)
            "You can sit here{uid}if you want",
            "Hey {uid}, you can sit here if you want{uid}!",
            "Hey{uid} , you can sit here if you want {uid}!",
        ]
    ]

    target_messages = [
        target_message_1,
        target_message_1,
        target_message_1,
        target_message_1,
        target_message_2,
        target_message_3,
    ]

    sanitized_messages = [
        SlackInput._sanitize_user_message(message, [test_uid])
        for message in raw_messages
    ]

    # no message that is wrongly sanitized please
    assert (
        len(
            [
                sanitized
                for sanitized, target in zip(sanitized_messages, target_messages)
                if sanitized != target
            ]
        )
        == 0
    )


def test_slack_init_one_parameter():
    from rasa.core.channels.slack import SlackInput

    ch = SlackInput("xoxb-test")
    assert ch.slack_token == "xoxb-test"
    assert ch.slack_channel is None


def test_slack_init_two_parameters():
    from rasa.core.channels.slack import SlackInput

    ch = SlackInput("xoxb-test", "test")
    assert ch.slack_token == "xoxb-test"
    assert ch.slack_channel == "test"


def test_is_slack_message_none():
    from rasa.core.channels.slack import SlackInput

    payload = {}
    slack_message = json.loads(json.dumps(payload))
    assert SlackInput._is_user_message(slack_message) is None


def test_is_slack_message_true():
    from rasa.core.channels.slack import SlackInput

    event = {
        "type": "message",
        "channel": "C2147483705",
        "user": "U2147483697",
        "text": "Hello world",
        "ts": "1355517523",
    }
    payload = json.dumps({"event": event})
    slack_message = json.loads(payload)
    assert SlackInput._is_user_message(slack_message) is True


def test_is_slack_message_false():
    from rasa.core.channels.slack import SlackInput

    event = {
        "type": "message",
        "channel": "C2147483705",
        "user": "U2147483697",
        "text": "Hello world",
        "ts": "1355517523",
        "bot_id": "1355517523",
    }
    payload = json.dumps({"event": event})
    slack_message = json.loads(payload)
    assert SlackInput._is_user_message(slack_message) is False


def test_slackbot_init_one_parameter():
    from rasa.core.channels.slack import SlackBot

    ch = SlackBot("DummyToken")
    assert ch.token == "DummyToken"
    assert ch.slack_channel is None


def test_slackbot_init_two_parameter():
    from rasa.core.channels.slack import SlackBot

    bot = SlackBot("DummyToken", "General")
    assert bot.token == "DummyToken"
    assert bot.slack_channel == "General"


# Use monkeypatch for sending attachments, images and plain text.
@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@responses.activate
async def test_slackbot_send_attachment_only():
    from rasa.core.channels.slack import SlackBot

    responses.add(
        responses.POST,
        "https://slack.com/api/chat.postMessage",
        body='{"ok":true,"purpose":"Testing bots"}',
    )

    bot = SlackBot("DummyToken", "General")
    attachment = {
        "fallback": "Financial Advisor Summary",
        "color": "#36a64f",
        "author_name": "ABE",
        "title": "Financial Advisor Summary",
        "title_link": "http://tenfactorialrocks.com",
        "image_url": "https://r.com/cancel/r12",
        "thumb_url": "https://r.com/cancel/r12",
        "actions": [
            {
                "type": "button",
                "text": "\ud83d\udcc8 Dashboard",
                "url": "https://r.com/cancel/r12",
                "style": "primary",
            },
            {
                "type": "button",
                "text": "\ud83d\udccb Download XL",
                "url": "https://r.com/cancel/r12",
                "style": "danger",
            },
            {
                "type": "button",
                "text": "\ud83d\udce7 E-Mail",
                "url": "https://r.com/cancel/r12",
                "style": "danger",
            },
        ],
        "footer": "Powered by 1010rocks",
        "ts": 1531889719,
    }

    await bot.send_attachment("ID", attachment)

    r = responses.calls[-1]

    assert r.request.body == {
        "channel": ["General"],
        "as_user": ["True"],
        "text": ["Attachment"],
        "attachments": [json.dumps([attachment])],
    }


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@responses.activate
async def test_slackbot_send_attachment_withtext():
    from rasa.core.channels.slack import SlackBot

    responses.add(
        responses.POST,
        "https://slack.com/api/chat.postMessage",
        body='{"ok":true,"purpose":"Testing bots"}',
    )

    bot = SlackBot("DummyToken", "General")
    attachment = {
        "fallback": "Financial Advisor Summary",
        "color": "#36a64f",
        "author_name": "ABE",
        "title": "Financial Advisor Summary",
        "title_link": "http://tenfactorialrocks.com",
        "text": "Here is the summary:",
        "image_url": "https://r.com/cancel/r12",
        "thumb_url": "https://r.com/cancel/r12",
        "actions": [
            {
                "type": "button",
                "text": "\ud83d\udcc8 Dashboard",
                "url": "https://r.com/cancel/r12",
                "style": "primary",
            },
            {
                "type": "button",
                "text": "\ud83d\udccb XL",
                "url": "https://r.com/cancel/r12",
                "style": "danger",
            },
            {
                "type": "button",
                "text": "\ud83d\udce7 E-Mail",
                "url": "https://r.com/cancel/r123",
                "style": "danger",
            },
        ],
        "footer": "Powered by 1010rocks",
        "ts": 1531889719,
    }

    await bot.send_attachment("ID", attachment)

    r = responses.calls[-1]

    assert r.request.body == {
        "channel": ["General"],
        "as_user": ["True"],
        "text": ["Here is the summary:"],
        "attachments": [json.dumps([attachment])],
    }


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@responses.activate
async def test_slackbot_send_image_url():
    from rasa.core.channels.slack import SlackBot

    responses.add(
        responses.POST,
        "https://slack.com/api/chat.postMessage",
        body='{"ok":true,"purpose":"Testing bots"}',
    )

    bot = SlackBot("DummyToken", "General")
    url = "http://www.rasa.net"
    await bot.send_image_url("ID", url)

    r = responses.calls[-1]

    assert r.request.body["as_user"] == ["True"]
    assert r.request.body["channel"] == ["General"]
    assert len(r.request.body["blocks"]) == 1
    assert '"type": "image"' in r.request.body["blocks"][0]
    assert '"alt_text": "http://www.rasa.net"' in r.request.body["blocks"][0]
    assert '"image_url": "http://www.rasa.net"' in r.request.body["blocks"][0]


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@responses.activate
async def test_slackbot_send_text():
    from rasa.core.channels.slack import SlackBot

    responses.add(
        responses.POST,
        "https://slack.com/api/chat.postMessage",
        body='{"ok":true,"purpose":"Testing bots"}',
    )

    bot = SlackBot("DummyToken", "General")
    await bot.send_text_message("ID", "my message")

    r = responses.calls[-1]

    assert r.parsed_body == {
        "as_user": ["True"],
        "channel": ["General"],
        "text": ["my message"],
        "type": ["mrkdwn"],
    }


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
def test_channel_inheritance():
    from rasa.core.channels.channel import RestInput
    from rasa.core.channels.rasa_chat import RasaChatInput

    rasa_input = RasaChatInput("https://example.com")

    s = rasa.core.run.configure_app([RestInput(), rasa_input], port=5004)

    routes_list = utils.list_routes(s)
    assert routes_list.get("custom_webhook_RasaChatInput.health").startswith(
        "/webhooks/rasa"
    )
    assert routes_list.get("custom_webhook_RasaChatInput.receive").startswith(
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
    from rasa.core.channels.channel import RestInput
    import rasa.core

    input_channel = RestInput()

    app = Sanic(__name__)
    rasa.core.channels.channel.register([input_channel], app, route=None)

    routes_list = utils.list_routes(app)
    assert routes_list.get("custom_webhook_RestInput.receive").startswith("/webhook")


def test_channel_registration_with_absolute_url_prefix_overwrites_route():
    from rasa.core.channels.channel import RestInput
    import rasa.core

    input_channel = RestInput()
    test_route = "/absolute_route"
    input_channel.url_prefix = lambda: test_route

    app = Sanic(__name__)
    ignored_base_route = "/should_be_ignored"
    rasa.core.channels.channel.register(
        [input_channel], app, route="/should_be_ignored"
    )

    # Assure that an absolute url returned by `url_prefix` overwrites route parameter
    # given in `register`.
    routes_list = utils.list_routes(app)
    assert routes_list.get("custom_webhook_RestInput.health").startswith(test_route)
    assert ignored_base_route not in routes_list.get("custom_webhook_RestInput.health")


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ({}, "rest"),
        ({"input_channel": None}, "rest"),
        ({"input_channel": "custom"}, "custom"),
    ],
)
def test_extract_input_channel(test_input, expected):
    from rasa.core.channels.channel import RestInput

    input_channel = RestInput()

    fake_request = MagicMock()
    fake_request.json = test_input

    assert input_channel._extract_input_channel(fake_request) == expected
