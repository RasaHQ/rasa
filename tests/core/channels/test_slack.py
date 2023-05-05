from http import HTTPStatus
import json
import logging
import time
from typing import Any, Dict, Text
from unittest import mock
from unittest.mock import Mock, patch

from aioresponses import aioresponses
import pytest
from sanic.compat import Header
from sanic.request import Request

from rasa.core.channels import SlackInput
from rasa.core.channels.channel import UserMessage
from rasa.shared.exceptions import InvalidConfigException
from tests.utilities import json_of_latest_request, latest_request

logger = logging.getLogger(__name__)

SLACK_TEST_ATTACHMENT = {
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


def test_slack_metadata():
    user = "user1"
    channel = "channel1"
    authed_users = ["XXXXXXX", "YYYYYYY", "ZZZZZZZ"]
    ts = "1579802617.000800"
    header = {"content-type": "application/json"}
    direct_message_event = {
        "authed_users": authed_users,
        "event": {
            "client_msg_id": "XXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",
            "type": "message",
            "text": "hello world",
            "user": user,
            "ts": ts,
            "team": "XXXXXXXXX",
            "blocks": [
                {
                    "type": "rich_text",
                    "block_id": "XXXXX",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [{"type": "text", "text": "hi"}],
                        }
                    ],
                }
            ],
            "channel": channel,
            "event_ts": "1579802617.000800",
            "channel_type": "im",
        },
    }

    input_channel = SlackInput(
        slack_token="YOUR_SLACK_TOKEN",
        slack_channel="YOUR_SLACK_CHANNEL",
        slack_signing_secret="foobar",
    )

    r = Mock()
    r.json = direct_message_event
    r.headers = header
    metadata = input_channel.get_metadata(request=r)
    assert metadata["out_channel"] == channel
    assert metadata["users"] == authed_users
    assert metadata["thread_id"] == ts


def test_slack_form_metadata():
    user = "user1"
    channel = "channel1"
    authed_user = "XXXXXXX"
    ts = "1579802617.000800"
    header = {"content-type": "application/x-www-form-urlencoded"}
    payload = {
        "type": "block_actions",
        "user": {"id": authed_user, "username": user, "name": "name"},
        "channel": {"id": channel},
        "message": {
            "type": "message",
            "text": "text",
            "user": authed_user,
            "ts": ts,
            "blocks": [
                {
                    "type": "actions",
                    "block_id": "XXXXX",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": "XXXXX",
                            "text": {"type": "plain_text", "text": "text"},
                            "value": "value",
                        }
                    ],
                }
            ],
        },
    }
    form_event = {"payload": [json.dumps(payload)]}

    input_channel = SlackInput(
        slack_token="YOUR_SLACK_TOKEN",
        slack_channel="YOUR_SLACK_CHANNEL",
        slack_signing_secret="foobar",
    )

    r = Mock()
    r.form = form_event
    r.headers = header
    metadata = input_channel.get_metadata(request=r)
    assert metadata["out_channel"] == channel
    assert metadata["users"][0] == authed_user
    assert metadata["thread_id"] == ts


def test_slack_metadata_missing_keys():
    channel = "channel1"
    ts = "1579802617.000800"
    header = {"content-type": "application/json"}
    direct_message_event = {
        "event": {
            "client_msg_id": "XXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX",
            "type": "message",
            "text": "hello world",
            "ts": ts,
            "team": "XXXXXXXXX",
            "blocks": [
                {
                    "type": "rich_text",
                    "block_id": "XXXXX",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [{"type": "text", "text": "hi"}],
                        }
                    ],
                }
            ],
            "channel": channel,
            "event_ts": "1579802617.000800",
            "channel_type": "im",
        }
    }

    input_channel = SlackInput(
        slack_token="YOUR_SLACK_TOKEN",
        slack_channel="YOUR_SLACK_CHANNEL",
        slack_signing_secret="foobar",
    )

    r = Mock()
    r.json = direct_message_event
    r.headers = header
    metadata = input_channel.get_metadata(request=r)
    assert metadata["users"] == []
    assert metadata["out_channel"] == channel
    assert metadata["thread_id"] == ts


def test_slack_form_metadata_missing_keys():
    channel = "channel1"
    ts = "1579802617.000800"
    header = {"content-type": "application/x-www-form-urlencoded"}
    payload = {
        "type": "block_actions",
        "channel": {"id": channel},
        "message": {
            "type": "message",
            "text": "text",
            "ts": ts,
            "blocks": [
                {
                    "type": "actions",
                    "block_id": "XXXXX",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": "XXXXX",
                            "text": {"type": "plain_text", "text": "text"},
                            "value": "value",
                        }
                    ],
                }
            ],
        },
    }
    form_event = {"payload": [json.dumps(payload)]}

    input_channel = SlackInput(
        slack_token="YOUR_SLACK_TOKEN",
        slack_channel="YOUR_SLACK_CHANNEL",
        slack_signing_secret="foobar",
    )

    r = Mock()
    r.form = form_event
    r.headers = header
    metadata = input_channel.get_metadata(request=r)
    assert metadata["users"] == []
    assert metadata["out_channel"] == channel
    assert metadata["thread_id"] == ts


def test_slack_no_metadata():
    input_channel = SlackInput(
        slack_token="YOUR_SLACK_TOKEN",
        slack_channel="YOUR_SLACK_CHANNEL",
        slack_signing_secret="foobar",
    )

    r = Mock()
    metadata = input_channel.get_metadata(request=r)
    assert metadata == {}


def test_slack_message_sanitization():
    test_uid = "17213535"
    target_message_1 = "You can sit here if you want"
    target_message_2 = "Hey, you can sit here if you want !"
    target_message_3 = "Hey, you can sit here if you want!"
    target_message_4 = "convert garbled url to vicdb-f.net"
    target_message_5 = "convert multiple garbled url to vicdb-f.net. Also eemdb-p.net"

    uid_token = f"<@{test_uid}>"
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
            "convert garbled url to <http://vicdb-f.net|vicdb-f.net>",
            "convert multiple garbled url to <http://vicdb-f.net|vicdb-f.net>. "
            "Also <http://eemdb-p.net|eemdb-p.net>",
        ]
    ]

    target_messages = [
        target_message_1,
        target_message_1,
        target_message_1,
        target_message_1,
        target_message_2,
        target_message_3,
        target_message_4,
        target_message_5,
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


def test_escape_called():
    with patch("re.escape") as mock_escape:
        input_text = "Some text"
        uids_to_remove = ["uid1"]
        SlackInput._sanitize_user_message(input_text, uids_to_remove)

        # Check if re.escape was called with the expected argument
        mock_escape.assert_called_with("uid1")


def test_slack_init_token_parameter():
    ch = SlackInput("xoxb-test", slack_signing_secret="foobar")
    assert ch.slack_token == "xoxb-test"
    assert ch.slack_channel is None


def test_slack_init_token_channel_parameters():
    ch = SlackInput("xoxb-test", "test", slack_signing_secret="foobar")
    assert ch.slack_token == "xoxb-test"
    assert ch.slack_channel == "test"
    assert ch.conversation_granularity == "sender"


def test_slack_init_token_channel_conversation_granularity_parameters():
    ch = SlackInput(
        "xoxb-test",
        "test",
        slack_signing_secret="foobar",
        conversation_granularity="channel",
    )
    assert ch.slack_token == "xoxb-test"
    assert ch.slack_channel == "test"
    assert ch.conversation_granularity == "channel"


def test_slack_init_token_channel_threads_parameters():
    ch = SlackInput(
        "xoxb-test",
        "test",
        slack_signing_secret="foobar",
        use_threads=True,
        conversation_granularity="thread",
    )
    assert ch.slack_token == "xoxb-test"
    assert ch.slack_channel == "test"
    assert ch.use_threads is True
    assert ch.conversation_granularity == "thread"


def test_get_conversation_id_sender_id():
    ch = SlackInput(
        "xoxb-test",
        "test",
        slack_signing_secret="foobar",
        use_threads=True,
        conversation_granularity="sender",
    )
    conversation_id = ch._get_conversation_id(
        "test_sender_id", "test_channel_id", "test_thread_id"
    )
    assert conversation_id == "test_sender_id"


def test_get_conversation_id_channel_id():
    ch = SlackInput(
        "xoxb-test",
        "test",
        slack_signing_secret="foobar",
        use_threads=True,
        conversation_granularity="channel",
    )
    conversation_id = ch._get_conversation_id("test_sender_id", "test_channel_id", None)
    assert conversation_id == "test_sender_id_test_channel_id"

    conversation_id = ch._get_conversation_id("test_sender_id", None, "test_thread_id")
    assert conversation_id == "test_sender_id"


def test_get_conversation_id_thread_id():
    ch = SlackInput(
        "xoxb-test",
        "test",
        slack_signing_secret="foobar",
        use_threads=True,
        conversation_granularity="thread",
    )
    conversation_id = ch._get_conversation_id(
        "test_sender_id", "test_channel_id", "test_thread_id"
    )
    assert conversation_id == "test_sender_id_test_channel_id_test_thread_id"

    conversation_id = ch._get_conversation_id("test_sender_id", None, "test_thread_id")
    assert conversation_id == "test_sender_id"

    conversation_id = ch._get_conversation_id("test_sender_id", "test_channel_id", None)
    assert conversation_id == "test_sender_id"

    conversation_id = ch._get_conversation_id("test_sender_id", None, None)
    assert conversation_id == "test_sender_id"


def test_is_slack_message_none():
    payload = {}
    slack_message = json.loads(json.dumps(payload))
    assert SlackInput._is_user_message(slack_message) is False


def test_is_slack_message_true():
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
    assert ch.client.token == "DummyToken"
    assert ch.slack_channel is None


def test_slackbot_init_two_parameter():
    from rasa.core.channels.slack import SlackBot

    bot = SlackBot("DummyToken", "General")
    assert bot.client.token == "DummyToken"
    assert bot.slack_channel == "General"


def test_slackbot_init_three_parameter():
    from rasa.core.channels.slack import SlackBot

    bot = SlackBot("DummyToken", "General", thread_id="DummyThread")
    assert bot.client.token == "DummyToken"
    assert bot.slack_channel == "General"
    assert bot.thread_id == "DummyThread"


# Use monkeypatch for sending attachments, images and plain text.
@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@pytest.mark.asyncio
async def test_slackbot_send_attachment_only():
    from rasa.core.channels.slack import SlackBot

    with aioresponses() as mocked:
        mocked.post(
            "https://www.slack.com/api/chat.postMessage",
            payload={"ok": True, "purpose": "Testing bots"},
        )

        bot = SlackBot("DummyToken", "General")
        attachment = SLACK_TEST_ATTACHMENT

        await bot.send_attachment("ID", attachment)

        r = latest_request(mocked, "POST", "https://www.slack.com/api/chat.postMessage")

        assert r

        request_params = json_of_latest_request(r)

        assert request_params == {
            "channel": "General",
            "as_user": True,
            "attachments": [attachment],
        }


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@pytest.mark.asyncio
async def test_slackbot_send_attachment_only_threaded():
    from rasa.core.channels.slack import SlackBot

    with aioresponses() as mocked:
        mocked.post(
            "https://www.slack.com/api/chat.postMessage",
            payload={"ok": True, "purpose": "Testing bots"},
        )

        bot = SlackBot("DummyToken", "General", thread_id="DummyThread")
        attachment = SLACK_TEST_ATTACHMENT

        await bot.send_attachment("ID", attachment)

        r = latest_request(mocked, "POST", "https://www.slack.com/api/chat.postMessage")

        assert r

        request_params = json_of_latest_request(r)

        assert request_params == {
            "channel": "General",
            "as_user": True,
            "attachments": [attachment],
            "thread_ts": "DummyThread",
        }


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@pytest.mark.asyncio
async def test_slackbot_send_attachment_with_text():
    from rasa.core.channels.slack import SlackBot

    with aioresponses() as mocked:
        mocked.post(
            "https://www.slack.com/api/chat.postMessage",
            payload={"ok": True, "purpose": "Testing bots"},
        )

        bot = SlackBot("DummyToken", "General")
        attachment = SLACK_TEST_ATTACHMENT
        attachment["text"] = "Here is the summary:"

        await bot.send_attachment("ID", attachment)

        r = latest_request(mocked, "POST", "https://www.slack.com/api/chat.postMessage")

        assert r

        request_params = json_of_latest_request(r)

        assert request_params == {
            "channel": "General",
            "as_user": True,
            "attachments": [attachment],
        }


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@pytest.mark.asyncio
async def test_slackbot_send_attachment_with_text_threaded():
    from rasa.core.channels.slack import SlackBot

    with aioresponses() as mocked:
        mocked.post(
            "https://www.slack.com/api/chat.postMessage",
            payload={"ok": True, "purpose": "Testing bots"},
        )

        bot = SlackBot("DummyToken", "General", thread_id="DummyThread")
        attachment = SLACK_TEST_ATTACHMENT
        attachment["text"] = "Here is the summary:"

        await bot.send_attachment("ID", attachment)

        r = latest_request(mocked, "POST", "https://www.slack.com/api/chat.postMessage")

        assert r

        request_params = json_of_latest_request(r)

        assert request_params == {
            "channel": "General",
            "as_user": True,
            "attachments": [attachment],
            "thread_ts": "DummyThread",
        }


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@pytest.mark.asyncio
async def test_slackbot_send_image_url():
    from rasa.core.channels.slack import SlackBot

    with aioresponses() as mocked:
        mocked.post(
            "https://www.slack.com/api/chat.postMessage",
            payload={"ok": True, "purpose": "Testing bots"},
        )

        bot = SlackBot("DummyToken", "General")
        url = "http://www.rasa.net"
        await bot.send_image_url("ID", url)

        r = latest_request(mocked, "POST", "https://www.slack.com/api/chat.postMessage")

        assert r

        request_params = json_of_latest_request(r)

        assert request_params["as_user"] is True
        assert request_params["channel"] == "General"
        assert len(request_params["blocks"]) == 1
        assert request_params["blocks"][0].get("type") == "image"
        assert request_params["blocks"][0].get("alt_text") == "http://www.rasa.net"
        assert request_params["blocks"][0].get("image_url") == "http://www.rasa.net"


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@pytest.mark.asyncio
async def test_slackbot_send_image_url_threaded():
    from rasa.core.channels.slack import SlackBot

    with aioresponses() as mocked:
        mocked.post(
            "https://www.slack.com/api/chat.postMessage",
            payload={"ok": True, "purpose": "Testing bots"},
        )

        bot = SlackBot("DummyToken", "General", thread_id="DummyThread")
        url = "http://www.rasa.net"
        await bot.send_image_url("ID", url)

        r = latest_request(mocked, "POST", "https://www.slack.com/api/chat.postMessage")

        assert r

        request_params = json_of_latest_request(r)

        assert request_params["as_user"] is True
        assert request_params["channel"] == "General"
        assert request_params["thread_ts"] == "DummyThread"
        assert len(request_params["blocks"]) == 1
        assert request_params["blocks"][0].get("type") == "image"
        assert request_params["blocks"][0].get("alt_text") == "http://www.rasa.net"
        assert request_params["blocks"][0].get("image_url") == "http://www.rasa.net"


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@pytest.mark.asyncio
async def test_slackbot_send_text():
    from rasa.core.channels.slack import SlackBot

    with aioresponses() as mocked:
        mocked.post(
            "https://www.slack.com/api/chat.postMessage",
            payload={"ok": True, "purpose": "Testing bots"},
        )

        bot = SlackBot("DummyToken", "General")
        await bot.send_text_message("ID", "my message")

        r = latest_request(mocked, "POST", "https://www.slack.com/api/chat.postMessage")

        assert r

        request_params = json_of_latest_request(r)

        assert request_params == {
            "as_user": True,
            "channel": "General",
            "text": "my message",
            "type": "mrkdwn",
        }


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@pytest.mark.asyncio
async def test_slackbot_send_text_threaded():
    from rasa.core.channels.slack import SlackBot

    with aioresponses() as mocked:
        mocked.post(
            "https://www.slack.com/api/chat.postMessage",
            payload={"ok": True, "purpose": "Testing bots"},
        )

        bot = SlackBot("DummyToken", "General", thread_id="DummyThread")
        await bot.send_text_message("ID", "my message")

        r = latest_request(mocked, "POST", "https://www.slack.com/api/chat.postMessage")

        assert r

        request_params = json_of_latest_request(r)

        assert request_params == {
            "as_user": True,
            "channel": "General",
            "text": "my message",
            "type": "mrkdwn",
            "thread_ts": "DummyThread",
        }


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@pytest.mark.asyncio
async def test_slackbot_send_text_with_buttons():
    from rasa.core.channels.slack import SlackBot

    with aioresponses() as mocked:
        mocked.post(
            "https://www.slack.com/api/chat.postMessage",
            payload={"ok": True, "purpose": "Testing bots"},
        )

        bot = SlackBot("DummyToken", "General")
        buttons = [{"title": "title", "payload": "payload"}]

        await bot.send_text_with_buttons("ID", "my message", buttons)

        r = latest_request(mocked, "POST", "https://www.slack.com/api/chat.postMessage")

        assert r

        request_params = json_of_latest_request(r)

        text_block = {
            "type": "section",
            "text": {"type": "plain_text", "text": "my message"},
        }
        button_block = {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "title"},
                    "value": "payload",
                }
            ],
        }
        assert request_params == {
            "as_user": True,
            "channel": "General",
            "text": "my message",
            "blocks": [text_block, button_block],
        }


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@pytest.mark.asyncio
async def test_slackbot_send_text_with_buttons_threaded():
    from rasa.core.channels.slack import SlackBot

    with aioresponses() as mocked:
        mocked.post(
            "https://www.slack.com/api/chat.postMessage",
            payload={"ok": True, "purpose": "Testing bots"},
        )

        bot = SlackBot("DummyToken", "General", thread_id="DummyThread")
        buttons = [{"title": "title", "payload": "payload"}]

        await bot.send_text_with_buttons("ID", "my message", buttons)

        r = latest_request(mocked, "POST", "https://www.slack.com/api/chat.postMessage")

        assert r

        request_params = json_of_latest_request(r)

        text_block = {
            "type": "section",
            "text": {"type": "plain_text", "text": "my message"},
        }
        button_block = {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "title"},
                    "value": "payload",
                }
            ],
        }
        assert request_params == {
            "as_user": True,
            "channel": "General",
            "text": "my message",
            "blocks": [text_block, button_block],
            "thread_ts": "DummyThread",
        }


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@pytest.mark.asyncio
async def test_slackbot_send_custom_json():
    from rasa.core.channels.slack import SlackBot

    with aioresponses() as mocked:
        mocked.post(
            "https://www.slack.com/api/chat.postMessage",
            payload={"ok": True, "purpose": "Testing bots"},
        )

        bot = SlackBot("DummyToken", "General")
        await bot.send_custom_json("ID", {"test_key": "test_value"})

        r = latest_request(mocked, "POST", "https://www.slack.com/api/chat.postMessage")

        assert r

        request_params = json_of_latest_request(r)

        assert request_params == {
            "as_user": True,
            "channel": "General",
            "test_key": "test_value",
        }


@pytest.mark.filterwarnings("ignore:unclosed.*:ResourceWarning")
@pytest.mark.asyncio
async def test_slackbot_send_custom_json_threaded():
    from rasa.core.channels.slack import SlackBot

    with aioresponses() as mocked:
        mocked.post(
            "https://www.slack.com/api/chat.postMessage",
            payload={"ok": True, "purpose": "Testing bots"},
        )

        bot = SlackBot("DummyToken", "General", thread_id="DummyThread")
        await bot.send_custom_json("ID", {"test_key": "test_value"})

        r = latest_request(mocked, "POST", "https://www.slack.com/api/chat.postMessage")

        assert r

        request_params = json_of_latest_request(r)

        assert request_params == {
            "as_user": True,
            "channel": "General",
            "thread_ts": "DummyThread",
            "test_key": "test_value",
        }


def prepare_slack_request(headers: Dict[Text, Any]) -> Request:
    request = Request(
        b"/webhooks/slack/webhook",
        headers=Header(headers),
        version="1.1",
        method="POST",
        transport=None,
        app=None,
    )
    request.body = b"""{"foo": "bar"}"""
    return request


def test_slack_fails_if_signature_is_missing():
    with pytest.raises(InvalidConfigException):
        SlackInput("mytoken")


@mock.patch("time.time", mock.MagicMock(return_value=1604586653))
def test_slack_verify_signature():
    request = prepare_slack_request(
        {
            "x-slack-signature": "v0=80a3bd62ce5af04d8d80781134f165df"
            "185b90342d467abf5c74a27d2d0dd1f5",
            "x-slack-request-timestamp": str(int(time.time())),
        }
    )
    input_with_right_secret = SlackInput("mytoken", slack_signing_secret="foobar")

    assert input_with_right_secret.is_request_from_slack_authentic(request) is True


def test_slack_fail_on_old_timestamp():
    request = prepare_slack_request(
        {
            "x-slack-signature": "v0=80a3bd62ce5af04d8d80781134f165df"
            "185b90342d467abf5c74a27d2d0dd1f5",
            "x-slack-request-timestamp": str(int(time.time()) - 10 * 60),
        }
    )
    input_with_right_secret = SlackInput("mytoken", slack_signing_secret="foobar")

    assert input_with_right_secret.is_request_from_slack_authentic(request) is False


def test_slack_handles_invalid_timestamp():
    request = prepare_slack_request(
        {
            "x-slack-signature": "v0=80a3bd62ce5af04d8d80781134f165df"
            "185b90342d467abf5c74a27d2d0dd1f5",
            "x-slack-request-timestamp": "foobar",
        }
    )
    input_with_right_secret = SlackInput("mytoken", slack_signing_secret="foobar")

    assert input_with_right_secret.is_request_from_slack_authentic(request) is False


def test_slack_verify_wrong_signature():
    request = prepare_slack_request(
        {
            "x-slack-signature": "v0=80a3bd62ce5af04d8d80781134f165df"
            "185b90342d467abf5c74a27d2d0dd1f5",
            "x-slack-request-timestamp": str(int(time.time())),
        }
    )
    input_with_wrong_secret = SlackInput("mytoken", slack_signing_secret="foobaz")

    assert input_with_wrong_secret.is_request_from_slack_authentic(request) is False


def test_slack_verify_signature_missing_headers():
    request = prepare_slack_request(
        {
            # let's check what happens if verification headers are missing
        }
    )
    slack = SlackInput("mytoken", slack_signing_secret="foobar")

    assert slack.is_request_from_slack_authentic(request) is False


async def fake_on_new_message(message: UserMessage):
    pass


@pytest.mark.asyncio
async def test_slack_process_message_retry():
    input_channel = SlackInput(
        slack_token="YOUR_SLACK_TOKEN",
        slack_channel="YOUR_SLACK_CHANNEL",
        slack_signing_secret="foobar",
    )

    request = Mock()
    request.headers = {
        input_channel.retry_num_header: 1,
        input_channel.retry_reason_header: input_channel.errors_ignore_retry[0],
    }

    response = await input_channel.process_message(
        request=request,
        on_new_message=fake_on_new_message,
        text="",
        sender_id=None,
        metadata=None,
    )

    assert response.status == HTTPStatus.CREATED
    assert response.headers == {"X-Slack-No-Retry": "1"}


async def fake_on_new_message_sleep(message: UserMessage):
    time.sleep(3)


@pytest.mark.asyncio
async def test_slack_process_message_timeout():
    input_channel = SlackInput(
        slack_token="YOUR_SLACK_TOKEN",
        slack_channel="YOUR_SLACK_CHANNEL",
        slack_signing_secret="foobar",
    )

    request = Mock()
    request.headers = {}

    start = time.time()
    response = await input_channel.process_message(
        request=request,
        on_new_message=fake_on_new_message_sleep,
        text="",
        sender_id=None,
        metadata=None,
    )
    end = time.time()

    duration = end - start

    assert duration < 3
    assert response.status == HTTPStatus.OK
