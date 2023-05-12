import json
import logging

import pytest

from rasa.core import utils

logger = logging.getLogger(__name__)


def test_hangouts_channel():

    from rasa.core.channels.hangouts import HangoutsInput
    import rasa.core

    input_channel = HangoutsInput(
        project_id="12345678901",
        # intent name for bot added to direct message event
        hangouts_user_added_intent_name="/added_dm",
        # intent name for bot added to room event
        hangouts_room_added_intent_name="/added_room",
        # intent name for bot removed from space event
        hangouts_removed_intent_name="/removed",
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)

    routes_list = utils.list_routes(s)

    assert routes_list.get("hangouts_webhook.health").startswith("/webhooks/hangouts")
    assert routes_list.get("hangouts_webhook.receive").startswith(
        "/webhooks/hangouts/webhook"
    )


def test_hangouts_extract_functions():
    # from https://developers.google.com/hangouts/chat/reference/message-formats/events#added_to_space # noqa: E501
    ADDED_EVENT = {
        "type": "ADDED_TO_SPACE",
        "eventTime": "2017-03-02T19:02:59.910959Z",
        "space": {
            "name": "spaces/AAAAAAAAAAA",
            "displayName": "Chuck Norris Discussion Room",
            "type": "ROOM",
        },
        "user": {
            "name": "users/12345678901234567890",
            "displayName": "Chuck Norris",
            "avatarUrl": "https://lh3.googleusercontent.com/.../photo.jpg",
            "email": "chuck@example.com",
        },
    }

    # from https://developers.google.com/hangouts/chat/reference/message-formats/events#removed_from_space # noqa: E501
    REMOVED_EVENT = {
        "type": "REMOVED_FROM_SPACE",
        "eventTime": "2017-03-02T19:02:59.910959Z",
        "space": {"name": "spaces/AAAAAAAAAAA", "type": "DM"},
        "user": {
            "name": "users/12345678901234567890",
            "displayName": "Chuck Norris",
            "avatarUrl": "https://lh3.googleusercontent.com/.../photo.jpg",
            "email": "chuck@example.com",
        },
    }

    # from https://developers.google.com/hangouts/chat/reference/message-formats/events#message # noqa: E501
    MESSAGE = {
        "type": "MESSAGE",
        "eventTime": "2017-03-02T19:02:59.910959Z",
        "space": {
            "name": "spaces/AAAAAAAAAAA",
            "displayName": "Chuck Norris Discussion Room",
            "type": "ROOM",
        },
        "message": {
            "name": "spaces/AAAAAAAAAAA/messages/CCCCCCCCCCC",
            "sender": {
                "name": "users/12345678901234567890",
                "displayName": "Chuck Norris",
                "avatarUrl": "https://lh3.googleusercontent.com/.../photo.jpg",
                "email": "chuck@example.com",
            },
            "createTime": "2017-03-02T19:02:59.910959Z",
            "text": "@TestBot Violence is my last option.",
            "argumentText": " Violence is my last option.",
            "thread": {"name": "spaces/AAAAAAAAAAA/threads/BBBBBBBBBBB"},
            "annotations": [
                {
                    "length": 8,
                    "startIndex": 0,
                    "userMention": {
                        "type": "MENTION",
                        "user": {
                            "avatarUrl": "https://.../avatar.png",
                            "displayName": "TestBot",
                            "name": "users/1234567890987654321",
                            "type": "BOT",
                        },
                    },
                    "type": "USER_MENTION",
                }
            ],
        },
        "user": {
            "name": "users/12345678901234567890",
            "displayName": "Chuck Norris",
            "avatarUrl": "https://lh3.googleusercontent.com/.../photo.jpg",
            "email": "chuck@example.com",
        },
    }

    from rasa.core.channels.hangouts import HangoutsInput
    import rasa.core

    input_channel = HangoutsInput(
        project_id="12345678901",
        # intent name for bot added to direct message event
        hangouts_user_added_intent_name="/added_dm",
        # intent name for bot added to room event
        hangouts_room_added_intent_name="/added_room",
        # intent name for bot removed from space event
        hangouts_removed_intent_name="/removed",
    )

    app = rasa.core.run.configure_app([input_channel], port=5004)

    # This causes irritating error even though test passes...
    # req, _ = app.test_client.post("/webhooks/hangouts/webhook",
    #                                   data=json.dumps(MESSAGE))
    # ..therefore create Request object directly
    from sanic.request import Request

    def create_req(app):
        return Request(
            b"http://127.0.0.1:42101/webhooks/hangouts/webhook",
            [],
            None,
            "POST",
            None,
            app=app,
        )

    req = create_req(app)
    req.body = bytes(json.dumps(MESSAGE), encoding="utf-8")
    assert input_channel._extract_sender(req) == "Chuck Norris"
    assert input_channel._extract_room(req) == "Chuck Norris Discussion Room"
    assert input_channel._extract_message(req) == "@TestBot Violence is my last option."

    req = create_req(app)
    req.body = bytes(json.dumps(ADDED_EVENT), encoding="utf-8")
    assert input_channel._extract_sender(req) == "Chuck Norris"
    assert input_channel._extract_room(req) == "Chuck Norris Discussion Room"
    assert input_channel._extract_message(req) == "/added_room"

    req = create_req(app)
    req.body = bytes(json.dumps(REMOVED_EVENT), encoding="utf-8")
    assert input_channel._extract_sender(req) == "Chuck Norris"
    assert input_channel._extract_room(req) is None
    assert input_channel._extract_message(req) == "/removed"


@pytest.mark.asyncio
async def test_hangouts_output_channel_functions():

    from rasa.core.channels.hangouts import HangoutsOutput

    output_channel = HangoutsOutput()

    # with every call to _persist_message, the messages attribute (dict) is altered,
    # as Hangouts always expects a single dict as response

    await output_channel.send_text_message(recipient_id="Chuck Norris", text="Test:")

    assert len(output_channel.messages) == 1
    assert output_channel.messages["text"] == "Test:"

    await output_channel.send_attachment(
        recipient_id="Chuck Norris", attachment="Attachment"
    )

    assert len(output_channel.messages) == 1
    # two text messages are appended with space inbetween
    assert output_channel.messages["text"] == "Test: Attachment"

    await output_channel.send_quick_replies(
        recipient_id="Chuck Norris",
        text="Test passing?",
        quick_replies=[
            {"title": "Yes", "payload": "/confirm"},
            {"title": "No", "payload": "/deny"},
        ],
    )
    assert len(output_channel.messages) == 1
    # for text and cards, text is turned to card format and two cards are returned
    assert (
        output_channel.messages["cards"][1]["sections"][0]["widgets"][0][
            "textParagraph"
        ]["text"]
        == "Test passing?"
    )
    assert (
        output_channel.messages["cards"][1]["sections"][0]["widgets"][1]["buttons"][0][
            "textButton"
        ]["onClick"]["action"]["actionMethodName"]
        == "/confirm"
    )

    await output_channel.send_image_url(recipient_id="Chuck Norris", image="test.png")
    assert len(output_channel.messages) == 1
    assert (
        output_channel.messages["cards"][2]["sections"][0]["widgets"][0]["image"][
            "imageUrl"
        ]
        == "test.png"
    )
