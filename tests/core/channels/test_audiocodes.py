import pytest
from rasa.core import utils
import datetime


def handle_activities_test_cb(expectedString):
    async def cb(resultUM):
        assert expectedString == resultUM.text

    return cb


def raise_if_called(p):
    def cb():
        assert False, p

    return cb


def test_audiocodes_token():
    from rasa.core.channels.audiocodes import AudiocodesInput

    with pytest.raises(Exception):
        AudiocodesInput.from_credentials(None)

    with pytest.raises(Exception):
        ac = AudiocodesInput.from_credentials({"token": "token"})
        ac._check_token("Bearer not token")

    ac = AudiocodesInput.from_credentials({"token": "token", "keep_alive": "200"})
    ac._check_token("Bearer token")
    assert ac.keep_alive == 200


def test_audiocodes_conversation_management():
    from rasa.core.channels.audiocodes import AudiocodesInput

    conversation_id = "some-generated-uuid"
    token = "token"
    input_channel = AudiocodesInput(token, None)
    c = AudiocodesInput.Conversation(conversation_id)
    input_channel.conversations[conversation_id] = c

    with pytest.raises(Exception):
        input_channel._get_conversation("other token", conversation_id)

    with pytest.raises(Exception):
        input_channel._get_conversation(token, "not exist cid")

    with pytest.raises(Exception):
        input_channel.handle_start_conversation({"conversation": conversation_id})

    assert c == input_channel._get_conversation(token, conversation_id)

    conversation_id_2 = "some-generated-uuid_2"
    c_2 = AudiocodesInput.Conversation(conversation_id_2)
    input_channel.conversations[conversation_id_2] = c_2

    assert len(input_channel.conversations) == 2
    c_2.lastActivity = datetime.datetime.utcnow() - datetime.timedelta(minutes=4)
    input_channel.clean_old_conversations()
    assert len(input_channel.conversations) == 1

    urls_dict = input_channel.handle_start_conversation(
        {"conversation": conversation_id_2}
    )
    assert len(input_channel.conversations) == 2
    assert "activitiesURL" in urls_dict


def test_audiocodes_channel():

    from rasa.core.channels.audiocodes import AudiocodesInput
    import rasa.core

    input_channel = AudiocodesInput("some_generated_token", None)

    s = rasa.core.run.configure_app([input_channel], port=5004)

    routes_list = utils.list_routes(s)
    assert routes_list.get("ac_webhook.health", "").startswith("/webhooks/audiocodes")
    assert routes_list.get("ac_webhook.receive", "").startswith(
        "/webhooks/audiocodes/webhook"
    )


@pytest.mark.asyncio
async def test_audiocodes_handle_events():

    from rasa.core.channels.audiocodes import AudiocodesInput, AudiocodesOutput

    ac_output = AudiocodesOutput()
    conversation_id = "some-generated-uuid"

    async def test_message(activity, expected_text, cb=handle_activities_test_cb):
        message = {"conversation": conversation_id, "activities": [activity]}
        return await c.handle_activities(message, ac_output, cb(expected_text))

    input_channel = AudiocodesInput("some_generated_token", None)
    c = AudiocodesInput.Conversation(conversation_id)
    input_channel.conversations[conversation_id] = c

    TEXT_MESSAGE = {
        "type": "message",
        "text": "This is a text message",
        "parameters": {"confidence": 0.8172},
        "id": "generated id 1",
    }

    await test_message(TEXT_MESSAGE, "This is a text message")

    TEXT_MESSAGE["id"] = "generated id 1.1"
    await test_message(TEXT_MESSAGE, "Test exception", cb=raise_if_called)
    assert len(ac_output.messages) == 1
    assert ac_output.messages[0]["name"] == "hangup"

    START_EVENT = {
        "type": "event",
        "name": "start",
        "parameters": {
            "caller": "some_caller",
            "callee": "123456789",
            "callerHost": "ip",
            "calleeHost": "other_ip",
        },
        "id": "generated id 2",
    }

    await test_message(
        START_EVENT,
        "/start{'caller': 'some_caller', 'callee': '123456789', 'callerHost': 'ip', 'calleeHost': 'other_ip'}",
    )

    DTMF_EVENT = {"type": "event", "name": "DTMF", "value": 3, "id": "generated id 3"}

    await test_message(DTMF_EVENT, "/DTMF{'value': 3}")

    UNKNOWN_MSG = {
        "type": "nonsense",
        "text": "nonsense nonsense",
        "id": "generated id 4",
    }

    await test_message(UNKNOWN_MSG, "Shouldn't be called", cb=raise_if_called)

    REPEAT_ID_MSG = {
        "type": "message",
        "text": "this is a text message",
        "id": "generated id 4",
    }
    await test_message(REPEAT_ID_MSG, "Shouldn't be called", cb=raise_if_called)


@pytest.mark.asyncio
async def test_audiocodes_output_channel_functions():

    from rasa.core.channels.audiocodes import AudiocodesOutput

    output_channel = AudiocodesOutput()
    conversation_id = "some-generated-uuid"

    await output_channel.send_text_message(
        recipient_id=conversation_id, text="This is a text message"
    )

    assert len(output_channel.messages) == 1
    assert output_channel.messages[0]["text"] == "This is a text message"

    await output_channel.send_custom_json(
        "conversation id",
        {
            "type": "event",
            "name": "transfer",
            "activityParams": {
                "handoverReason": "userRequest",
                "transferTarget": "tel:123456789",
            },
        },
    )

    assert len(output_channel.messages) == 2
    message = output_channel.messages[1]
    assert "id" in message
    assert "timestamp" in message
    assert message["activityParams"]["handoverReason"] == "userRequest"
