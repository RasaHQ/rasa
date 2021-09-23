import logging

import pytest
from http import HTTPStatus

from rasa import server
from rasa.core.agent import Agent
from rasa.core.channels import channel
from rasa.shared.exceptions import InvalidConfigException, RasaException
from rasa.core.channels.twilio_voice import TwilioVoiceInput
from rasa.core.channels.twilio_voice import TwilioVoiceCollectingOutputChannel
from typing import Text, Any, Dict, Type

logger = logging.getLogger(__name__)


async def test_twilio_voice_twiml_response_text():

    inputs = {
        "initial_prompt": "hello",
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    tv = TwilioVoiceInput(**inputs)

    output_channel = TwilioVoiceCollectingOutputChannel()

    await output_channel.send_text_message(recipient_id="Chuck Norris", text="Test:")
    assert len(output_channel.messages) == 1
    assert output_channel.messages[0]["text"] == "Test:"

    twiml = tv._build_twilio_voice_response(output_channel.messages)
    assert (
        str(twiml) == '<?xml version="1.0" encoding="UTF-8"?><Response>'
        '<Gather action="/webhooks/twilio_voice/webhook" '
        'actionOnEmptyResult="true" enhanced="false" input="speech" '
        'speechModel="default" speechTimeout="5"><Say voice="woman">'
        "Test:</Say></Gather></Response>"
    )


async def test_twilio_voice_twiml_response_buttons():

    inputs = {
        "initial_prompt": "hello",
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    tv = TwilioVoiceInput(**inputs)

    output_channel = TwilioVoiceCollectingOutputChannel()
    await output_channel.send_text_with_buttons(
        recipient_id="Chuck Norris",
        text="Buttons:",
        buttons=[
            {"title": "Yes", "payload": "/affirm"},
            {"title": "No", "payload": "/deny"},
        ],
    )
    assert len(output_channel.messages) == 3
    message_str = " ".join([m["text"] for m in output_channel.messages])
    assert message_str == "Buttons: Yes No"

    twiml = tv._build_twilio_voice_response(output_channel.messages)
    assert (
        str(twiml) == '<?xml version="1.0" encoding="UTF-8"?><Response>'
        '<Say voice="woman">Buttons:</Say><Pause length="1" />'
        '<Say voice="woman">Yes</Say><Pause length="1" />'
        '<Gather action="/webhooks/twilio_voice/webhook" '
        'actionOnEmptyResult="true" enhanced="false" input="speech" '
        'speechModel="default" speechTimeout="5">'
        '<Say voice="woman">No</Say></Gather></Response>'
    )


@pytest.mark.parametrize(
    "configs, expected",
    [
        (
            {
                "initial_prompt": "hello",
                "reprompt_fallback_phrase": "i didn't get that",
                "speech_model": "default",
                "speech_timeout": "5",
                "assistant_voice": "alien",
                "enhanced": "false",
            },
            InvalidConfigException,
        ),
        (
            {
                "initial_prompt": "hello",
                "reprompt_fallback_phrase": "i didn't get that",
                "speech_model": "default",
                "speech_timeout": "not a number",
                "assistant_voice": "woman",
                "enhanced": "false",
            },
            InvalidConfigException,
        ),
        (
            {
                "initial_prompt": "hello",
                "reprompt_fallback_phrase": "i didn't get that",
                "speech_model": "default",
                "speech_timeout": "auto",
                "assistant_voice": "woman",
                "enhanced": "wrong",
            },
            InvalidConfigException,
        ),
        (
            {
                "initial_prompt": "hello",
                "reprompt_fallback_phrase": "i didn't get that",
                "speech_model": "default",
                "speech_timeout": "5",
                "assistant_voice": "woman",
                "enhanced": "true",
            },
            InvalidConfigException,
        ),
        (
            {
                "initial_prompt": "hello",
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "enhanced": "true",
                "speech_model": "default",
                "speech_timeout": "auto",
            },
            InvalidConfigException,
        ),
        (
            {
                "initial_prompt": "hello",
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "enhanced": "true",
                "speech_model": "phone_call",
                "speech_timeout": "auto",
            },
            InvalidConfigException,
        ),
    ],
)
def test_invalid_configs(configs: Dict[Text, Any], expected: Type[RasaException]):
    with pytest.raises(expected):
        TwilioVoiceInput(**configs)


async def test_twilio_voice_remove_image():

    with pytest.warns(UserWarning):
        output_channel = TwilioVoiceCollectingOutputChannel()
        await output_channel.send_response(
            recipient_id="Chuck Norris",
            message={"image": "https://i.imgur.com/nGF1K8f.jpg", "text": "Some text."},
        )


async def test_twilio_voice_keep_image_text():

    output_channel = TwilioVoiceCollectingOutputChannel()
    await output_channel.send_response(
        recipient_id="Chuck Norris",
        message={"image": "https://i.imgur.com/nGF1K8f.jpg", "text": "Some text."},
    )
    assert len(output_channel.messages) == 1
    assert output_channel.messages[0]["text"] == "Some text."


async def test_twilio_emoji_warning():

    with pytest.warns(UserWarning):
        output_channel = TwilioVoiceCollectingOutputChannel()
        await output_channel.send_response(
            recipient_id="User", message={"text": "Howdy 😀"}
        )


async def test_twilio_voice_multiple_responses():

    inputs = {
        "initial_prompt": "hello",
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    tv = TwilioVoiceInput(**inputs)

    output_channel = TwilioVoiceCollectingOutputChannel()

    await output_channel.send_text_message(
        recipient_id="Chuck Norris", text="message 1"
    )
    await output_channel.send_text_message(
        recipient_id="Chuck Norris", text="message 2"
    )
    assert len(output_channel.messages) == 2
    assert output_channel.messages[0]["text"] == "message 1"
    assert output_channel.messages[1]["text"] == "message 2"

    twiml = tv._build_twilio_voice_response(output_channel.messages)

    assert (
        str(twiml) == '<?xml version="1.0" encoding="UTF-8"?><Response>'
        '<Say voice="woman">message 1</Say><Pause length="1" />'
        '<Gather action="/webhooks/twilio_voice/webhook" actionOnEmptyResult="true" '
        'enhanced="false" input="speech" speechModel="default" speechTimeout="5">'
        '<Say voice="woman">message 2</Say></Gather></Response>'
    )


async def test_twilio_receive_answer(stack_agent: Agent):
    app = server.create_app(agent=stack_agent)

    inputs = {
        "initial_prompt": "hello",
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    tv = TwilioVoiceInput(**inputs)
    channel.register([tv], app, "/webhooks/")

    client = app.asgi_client

    body = {"From": "Tobias", "CallStatus": "ringing"}
    _, response = await client.post(
        "/webhooks/twilio_voice/webhook",
        headers={"Content-type": "application/x-www-form-urlencoded"},
        data=body,
    )
    assert response.status == HTTPStatus.OK
    # Actual test xml content
    assert (
        response.body == b'<?xml version="1.0" encoding="UTF-8"?><Response>'
        b'<Gather action="/webhooks/twilio_voice/webhook" actionOnEmptyResult="true" '
        b'enhanced="false" input="speech" speechModel="default" speechTimeout="5">'
        b'<Say voice="woman">hey there None!</Say></Gather></Response>'
    )


async def test_twilio_receive_no_response(stack_agent: Agent):
    app = server.create_app(agent=stack_agent)

    inputs = {
        "initial_prompt": "hello",
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    tv = TwilioVoiceInput(**inputs)
    channel.register([tv], app, "/webhooks/")

    client = app.asgi_client

    body = {"From": "Matthew", "CallStatus": "ringing"}
    _, response = await client.post(
        "/webhooks/twilio_voice/webhook",
        headers={"Content-type": "application/x-www-form-urlencoded"},
        data=body,
    )
    assert response.status == HTTPStatus.OK
    assert response.body

    body = {"From": "Matthew", "CallStatus": "answered"}
    _, response = await client.post(
        "/webhooks/twilio_voice/webhook",
        headers={"Content-type": "application/x-www-form-urlencoded"},
        data=body,
    )

    assert response.status == HTTPStatus.OK
    assert (
        response.body == b'<?xml version="1.0" encoding="UTF-8"?><Response>'
        b'<Gather action="/webhooks/twilio_voice/webhook" actionOnEmptyResult="true" '
        b'enhanced="false" input="speech" speechModel="default" speechTimeout="5">'
        b'<Say voice="woman">hey there None!</Say></Gather></Response>'
    )


async def test_twilio_receive_no_previous_response(stack_agent: Agent):
    app = server.create_app(agent=stack_agent)

    inputs = {
        "initial_prompt": "hello",
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    tv = TwilioVoiceInput(**inputs)
    channel.register([tv], app, "/webhooks/")

    client = app.asgi_client

    body = {"From": "Ray", "CallStatus": "answered"}
    _, response = await client.post(
        "/webhooks/twilio_voice/webhook",
        headers={"Content-type": "application/x-www-form-urlencoded"},
        data=body,
    )

    assert response.status == HTTPStatus.OK
    assert (
        response.body == b'<?xml version="1.0" encoding="UTF-8"?><Response>'
        b'<Gather action="/webhooks/twilio_voice/webhook" actionOnEmptyResult="true" '
        b'enhanced="false" input="speech" speechModel="default" speechTimeout="5">'
        b'<Say voice="woman">i didn\'t get that</Say></Gather></Response>'
    )
