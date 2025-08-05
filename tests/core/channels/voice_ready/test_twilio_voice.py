import logging
from typing import Any, Dict, Text, Type

import pytest

from rasa.core.channels.voice_ready.twilio_voice import (
    TwilioVoiceCollectingOutputChannel,
    TwilioVoiceInput,
)
from rasa.shared.exceptions import InvalidConfigException, RasaException

logger = logging.getLogger(__name__)


async def test_twilio_voice_twiml_response_text():
    inputs = {
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    twilio_voice_input = TwilioVoiceInput(**inputs)

    output_channel = TwilioVoiceCollectingOutputChannel()

    await output_channel.send_text_message(recipient_id="Chuck Norris", text="Test:")
    assert len(output_channel.messages) == 1
    assert output_channel.messages[0]["text"] == "Test:"

    twiml = twilio_voice_input._build_twilio_voice_response(output_channel.messages)
    assert (
        str(twiml) == '<?xml version="1.0" encoding="UTF-8"?><Response>'
        '<Gather action="/webhooks/twilio_voice/webhook" '
        'actionOnEmptyResult="true" enhanced="false" input="speech" '
        'speechModel="default" speechTimeout="5"><Say voice="woman">'
        "Test:</Say></Gather></Response>"
    )


async def test_twilio_voice_twiml_response_buttons():
    inputs = {
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    twilio_voice_input = TwilioVoiceInput(**inputs)

    output_channel = TwilioVoiceCollectingOutputChannel()
    await output_channel.send_text_with_buttons(
        recipient_id="Chuck Norris",
        text="Buttons:",
        buttons=[
            {"title": "Yes", "payload": "/affirm"},
            {"title": "No", "payload": "/deny"},
        ],
    )
    assert len(output_channel.messages) == 1
    message_str = " ".join([m["text"] for m in output_channel.messages])
    assert message_str == "Buttons: Yes, No"

    twiml = twilio_voice_input._build_twilio_voice_response(output_channel.messages)
    assert (
        str(twiml) == '<?xml version="1.0" encoding="UTF-8"?><Response>'
        '<Gather action="/webhooks/twilio_voice/webhook" '
        'actionOnEmptyResult="true" enhanced="false" input="speech" '
        'speechModel="default" speechTimeout="5">'
        '<Say voice="woman">Buttons: Yes, No</Say>'
        "</Gather></Response>"
    )


@pytest.mark.parametrize(
    "configs, expected",
    [
        (
            {
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
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "enhanced": "true",
                "speech_model": "phone_call",
                "speech_timeout": "auto",
            },
            InvalidConfigException,
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "5",
                "speech_model": "default",
                "enhanced": "false",
                "password": "test_password",
            },
            InvalidConfigException,
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "5",
                "speech_model": "default",
                "enhanced": "false",
                "username": "test_user",
            },
            InvalidConfigException,
        ),
    ],
)
def test_invalid_configs(configs: Dict[Text, Any], expected: Type[RasaException]):
    with pytest.raises(expected):
        TwilioVoiceInput(**configs)


@pytest.mark.parametrize(
    "configs, expected",
    [
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
            },
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "5",
                "speech_model": "default",
                "enhanced": "false",
                "username": None,
                "password": None,
            },
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "3",
                "speech_model": "phone_call",
                "enhanced": "true",
            },
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "3",
                "speech_model": "phone_call",
                "enhanced": "true",
                "username": None,
                "password": None,
            },
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "3",
                "speech_model": "phone_call",
                "enhanced": "true",
                "username": "test_user",
                "password": "test_password",
            },
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "3",
                "speech_model": "phone_call",
                "enhanced": "true",
                "username": "test_user",
                "password": "test_password",
            },
        ),
    ],
)
def test_twilio_voice_input(configs: Dict[Text, Any], expected: Dict[Text, Any]):
    """Test TwilioVoiceInput initialization with various configurations."""
    twilio_voice_input = TwilioVoiceInput.from_credentials(configs)

    assert isinstance(twilio_voice_input, TwilioVoiceInput)

    assert (
        twilio_voice_input.reprompt_fallback_phrase
        == expected["reprompt_fallback_phrase"]
    )
    assert twilio_voice_input.speech_model == expected["speech_model"]
    assert twilio_voice_input.speech_timeout == expected["speech_timeout"]
    assert twilio_voice_input.assistant_voice == expected["assistant_voice"]
    assert twilio_voice_input.enhanced == expected["enhanced"]
    assert twilio_voice_input.username == expected["username"]
    assert twilio_voice_input.password == expected["password"]


@pytest.mark.parametrize(
    "config",
    [
        {
            "reprompt_fallback_phrase": "i didn't get that",
            "assistant_voice": "woman",
            "speech_timeout": "3",
            "speech_model": "phone_call",
            "enhanced": "true",
            "username": "test_user",
        },
        {
            "reprompt_fallback_phrase": "i didn't get that",
            "assistant_voice": "woman",
            "speech_timeout": "3",
            "speech_model": "phone_call",
            "enhanced": "true",
            "password": "test_password",
        },
    ],
)
def test_twilio_voice_input_invalid_credentials(
    config: Dict[str, str],
):
    with pytest.raises(RasaException):
        TwilioVoiceInput.from_credentials(config)


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


@pytest.fixture
def twilio_voice_input() -> TwilioVoiceInput:
    inputs = {
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    return TwilioVoiceInput(**inputs)


async def test_twilio_voice_multiple_responses(
    twilio_voice_input: TwilioVoiceInput,
):
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

    twiml = twilio_voice_input._build_twilio_voice_response(output_channel.messages)

    assert (
        str(twiml) == '<?xml version="1.0" encoding="UTF-8"?><Response>'
        '<Say voice="woman">message 1</Say>'
        '<Pause length="1" />'
        '<Gather action="/webhooks/twilio_voice/webhook" '
        'actionOnEmptyResult="true" '
        'enhanced="false" '
        'input="speech" '
        'speechModel="default" '
        'speechTimeout="5">'
        '<Say voice="woman">message 2</Say>'
        "</Gather></Response>"
    )
