import json
import logging

import pytest

from rasa.shared.exceptions import RasaException
from rasa.core.channels.twilio_voice import TwilioVoiceInput
from rasa.core.channels.twilio_voice import TwilioVoiceCollectingOutputChannel

logger = logging.getLogger(__name__)


async def test_twilio_voice_twiml_response_text():

    tv = TwilioVoiceInput(initial_prompt="hello", assistant_voice="woman")

    output_channel = TwilioVoiceCollectingOutputChannel()

    await output_channel.send_text_message(recipient_id="Chuck Norris", text="Test:")
    assert len(output_channel.messages) == 1
    assert output_channel.messages[0]["text"] == "Test:"

    twiml = tv.build_twilio_voice_response(output_channel.messages)
    assert (
        str(twiml)
        == '<?xml version="1.0" encoding="UTF-8"?><Response><Gather action="/webhooks/twilio_voice/webhook" actionOnEmptyResult="true" input="speech" speechTimeout="auto"><Say voice="woman">Test:</Say></Gather></Response>'
    )


async def test_twilio_voice_twiml_response_buttons():

    tv = TwilioVoiceInput(initial_prompt="hello", assistant_voice="woman")

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

    twiml = tv.build_twilio_voice_response(output_channel.messages)
    assert (
        str(twiml)
        == '<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="woman">Buttons:</Say><Pause length="1" /><Say voice="woman">Yes</Say><Pause length="1" /><Gather action="/webhooks/twilio_voice/webhook" actionOnEmptyResult="true" input="speech" speechTimeout="auto"><Say voice="woman">No</Say></Gather></Response>'
    )


async def test_twilio_invalid_assistant_voice():

    with pytest.raises(RasaException):
        tv = TwilioVoiceInput(initial_prompt="hello", assistant_voice="alien")
