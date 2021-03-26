import json
import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_twilio_voice_twiml_response_text():

    from rasa.core.channels.twilio_voice import TwilioVoiceInput
    from rasa.core.channels.twilio_voice import TwilioVoiceCollectingOutputChannel

    tv = TwilioVoiceInput("woman")

    output_channel = TwilioVoiceCollectingOutputChannel()

    await output_channel.send_text_message(recipient_id="Chuck Norris", text="Test:")
    assert len(output_channel.messages) == 1
    assert output_channel.messages[0]["text"] == "Test:"

    twiml = tv.build_twilio_voice_response(output_channel.messages)
    assert (
        str(twiml)
        == '<?xml version="1.0" encoding="UTF-8"?><Response><Gather action="/webhooks/twilio_voice/webhook" actionOnEmptyResult="true" input="speech" speechTimeout="auto"><Say language="en" voice="woman">Test:</Say></Gather></Response>'
    )


@pytest.mark.asyncio
async def test_twilio_voice_twiml_response_buttons():

    from rasa.core.channels.twilio_voice import TwilioVoiceInput
    from rasa.core.channels.twilio_voice import TwilioVoiceCollectingOutputChannel

    tv = TwilioVoiceInput("woman")

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
        == '<?xml version="1.0" encoding="UTF-8"?><Response><Say language="en" voice="woman">Buttons:</Say><Pause length="1" /><Say language="en" voice="woman">Yes</Say><Pause length="1" /><Gather action="/webhooks/twilio_voice/webhook" actionOnEmptyResult="true" input="speech" speechTimeout="auto"><Say language="en" voice="woman">No</Say></Gather></Response>'
    )
