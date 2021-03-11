import inspect
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from typing import Text, Callable, Awaitable, List

from rasa.core.channels.channel import (
    InputChannel,
    CollectingOutputChannel,
    UserMessage,
)


class TwilioVoiceInput(InputChannel):
    @classmethod
    def name(cls) -> Text:
        """Name of your custom channel."""
        return "twilio_voice"

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Blueprint:

        twilio_voice_webhook = Blueprint(
            "custom_webhook_{}".format(type(self).__name__),
            inspect.getmodule(self).__name__,
        )

        @twilio_voice_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @twilio_voice_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request):
            sender_id = request.form.get("From")  # method to get sender_id
            text = request.form.get("SpeechResult")  # method to fetch text
            input_channel = self.name()  # method to fetch input channel
            call_status = request.form.get("CallStatus")  # method to fetch call status
            call_sid = request.form.get("CallSid")  # Get the call identifier

            collector = TwilioVoiceCollectingOutputChannel()

            # Provide an initial greeting to answer the user's call.
            if (text is None) & (call_status == "ringing"):
                text = "hello"
            # If the user doesn't respond to the previous message resend the last message.
            elif text is None:
                # Get last user utterance from tracker.
                tracker = request.app.agent.tracker_store.retrieve(sender_id)
                text = tracker.current_state()["latest_message"]["text"]

            # determine the response.
            if text is not None:
                await on_new_message(
                    UserMessage(
                        text,
                        collector,
                        sender_id,
                        input_channel=input_channel,
                    )
                )

                # Parse the text responses and build the Twilio VoiceResponse.
                respond_segments = []
                for message in collector.messages:
                    respond_segments.append(message["text"])
                    if "buttons" in message:
                        for button in message["buttons"]:
                            respond_segments.append(button["title"])
                twilio_response = build_twilio_voice_response(respond_segments)
                return response.text(str(twilio_response), content_type="text/xml")

        return twilio_voice_webhook


class TwilioVoiceCollectingOutputChannel(CollectingOutputChannel):
    """Output channel that collects send messages in a list
    (doesn't send them anywhere, just collects them)."""

    @classmethod
    def name(cls) -> Text:
        return "twilio_voice"


def build_twilio_voice_response(messages: List[Text]) -> VoiceResponse:
    """Builds the Twilio Voice Response object."""
    vr = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/webhooks/twilio_voice/webhook",
        actionOnEmptyResult=True,
        speechTimeout="auto",
    )

    # Add pauses between messages.
    # Add a listener to the last message to listen for user response.
    for i, message in enumerate(messages):
        if i + 1 == len(messages):
            gather.say(message)
            vr.append(gather)
        else:
            vr.say(message)
            vr.pause(length=1)

    return vr
