import inspect
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from typing import Text, Callable, Awaitable, List, Any, Dict

from rasa.shared.core.events import BotUttered
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

            # determine the response.
            if text is not None:
                await on_new_message(
                    UserMessage(
                        text, collector, sender_id, input_channel=input_channel,
                    )
                )

                # Build the Twilio VoiceResponse.
                twilio_response = build_twilio_voice_response(collector.messages)
                return response.text(str(twilio_response), content_type="text/xml")
            # If the user doesn't respond to the previous message resend the last message.
            elif text is None:
                # Get last user utterance from tracker.
                tracker = request.app.agent.tracker_store.retrieve(sender_id)
                last_response = next(
                    (e for e in reversed(tracker.events) if isinstance(e, BotUttered)),
                    None,
                )

                # If no previous utterance found say something generic.
                if last_response is None:
                    last_response = "I didn't get that."
                else:
                    last_response = last_response.text

                twilio_response = build_twilio_voice_response([{"text": last_response}])
                return response.text(str(twilio_response), content_type="text/xml")

        return twilio_voice_webhook


class TwilioVoiceCollectingOutputChannel(CollectingOutputChannel):
    """Output channel that collects send messages in a list
    (doesn't send them anywhere, just collects them)."""

    @classmethod
    def name(cls) -> Text:
        return "twilio_voice"

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        await self._persist_message(self._message(recipient_id, text=text))

        for b in buttons:
            await self._persist_message(
                self._message(recipient_id, text=b["title"])
            )

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """For voice channel do not send images."""

        pass


def build_twilio_voice_response(messages: List[Dict[Text, Any]]) -> VoiceResponse:
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
            gather.say(message["text"])
            vr.append(gather)
        else:
            vr.say(message["text"])
            vr.pause(length=1)

    return vr
