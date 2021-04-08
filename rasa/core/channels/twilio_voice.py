from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from typing import Text, Callable, Awaitable, List, Any, Dict, Optional, NoReturn

from rasa.shared.utils.io import raise_warning
from rasa.shared.core.events import BotUttered
from rasa.shared.exceptions import RasaException
from rasa.core.channels.channel import (
    InputChannel,
    CollectingOutputChannel,
    UserMessage,
)


class TwilioVoiceInput(InputChannel):
    """Input channel for Twilio Voice."""

    SUPPORTED_VOICES = [
        "man",
        "woman",
        "alice",
        "Polly.Mads",
        "Polly.Naja",
        "Polly.Lotte",
        "Polly.Reuben",
        "Polly.Nicole",
        "Polly.Russell",
        "Polly.Amy",
        "Polly.Brian",
        "Polly.Emma",
        "Polly.Amy-Neural",
        "Polly.Emma-Neural",
        "Polly.Brian-Neural",
        "Polly.Raveena",
        "Polly.Ivy",
        "Polly.Joanna",
        "Polly.Joey",
        "Polly.Justin",
        "Polly.Kendra",
        "Polly.Kimberly",
        "Polly.Matthew",
        "Polly.Salli",
        "Polly.Ivy-Neural",
        "Polly.Joanna-Neural",
        "Polly.Kendra-Neural",
        "Polly.Kimberly-Neural",
        "Polly.Sally-Neural",
        "Polly.Joey-Neural",
        "Polly.Justin-Neural",
        "Polly.Matthew-Neural",
        "Polly.Geraint",
        "Polly.Celine",
        "Polly.Mathieu",
        "Polly.Chantal",
        "Polly.Hans",
        "Polly.Marlene",
        "Polly.Vicki",
        "Polly.Dora",
        "Polly.Karl",
        "Polly.Carla",
        "Polly.Giorgio",
        "Polly.Mizuki",
        "Polly.Takumi",
        "Polly.Liv",
        "Polly.Jacek",
        "Polly.Jan",
        "Polly.Ewa",
        "Polly.Maja",
        "Polly.Ricardo",
        "Polly.Vitoria",
        "Polly.Camila-Neural",
        "Polly.Cristiano",
        "Polly.Ines",
        "Polly.Carmen",
        "Polly.Maxim",
        "Polly.Tatyana",
        "Polly.Conchita",
        "Polly.Enrique",
        "Polly.Miguel",
        "Polly.Penelope",
        "Polly.Lupe-Neural",
        "Polly.Astrid",
        "Polly.Filiz",
        "Polly.Gwyneth",
    ]

    @classmethod
    def name(cls) -> Text:
        """Name of your custom channel."""
        return "twilio_voice"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        """Load custom configurations."""
        if not credentials:
            return cls(initial_prompt="hello", assistant_voice="woman")

        return cls(
            credentials.get("initial_prompt", "hello"),
            credentials.get("assistant_voice", "woman"),
        )

    def __init__(
        self, initial_prompt: Optional[Text], assistant_voice: Optional[Text],
    ) -> None:
        """Creates a connection to Twilio voice.

        Args:
            assistant_voice: name of the assistant voice to use.
        """
        self.initial_prompt = initial_prompt
        self.assistant_voice = assistant_voice

        if assistant_voice not in self.SUPPORTED_VOICES:
            self.raise_invalid_voice_exception()

    def raise_invalid_voice_exception(self) -> None:
        """Raises an error if an invalid voice is provided."""
        raise RasaException(
            f"{self.assistant_voice} is an invalid as an assistant voice. Please refer to the documentation for a list "
            f"of valid voices you can use for your voice assistant."
        )

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Blueprint:
        """Defines endpoints for Twilio voice channel."""
        twilio_voice_webhook = Blueprint("Twilio_voice_webhook", __name__)

        @twilio_voice_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @twilio_voice_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request):
            sender_id = request.form.get("From")
            text = request.form.get("SpeechResult")
            input_channel = self.name()
            call_status = request.form.get("CallStatus")

            collector = TwilioVoiceCollectingOutputChannel()

            # Provide an initial greeting to answer the user's call.
            if (text is None) and (call_status == "ringing"):
                text = self.initial_prompt

            # determine the response.
            if text is not None:
                await on_new_message(
                    UserMessage(
                        text, collector, sender_id, input_channel=input_channel,
                    )
                )

                twilio_response = self.build_twilio_voice_response(collector.messages)
            # If the user doesn't respond resend the last message.
            else:
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

                twilio_response = self.build_twilio_voice_response(
                    [{"text": last_response}]
                )
            return response.text(str(twilio_response), content_type="text/xml")

        return twilio_voice_webhook

    def build_twilio_voice_response(
        self, messages: List[Dict[Text, Any]]
    ) -> VoiceResponse:
        """Builds the Twilio Voice Response object."""
        voice_response = VoiceResponse()
        gather = Gather(
            input="speech",
            action=f"/webhooks/{self.name()}/webhook",
            actionOnEmptyResult=True,
            speechTimeout="auto",
        )

        # Add pauses between messages.
        # Add a listener to the last message to listen for user response.
        for i, message in enumerate(messages):
            if i + 1 == len(messages):
                gather.say(message["text"], voice=self.assistant_voice)
                voice_response.append(gather)
            else:
                voice_response.say(message["text"], voice=self.assistant_voice)
                voice_response.pause(length=1)

        return voice_response


class TwilioVoiceCollectingOutputChannel(CollectingOutputChannel):
    """Output channel that collects send messages in a list.

    (doesn't send them anywhere, just collects them).
    """

    @classmethod
    def name(cls) -> Text:
        """Name of the output channel."""
        return "twilio_voice"

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """Convert buttons into a voice representation."""
        await self._persist_message(self._message(recipient_id, text=text))

        for b in buttons:
            await self._persist_message(self._message(recipient_id, text=b["title"]))

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """For voice channel do not send images."""
        raise_warning(
            "Image removed from voice message. Only text of message is sent."
        )
