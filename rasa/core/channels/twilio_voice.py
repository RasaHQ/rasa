import re
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from typing import Text, Callable, Awaitable, List, Any, Dict, Optional

from rasa.shared.utils.io import raise_warning
from rasa.shared.core.events import BotUttered
from rasa.shared.exceptions import InvalidConfigException
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

    SUPPORTED_SPEECH_MODELS = [
        "default",
        "numbers_and_commands",
        "phone_call"
    ]

    @classmethod
    def name(cls) -> Text:
        """Name of your custom channel."""
        return "twilio_voice"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        """Load custom configurations."""
        credentials = credentials or {}

        return cls(
            credentials.get("initial_prompt", "hello"),
            credentials.get("reprompt_fallback_phrase", "I'm sorry I didn't get that could you rephrase."),
            credentials.get("assistant_voice", "woman"),
            credentials.get("speech_timeout", "5"),
            credentials.get("speech_model", "default"),
            credentials.get("enhanced", "false")
        )

    def __init__(
        self,
        initial_prompt: Optional[Text],
        reprompt_fallback_phrase: Optional[Text],
        assistant_voice: Optional[Text],
        speech_timeout: Optional[Text],
        speech_model: Optional[Text],
        enhanced: Optional[Text]
    ) -> None:
        """Creates a connection to Twilio voice.

        Args:
            assistant_voice: name of the assistant voice to use.
        """
        self.initial_prompt = initial_prompt
        self.reprompt_fallback_phrase = reprompt_fallback_phrase
        self.assistant_voice = assistant_voice
        self.speech_timeout = speech_timeout
        self.speech_model = speech_model
        self.enhanced = enhanced

        if assistant_voice not in self.SUPPORTED_VOICES:
            self.raise_invalid_voice_exception()

        try:
            int(self.speech_timeout)
        except ValueError:
            if self.speech_timeout.lower() != "auto":
                self.raise_invalid_speech_timeout_exception()

        if speech_model not in self.SUPPORTED_SPEECH_MODELS:
            self.raise_invalid_speech_model_exception()

        if enhanced.lower() not in ["true", "false"]:
            self.raise_invalid_enhanced_option_exception()

        if (enhanced.lower() == "true") & (speech_model.lower() != "phone_call"):
            self.raise_invalid_enhanced_speech_model_exception()

        if (speech_model.lower() != "numbers_and_commands") & (speech_timeout.lower() == "auto"):
            self.raise_invalid_speech_model_timeout_exception()

    def raise_invalid_speech_model_timeout_exception(self) -> None:
        """Raises an error if incompatible speech_timeout and speech_model are provided."""
        raise InvalidConfigException(
            f"If speech_time is 'auto' the speech_model must be 'numbers_and_commands'. Please update your "
            f"speech_model to be 'numbers_and_commands' if you would like to continue using the 'auto' speech_model."
        )

    def raise_invalid_enhanced_option_exception(self) -> None:
        """Raises an error if an invalid value is passed to the enhanced parameter."""
        raise InvalidConfigException(
            f"{self.enhanced} is invalid. You must provide either `true` or `false` for this value."
        )

    def raise_invalid_speech_model_exception(self) -> None:
        """Raises an error if an invalid speech_model is provided."""
        raise InvalidConfigException(
            f"{self.speech_model} is invalid. You must choose one of 'default', 'numbers_and_commands', "
            f"or 'phone_call'. Refer to the documentation for details about the selections."
        )

    def raise_invalid_speech_timeout_exception(self) -> None:
        """Raises an error if an invalid speech_timeout is provided."""
        raise InvalidConfigException(
            f"{self.speech_timeout} is an invalid value for speech timeout. Only integers and 'auto' are valid entries."
        )

    def raise_invalid_voice_exception(self) -> None:
        """Raises an error if an invalid voice is provided."""
        raise InvalidConfigException(
            f"{self.assistant_voice} is an invalid as an assistant voice. Please refer to the documentation for a list "
            f"of valid voices you can use for your voice assistant."
        )

    def raise_invalid_enhanced_speech_model_exception(self) -> None:
        """Raises an error if enhanced is turned on and an incompatible speech_model is used."""
        raise InvalidConfigException(
            f"If you set enhanced to 'true' then speech_model must be 'phone_call'. Current speech_model is: "
            f"{self.speech_model}."
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

                # If no previous utterance found use the reprompt_fallback phrase.
                if last_response is None:
                    last_response = self.reprompt_fallback_phrase
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
            speechTimeout=self.speech_timeout,
            speechModel=self.speech_model,
            enhanced=self.enhanced,
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

    EMOJI = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "\u200d"  # zero width joiner
                "\u200c"  # zero width non-joiner
                "]+",
                flags=re.UNICODE,
            )

    @classmethod
    def name(cls) -> Text:
        """Name of the output channel."""
        return "twilio_voice"

    async def emoji_warning(
        self,
        text: Text,
    ) -> None:
        """Raises a warning if text contains an emoji."""
        if self.EMOJI.findall(text):
            raise_warning(
                "Text contains an emoji in a voice response. Review responses to provide a voice-friendly "
                "alternative."
            )

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        await self.emoji_warning(text)
        for message_part in text.strip().split("\n\n"):
            await self._persist_message(self._message(recipient_id, text=message_part))

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """Convert buttons into a voice representation."""
        await self.emoji_warning(text)
        await self._persist_message(self._message(recipient_id, text=text))

        for b in buttons:
            await self.emoji_warning(b["title"])
            await self._persist_message(self._message(recipient_id, text=b["title"]))

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """For voice channel do not send images."""
        raise_warning(
            "Image removed from voice message. Only text of message is sent."
        )
