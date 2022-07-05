from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from typing import Text, Callable, Awaitable, List, Any, Dict, Optional

import rasa.utils.io
import rasa.shared.utils.io
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

    SUPPORTED_SPEECH_MODELS = ["default", "numbers_and_commands", "phone_call"]

    @classmethod
    def name(cls) -> Text:
        """Name of channel."""
        return "twilio_voice"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        """Load custom configurations."""
        credentials = credentials or {}

        return cls(
            credentials.get("initial_prompt", "hello"),
            credentials.get(
                "reprompt_fallback_phrase",
                "I'm sorry I didn't get that could you rephrase.",
            ),
            credentials.get("assistant_voice", "woman"),
            credentials.get("speech_timeout", "5"),
            credentials.get("speech_model", "default"),
            credentials.get("enhanced", "false"),
        )

    def __init__(
        self,
        initial_prompt: Optional[Text],
        reprompt_fallback_phrase: Optional[Text],
        assistant_voice: Optional[Text],
        speech_timeout: Optional[Text],
        speech_model: Optional[Text],
        enhanced: Optional[Text],
    ) -> None:
        """Creates a connection to Twilio voice.

        Args:
            initial_prompt: text to use to prompt a conversation when call is answered.
            reprompt_fallback_phrase: phrase to use if no user response.
            assistant_voice: name of the assistant voice to use.
            speech_timeout: how long to pause when user finished speaking.
            speech_model: type of transcription model to use from Twilio.
            enhanced: toggle to use Twilio's premium speech transcription model.
        """
        self.initial_prompt = initial_prompt
        self.reprompt_fallback_phrase = reprompt_fallback_phrase
        self.assistant_voice = assistant_voice
        self.speech_timeout = speech_timeout
        self.speech_model = speech_model
        self.enhanced = enhanced

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Checks that the user configurations are valid."""
        if self.assistant_voice not in self.SUPPORTED_VOICES:
            self._raise_invalid_voice_exception()

        try:
            int(self.speech_timeout)
        except ValueError:
            if self.speech_timeout.lower() != "auto":
                self._raise_invalid_speech_timeout_exception()

        if self.speech_model not in self.SUPPORTED_SPEECH_MODELS:
            self._raise_invalid_speech_model_exception()

        if self.enhanced.lower() not in ["true", "false"]:
            self._raise_invalid_enhanced_option_exception()

        if (self.enhanced.lower() == "true") and (
            self.speech_model.lower() != "phone_call"
        ):
            self._raise_invalid_enhanced_speech_model_exception()

        if (self.speech_model.lower() != "numbers_and_commands") and (
            self.speech_timeout.lower() == "auto"
        ):
            self._raise_invalid_speech_model_timeout_exception()

    def _raise_invalid_speech_model_timeout_exception(self) -> None:
        """Raises an error if incompatible speech_timeout and speech_model used."""
        raise InvalidConfigException(
            "If speech_timeout is 'auto' the speech_model must be "
            "'numbers_and_commands'. Please update your speech_model "
            "to be 'numbers_and_commands' if you would like to continue "
            "using the 'auto' speech_model."
        )

    def _raise_invalid_enhanced_option_exception(self) -> None:
        """Raises an error if an invalid value is passed to the enhanced parameter."""
        raise InvalidConfigException(
            f"The value {self.enhanced} is invalid for the enhanced parameter. "
            f"You must provide either `true` or `false` for this value."
        )

    def _raise_invalid_speech_model_exception(self) -> None:
        """Raises an error if an invalid speech_model is provided."""
        raise InvalidConfigException(
            f"The value {self.speech_model} for speech_model is invalid. "
            f"You must choose one of 'default', 'numbers_and_commands', "
            f"or 'phone_call'. Refer to the documentation for details "
            f"about the selections."
        )

    def _raise_invalid_speech_timeout_exception(self) -> None:
        """Raises an error if an invalid speech_timeout is provided."""
        raise InvalidConfigException(
            f"The vale {self.speech_timeout} is an invalid value for speech_timeout. "
            f"Only integers and 'auto' are valid entries."
        )

    def _raise_invalid_voice_exception(self) -> None:
        """Raises an error if an invalid voice is provided."""
        raise InvalidConfigException(
            f"The value {self.assistant_voice} is an invalid for assistant_voice. "
            f"Please refer to the documentation for a list of valid voices "
            f"you can use for your voice assistant."
        )

    def _raise_invalid_enhanced_speech_model_exception(self) -> None:
        """Raises error if enhanced is used with an incompatible speech_model."""
        raise InvalidConfigException(
            f"If you set enhanced to 'true' then speech_model must be 'phone_call'. "
            f"Current speech_model is: {self.speech_model}."
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
        async def receive(request: Request) -> Text:
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

                twilio_response = self._build_twilio_voice_response(collector.messages)
            # If the user doesn't respond resend the last message.
            else:
                # Get last user utterance from tracker.
                tracker = request.app.ctx.agent.tracker_store.retrieve(sender_id)
                last_response = None
                if tracker:
                    last_response = next(
                        (
                            e
                            for e in reversed(tracker.events)
                            if isinstance(e, BotUttered)
                        ),
                        None,
                    )

                # If no previous utterance found use the reprompt_fallback phrase.
                if last_response is None:
                    last_response = self.reprompt_fallback_phrase
                else:
                    last_response = last_response.text

                twilio_response = self._build_twilio_voice_response(
                    [{"text": last_response}]
                )
            return response.text(str(twilio_response), content_type="text/xml")

        return twilio_voice_webhook

    def _build_twilio_voice_response(
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
            msg_text = message["text"]
            if i + 1 == len(messages):
                gather.say(msg_text, voice=self.assistant_voice)
                voice_response.append(gather)
            else:
                voice_response.say(msg_text, voice=self.assistant_voice)
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

    @staticmethod
    def _emoji_warning(text: Text,) -> None:
        """Raises a warning if text contains an emoji."""
        emoji_regex = rasa.utils.io.get_emoji_regex()
        if emoji_regex.findall(text):
            rasa.shared.utils.io.raise_warning(
                "Text contains an emoji in a voice response. "
                "Review responses to provide a voice-friendly alternative."
            )

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Sends the text message after removing emojis."""
        self._emoji_warning(text)
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
        self._emoji_warning(text)
        await self._persist_message(self._message(recipient_id, text=text))

        for b in buttons:
            self._emoji_warning(b["title"])
            await self._persist_message(self._message(recipient_id, text=b["title"]))

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """For voice channel do not send images."""
        rasa.shared.utils.io.raise_warning(
            "An image was removed from the voice message and "
            "only the text of message was sent. "
            "It's recommended that you define voice-friendly "
            "alternatives for all responses "
            "with a visual elements such as images and emojis "
            "that are used in your voice channel."
        )
