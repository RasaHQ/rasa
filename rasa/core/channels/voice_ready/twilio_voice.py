from dataclasses import asdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Text

import structlog
from sanic import Blueprint, response
from sanic.request import Request, RequestParameters
from sanic.response import HTTPResponse
from twilio.twiml.voice_response import Gather, VoiceResponse

import rasa.shared.utils.io
import rasa.utils.io
from rasa.core.channels.channel import (
    CollectingOutputChannel,
    InputChannel,
    UserMessage,
    create_auth_requested_response_provider,
    requires_basic_auth,
)
from rasa.core.channels.voice_ready.utils import (
    CallParameters,
    validate_username_password_credentials,
)
from rasa.shared.core.events import BotUttered
from rasa.shared.exceptions import InvalidConfigException

logger = structlog.get_logger(__name__)


TWILIO_VOICE_PATH = "webhooks/twilio_voice/webhook"


def map_call_params(form: RequestParameters) -> CallParameters:
    """Map the Twilio Voice parameters to the CallParameters dataclass."""
    return CallParameters(
        call_id=form.get("CallSid"),
        user_phone=form.get("Caller"),
        bot_phone=form.get("Called"),
        direction=form.get("Direction"),
    )


class TwilioVoiceInput(InputChannel):
    """Input channel for Twilio Voice."""

    SUPPORTED_VOICES = [  # noqa: RUF012
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
        "Polly.Aditi",
    ]

    SUPPORTED_SPEECH_MODELS = [  # noqa: RUF012
        "default",
        "numbers_and_commands",
        "phone_call",
    ]

    @classmethod
    def name(cls) -> Text:
        """Name of channel."""
        return "twilio_voice"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        """Load custom configurations."""
        credentials = credentials or {}

        username = credentials.get("username")
        password = credentials.get("password")
        validate_username_password_credentials(username, password, "TwilioVoice")

        return cls(
            credentials.get(
                "reprompt_fallback_phrase",
                "I'm sorry I didn't get that could you rephrase.",
            ),
            credentials.get("assistant_voice", "woman"),
            credentials.get("speech_timeout", "5"),
            credentials.get("speech_model", "default"),
            credentials.get("enhanced", "false"),
            username=username,
            password=password,
        )

    def __init__(
        self,
        reprompt_fallback_phrase: Optional[Text],
        assistant_voice: Optional[Text],
        speech_timeout: Text = "5",
        speech_model: Text = "default",
        enhanced: Text = "false",
        username: Optional[Text] = None,
        password: Optional[Text] = None,
    ) -> None:
        """Creates a connection to Twilio voice.

        Args:
            reprompt_fallback_phrase: phrase to use if no user response.
            assistant_voice: name of the assistant voice to use.
            speech_timeout: how long to pause when user finished speaking.
            speech_model: type of transcription model to use from Twilio.
            enhanced: toggle to use Twilio's premium speech transcription model.
        """
        self.reprompt_fallback_phrase = reprompt_fallback_phrase
        self.assistant_voice = assistant_voice
        self.speech_timeout = speech_timeout
        self.speech_model = speech_model
        self.enhanced = enhanced
        self.username = username
        self.password = password

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Checks that the user configurations are valid."""
        if self.assistant_voice not in self.SUPPORTED_VOICES:
            self._raise_invalid_voice_exception()

        validate_username_password_credentials(
            self.username, self.password, "TwilioVoice"
        )

        try:
            int(self.speech_timeout)
        except ValueError:
            if self.speech_timeout.lower() != "auto":
                self._raise_invalid_speech_timeout_exception()

        if self.speech_model not in self.SUPPORTED_SPEECH_MODELS:
            self._raise_invalid_speech_model_exception()

        if self.enhanced.lower() not in [
            "true",
            "false",
        ]:
            self._raise_invalid_enhanced_option_exception()

        if (
            self.enhanced.lower() == "true"
            and self.speech_model.lower() != "phone_call"
        ):
            self._raise_invalid_enhanced_speech_model_exception()

        if (
            self.speech_model.lower() != "numbers_and_commands"
            and self.speech_timeout.lower() == "auto"
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
        @requires_basic_auth(
            username=self.username,
            password=self.password,
            auth_request_provider=create_auth_requested_response_provider(
                TWILIO_VOICE_PATH
            ),
        )
        async def receive(request: Request) -> HTTPResponse:
            sender_id = request.form.get("From")
            text = request.form.get("SpeechResult")
            input_channel = self.name()
            call_status = request.form.get("CallStatus")
            metadata = {}

            collector = TwilioVoiceCollectingOutputChannel()

            logger.debug(
                "twilio_voice.webhook",
                sender_id=sender_id,
                text=text,
                call_status=call_status,
            )
            # Provide an initial greeting to answer the user's call.
            if (text is None) and (call_status == "ringing"):
                text = "/session_start"
                metadata = asdict(map_call_params(request.form))

            # when call is disconnected
            if call_status == "completed":
                text = "/session_end"
                metadata = {"reason": "user disconnected"}

            # determine the response.
            if text is not None:
                logger.info("twilio_voice.webhook.text_not_none", sender_id=sender_id)
                await on_new_message(
                    UserMessage(
                        text,
                        collector,
                        sender_id,
                        input_channel=input_channel,
                        metadata=metadata,
                    )
                )

                twilio_response = self._build_twilio_voice_response(collector.messages)
            # If the user doesn't respond resend the last message.
            else:
                logger.info("twilio_voice.webhook.text_none", sender_id=sender_id)
                # Get last user utterance from tracker.
                tracker = await request.app.ctx.agent.tracker_store.retrieve(sender_id)
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
                    last_response_text = self.reprompt_fallback_phrase
                else:
                    last_response_text = last_response.text

                twilio_response = self._build_twilio_voice_response(
                    [{"text": last_response_text}]
                )

            logger.debug(
                "twilio_voice.webhook.twilio_response",
                twilio_response=str(twilio_response),
            )
            return response.text(str(twilio_response), content_type="text/xml")

        return twilio_voice_webhook

    def _build_twilio_voice_response(
        self, messages: List[Dict[Text, Any]]
    ) -> VoiceResponse:
        """Builds the Twilio Voice Response object."""
        logger.debug("twilio_voice.build_twilio_voice_response", messages=messages)
        voice_response = VoiceResponse()
        gather = Gather(
            input="speech",
            action=f"/webhooks/{self.name()}/webhook",
            actionOnEmptyResult=True,
            speechTimeout=self.speech_timeout,
            speechModel=self.speech_model,
            enhanced=self.enhanced,
        )

        if not messages:
            # In case bot has a greet message disabled
            # or if the bot is not configured to send an initial message
            # we need to send a voice response with speech settings
            voice_response.append(gather)
            return voice_response

        # Add pauses between messages.
        # Add a listener to the last message to listen for user response.
        for i, message in enumerate(messages):
            msg_text = message["text"]
            # Check if the message is a hangup message.
            if message.get("custom", {}).get("hangup"):
                voice_response.hangup()
                break

            if i + 1 == len(messages):
                gather.say(msg_text, voice=self.assistant_voice)
                voice_response.append(gather)
            else:
                voice_response.say(msg_text, voice=self.assistant_voice)
                voice_response.pause(length=1)

        return voice_response

    def _raise_invalid_credentials_exception(self) -> None:
        # This method is now redundant since we use the shared validation function
        # but keeping it for backward compatibility if any external code calls it
        validate_username_password_credentials(
            self.username, self.password, "TwilioVoice"
        )


class TwilioVoiceCollectingOutputChannel(CollectingOutputChannel):
    """Output channel that collects send messages in a list.

    (doesn't send them anywhere, just collects them).
    """

    @classmethod
    def name(cls) -> Text:
        """Name of the output channel."""
        return "twilio_voice"

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Sends the text message after removing emojis."""
        text = rasa.utils.io.remove_emojis(text)
        for message_part in text.strip().split("\n\n"):
            await self._persist_message(self._message(recipient_id, text=message_part))

    async def send_text_with_buttons(
        self,
        recipient_id: str,
        text: str,
        buttons: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Uses the concise button output format for voice channels."""
        await self.send_text_with_buttons_concise(recipient_id, text, buttons, **kwargs)

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

    async def hangup(self, recipient_id: Text, **kwargs: Any) -> None:
        """
        Indicate that the conversation should be ended.

        Parent class is a collecting output channel, so we don't actually hang up
        but we add a custom message to the list of messages to be sent.
        This message will be picked up by _build_twilio_voice_response
        which will hang up the call.
        """
        await self._persist_message(
            self._message(recipient_id, custom={"hangup": True})
        )
