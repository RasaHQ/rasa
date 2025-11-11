from __future__ import annotations

import base64
import json
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Optional, Text, Tuple

import structlog
from sanic import (  # type: ignore[attr-defined]
    Blueprint,
    HTTPResponse,
    Request,
    Websocket,
    response,
)

# Import twilio at module level to raise error if not installed
from twilio.twiml.voice_response import VoiceResponse

from rasa.core.channels import UserMessage
from rasa.core.channels.channel import (
    create_auth_requested_response_provider,
    requires_basic_auth,
)
from rasa.core.channels.voice_ready.utils import (
    CallParameters,
    validate_username_password_credentials,
)
from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.call_state import call_state
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine
from rasa.core.channels.voice_stream.util import repack_voice_credentials
from rasa.core.channels.voice_stream.voice_channel import (
    ContinueConversationAction,
    DTMFInputAction,
    EndConversationAction,
    NewAudioAction,
    VoiceChannelAction,
    VoiceInputChannel,
    VoiceOutputChannel,
)

if TYPE_CHECKING:
    from twilio.twiml.voice_response import VoiceResponse

logger = structlog.get_logger(__name__)


TWILIO_MEDIA_STREAMS_WEBHOOK_PATH = "webhooks/twilio_media_streams/webhook"
TWILIO_MEDIA_STREAMS_WEBSOCKET_PATH = "webhooks/twilio_media_streams/websocket"


CALL_SID_REQUEST_KEY = "CallSid"
FROM_NUMBER_REQUEST_KEY = "From"
TO_NUMBER_REQUEST_KEY = "To"
DIRECTION_REQUEST_KEY = "Direction"


def map_call_params(data: Dict[Text, Any]) -> CallParameters:
    """Map the twilio stream parameters to the CallParameters dataclass."""
    stream_sid = data["streamSid"]
    parameters = data["start"]["customParameters"]
    return CallParameters(
        call_id=parameters.get("call_id", ""),
        user_phone=parameters.get("user_phone", ""),
        bot_phone=parameters.get("bot_phone", ""),
        direction=parameters.get("direction"),
        stream_id=stream_sid,
    )


class TwilioMediaStreamsOutputChannel(VoiceOutputChannel):
    @classmethod
    def name(cls) -> str:
        return "twilio_media_streams"

    def rasa_audio_bytes_to_channel_bytes(
        self, rasa_audio_bytes: RasaAudioBytes
    ) -> bytes:
        return base64.b64encode(rasa_audio_bytes)

    def create_marker_message(self, recipient_id: str) -> Tuple[str, str]:
        message_id = uuid.uuid4().hex
        mark_message = json.dumps(
            {
                "event": "mark",
                "streamSid": recipient_id,
                "mark": {"name": message_id},
            }
        )
        return mark_message, message_id

    def channel_bytes_to_message(self, recipient_id: str, channel_bytes: bytes) -> str:
        media_message = json.dumps(
            {
                "event": "media",
                "streamSid": recipient_id,
                "media": {
                    "payload": channel_bytes.decode("utf-8"),
                },
            }
        )
        return media_message


class TwilioMediaStreamsInputChannel(VoiceInputChannel):
    def __init__(
        self,
        server_url: str,
        asr_config: Dict,
        tts_config: Dict,
        interruptions: Optional[Dict[str, int]] = None,
        username: Optional[Text] = None,
        password: Optional[Text] = None,
    ):
        super().__init__(
            server_url=server_url,
            asr_config=asr_config,
            tts_config=tts_config,
            interruptions=interruptions,
        )
        self.username = username
        self.password = password

    @classmethod
    def from_credentials(
        cls,
        credentials: Optional[Dict[str, Any]],
    ) -> VoiceInputChannel:
        cls.validate_credentials(credentials)
        new_creds = repack_voice_credentials(credentials)
        return cls(**new_creds)

    @classmethod
    def validate_credentials(
        cls,
        credentials: Optional[Dict[str, Any]],
    ) -> None:
        cls.validate_basic_credentials(credentials)
        username = credentials.get("username") if credentials else None
        password = credentials.get("password") if credentials else None
        validate_username_password_credentials(username, password, "TwilioMediaStreams")

    @classmethod
    def name(cls) -> str:
        return "twilio_media_streams"

    def get_sender_id(self, call_parameters: CallParameters) -> str:
        """Get the sender ID for the channel.

        Twilio Media Streams uses the Stream ID as Sender ID because
        it is required in OutputChannel.send_text_message to send messages.
        """
        return call_parameters.stream_id  # type: ignore[return-value]

    def channel_bytes_to_rasa_audio_bytes(self, input_bytes: bytes) -> RasaAudioBytes:
        return RasaAudioBytes(base64.b64decode(input_bytes))

    async def collect_call_parameters(
        self, channel_websocket: Websocket
    ) -> Optional[CallParameters]:
        async for message in channel_websocket:
            data = json.loads(message)
            if data["event"] == "start":
                # retrieve parameters set in the webhook - contains info about the
                # caller
                return map_call_params(data)
        return None

    def map_input_message(
        self,
        message: Any,
        ws: Websocket,
    ) -> VoiceChannelAction:
        data = json.loads(message)
        if data["event"] == "media":
            audio_bytes = self.channel_bytes_to_rasa_audio_bytes(
                data["media"]["payload"]
            )
            return NewAudioAction(audio_bytes)
        elif data["event"] == "stop":
            return EndConversationAction()
        elif data["event"] == "dtmf":
            return DTMFInputAction(digit=data["dtmf"]["digit"])
        elif data["event"] == "mark":
            if data["mark"]["name"] == call_state.latest_bot_audio_id:
                # Just finished streaming last audio bytes
                call_state.is_bot_speaking = False
                if call_state.should_hangup:
                    logger.debug(
                        "twilio_streams.hangup", marker=call_state.latest_bot_audio_id
                    )
                    return EndConversationAction()
            else:
                call_state.is_bot_speaking = True
        return ContinueConversationAction()

    def create_output_channel(
        self, voice_websocket: Websocket, tts_engine: TTSEngine
    ) -> VoiceOutputChannel:
        return TwilioMediaStreamsOutputChannel(
            voice_websocket,
            tts_engine,
            self.tts_cache,
        )

    async def interrupt_playback(
        self, ws: Websocket, call_parameters: CallParameters
    ) -> None:
        """Interrupt the current playback of audio."""
        logger.debug("twilio_media_streams.interrupt_playback")
        await ws.send(
            json.dumps(
                {
                    "event": "clear",
                    "streamSid": call_parameters.stream_id,
                }
            )
        )

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """Defines a Sanic blueprint for the voice input channel."""
        blueprint = Blueprint("twilio_media_streams", __name__)

        @blueprint.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @blueprint.route("/webhook", methods=["POST"])
        @requires_basic_auth(
            username=self.username,
            password=self.password,
            auth_request_provider=create_auth_requested_response_provider(
                realm=TWILIO_MEDIA_STREAMS_WEBHOOK_PATH
            ),
        )
        async def receive(request: Request) -> HTTPResponse:
            voice_response = self._build_twilio_response(request)

            logger.debug(
                "twilio_media_streams.webhook.twilio_response",
                twilio_response=str(voice_response),
            )

            return response.text(str(voice_response), content_type="text/xml")

        @blueprint.websocket("/websocket")  # type: ignore
        async def handle_message(request: Request, ws: Websocket) -> None:
            await self.run_audio_streaming(on_new_message, ws)

        return blueprint

    def _websocket_stream_url(self) -> str:
        """Returns the websocket stream URL."""
        # depending on the config value, the url might contain http as a
        # protocol or not - we'll make sure both work
        if self.server_url.startswith("http"):
            base_url = self.server_url.replace("http", "ws")
        else:
            base_url = f"wss://{self.server_url}"
        return f"{base_url}/{TWILIO_MEDIA_STREAMS_WEBSOCKET_PATH}"

    def _build_twilio_response(self, request: Request) -> VoiceResponse:
        from twilio.twiml.voice_response import Connect, VoiceResponse

        voice_response = VoiceResponse()
        start = Connect()
        stream = start.stream(url=self._websocket_stream_url())
        # pass information about the call to the webhook - so we can
        # store it in the input channel
        stream.parameter(
            name="call_id", value=request.form.get(CALL_SID_REQUEST_KEY, None)
        )
        stream.parameter(
            name="user_phone", value=request.form.get(FROM_NUMBER_REQUEST_KEY, None)
        )
        stream.parameter(
            name="bot_phone", value=request.form.get(TO_NUMBER_REQUEST_KEY, None)
        )
        stream.parameter(
            name="direction", value=request.form.get(DIRECTION_REQUEST_KEY, None)
        )
        voice_response.append(start)
        return voice_response
