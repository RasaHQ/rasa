from __future__ import annotations

import base64
import json
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Literal,
    Optional,
    Text,
    Union,
)

import structlog
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
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
from rasa.core.channels.voice_stream.call_state import (
    BotIsSpeaking,
    BotStoppedSpeaking,
    Marker,
    MarkerType,
    StepType,
    call_state,
)
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine
from rasa.core.channels.voice_stream.util import repack_voice_credentials
from rasa.core.channels.voice_stream.voice_channel import (
    ContinueConversationAction,
    DTMFInputAction,
    EndConversationAction,
    MarkerInput,
    MarkerMessageOutput,
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


class MarkerOutput(BaseModel):
    name: str
    step_type: Optional[StepType] = Field(
        default=None,
        validation_alias=AliasChoices("stepType", "step_type"),
        serialization_alias="stepType",
        exclude=True,
    )
    marker_type: Optional[MarkerType] = Field(
        default=None,
        validation_alias=AliasChoices("markerType", "marker_type"),
        serialization_alias="markerType",
        exclude=True,
    )

    def to_marker(self) -> Marker:
        return Marker(
            marker_id=self.name, marker_type=self.marker_type, step_type=self.step_type
        )


class MarkerMessage(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
    )

    event: Literal["mark"] = "mark"
    stream_sid: str = Field(
        validation_alias=AliasChoices("streamSid", "stream_sid"),
        serialization_alias="streamSid",
    )
    mark: MarkerOutput


class TwilioMediaStreamsOutputChannel(VoiceOutputChannel):
    @classmethod
    def name(cls) -> str:
        return "twilio_media_streams"

    def rasa_audio_bytes_to_channel_bytes(
        self, rasa_audio_bytes: RasaAudioBytes
    ) -> bytes:
        return base64.b64encode(rasa_audio_bytes.data)

    def create_marker_message(self, marker_input: MarkerInput) -> MarkerMessageOutput:
        mark_message = MarkerMessage(
            stream_sid=marker_input.recipient_id,
            mark=MarkerOutput(
                name=uuid.uuid4().hex,
                step_type=marker_input.step_type,
                marker_type=marker_input.marker_type,
            ),
        )

        logger.debug(
            "twilio_media_streams.create_marker_message",
            message_json=mark_message.model_dump_json(by_alias=True),
        )

        if marker_input.marker_type in {MarkerType.START, MarkerType.END}:
            call_state.set_marker(mark_message.mark.to_marker())

        mark_message_str = mark_message.model_dump_json(by_alias=True)
        return MarkerMessageOutput(
            message_id=mark_message.mark.name, message=mark_message_str
        )

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
        silence_timeout: Optional[Union[float, int]] = None,
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
        self.silence_timeout = silence_timeout

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
        return RasaAudioBytes(base64.b64decode(input_bytes), format=self.audio_format)

    async def collect_call_parameters(
        self,
        channel_websocket: Websocket,
        request: Optional[Any] = None,
    ) -> Optional[CallParameters]:
        async for message in channel_websocket:
            data = json.loads(message)
            if data["event"] == "start":
                # retrieve parameters set in the webhook - contains info about the
                # caller
                return map_call_params(data)
        return None

    async def map_input_message(
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
            marker_message = MarkerMessage.model_validate(data)

            marker = call_state.get_marker(marker_message.mark.name)

            if marker:
                call_state.remove_marker(marker.marker_id)

                if marker.step_type:
                    if marker.marker_type == MarkerType.START:
                        call_state.current_bot_utterance_type = marker.step_type
                        await call_state.enqueue_event(BotIsSpeaking())

                    if marker.marker_type == MarkerType.END:
                        call_state.current_bot_utterance_type = None
                        await call_state.enqueue_event(BotStoppedSpeaking())

                        if call_state.should_hangup:
                            logger.debug(
                                "twilio_streams.hangup",
                                marker=marker,
                            )
                            return EndConversationAction()
        return ContinueConversationAction()

    def create_output_channel(
        self, voice_websocket: Websocket, tts_engine: TTSEngine
    ) -> VoiceOutputChannel:
        return TwilioMediaStreamsOutputChannel(
            voice_websocket=voice_websocket,
            tts_engine=tts_engine,
            tts_cache=self.tts_cache,
            audio_format=self.audio_format,
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
        self._register_listeners(blueprint)

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
            logger.debug("twilio_media_streams.handle_message")
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
