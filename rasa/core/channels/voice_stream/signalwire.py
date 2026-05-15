from __future__ import annotations

import base64
import json
import uuid
import xml.etree.ElementTree as ET
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Text,
    Union,
)

import structlog
from sanic import (  # type: ignore[attr-defined]
    Blueprint,
    HTTPResponse,
    Request,
    Websocket,
    response,
)

from rasa.core.channels import UserMessage
from rasa.core.channels.voice_ready.utils import (
    CallParameters,
    validate_username_password_credentials,
)
from rasa.core.channels.voice_stream.audio_bytes import L16_24KHZ, RasaAudioBytes
from rasa.core.channels.voice_stream.call_state import (
    BotIsSpeaking,
    BotStoppedSpeaking,
    Marker,
    MarkerType,
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

logger = structlog.get_logger(__name__)

SIGNALWIRE_WEBSOCKET_PATH = "webhooks/signalwire/websocket"
SIGNALWIRE_STREAM_CODEC = "L16@24000h"

CALL_SID_REQUEST_KEY = "CallSid"
FROM_NUMBER_REQUEST_KEY = "From"
TO_NUMBER_REQUEST_KEY = "To"
DIRECTION_REQUEST_KEY = "Direction"


def _request_value(request: Request, key: Text) -> Text:
    """Return a SignalWire webhook value from form, query, or JSON data."""
    value = request.form.get(key) if request.form else None
    if value is None and request.args:
        value = request.args.get(key)
    content_type = request.headers.get("content-type", "")
    if value is None and "json" in content_type and isinstance(request.json, dict):
        value = request.json.get(key)
    return str(value) if value is not None else ""


def _websocket_stream_url(server_url: str) -> str:
    """Return the websocket stream URL."""
    # depending on the config value, the url might contain http as a
    # protocol or not - we'll make sure both work
    normalized_server_url = server_url.strip().rstrip("/")
    if not normalized_server_url:
        raise ValueError("SignalWire server_url must be configured")
    if normalized_server_url.startswith("http"):
        base_url = normalized_server_url.replace("http", "ws")
        if base_url.startswith("ws://"):
            logger.warning(
                "signalwire.websocket_stream_url",
                message=(
                    "Forcing wss protocol, since ws is not supported by SignalWire"
                ),
            )
            base_url = f"wss://{base_url[len('ws://'):]}"
    else:
        base_url = f"wss://{normalized_server_url}"
    return f"{base_url}/{SIGNALWIRE_WEBSOCKET_PATH}"


def _signalwire_call_parameters(request: Request) -> Dict[Text, Text]:
    """Map SignalWire webhook fields into stream custom parameters."""
    return {
        "call_id": _request_value(request, CALL_SID_REQUEST_KEY),
        "user_phone": _request_value(request, FROM_NUMBER_REQUEST_KEY),
        "bot_phone": _request_value(request, TO_NUMBER_REQUEST_KEY),
        "direction": _request_value(request, DIRECTION_REQUEST_KEY),
    }


def _signalwire_stream_response(
    websocket_url: Text, call_parameters: Dict[Text, Text]
) -> Text:
    """Create SignalWire cXML that connects the call to Rasa audio streaming."""
    root = ET.Element("Response")
    connect = ET.SubElement(root, "Connect")
    stream = ET.SubElement(
        connect,
        "Stream",
        {"url": websocket_url, "codec": SIGNALWIRE_STREAM_CODEC},
    )
    for name, value in call_parameters.items():
        ET.SubElement(stream, "Parameter", {"name": name, "value": value})

    xml_body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_body}'


def map_call_params(data: Dict[Text, Any]) -> CallParameters:
    """Map the SignalWire stream parameters to the CallParameters dataclass."""
    stream_sid = data["start"]["streamSid"]
    parameters = data["start"]["customParameters"]
    return CallParameters(
        call_id=parameters.get("call_id", ""),
        user_phone=parameters.get("user_phone", ""),
        bot_phone=parameters.get("bot_phone", ""),
        direction=parameters.get("direction"),
        stream_id=stream_sid,
    )


class SignalWireOutputChannel(VoiceOutputChannel):
    @classmethod
    def name(cls) -> str:
        return "signalwire"

    def rasa_audio_bytes_to_channel_bytes(
        self, rasa_audio_bytes: RasaAudioBytes
    ) -> bytes:
        return base64.b64encode(rasa_audio_bytes.data)

    def create_marker_message(self, marker_input: MarkerInput) -> MarkerMessageOutput:
        message_id = uuid.uuid4().hex
        mark_message = json.dumps(
            {
                "event": "mark",
                "streamSid": marker_input.recipient_id,
                "mark": {"name": message_id},
            }
        )

        if marker_input.marker_type in {MarkerType.START, MarkerType.END}:
            call_state.set_marker(
                Marker(
                    marker_id=message_id,
                    marker_type=marker_input.marker_type,
                    step_type=marker_input.step_type,
                )
            )

        return MarkerMessageOutput(message_id=message_id, message=mark_message)

    def channel_bytes_to_message(self, recipient_id: str, channel_bytes: bytes) -> str:
        return json.dumps(
            {
                "event": "media",
                "streamSid": recipient_id,
                "media": {
                    "payload": channel_bytes.decode("utf-8"),
                },
            }
        )


class SignalWireInputChannel(VoiceInputChannel):
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
        self.audio_format = L16_24KHZ

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
        validate_username_password_credentials(username, password, "SignalWire")

    @classmethod
    def name(cls) -> str:
        return "signalwire"

    def get_sender_id(self, call_parameters: CallParameters) -> str:
        """Get the sender ID for the channel.

        SignalWire uses the Stream ID as sender ID because it is required
        in OutputChannel.send_text_message to address messages back to the stream.
        """
        return call_parameters.stream_id  # type: ignore[return-value]

    def channel_bytes_to_rasa_audio_bytes(self, input_bytes: bytes) -> RasaAudioBytes:
        return RasaAudioBytes(base64.b64decode(input_bytes), format=self.audio_format)

    async def collect_call_parameters(
        self,
        channel_websocket: Websocket,
        request: Optional[Any] = None,
    ) -> Optional[CallParameters]:
        logger.info("signalwire.collect_call_parameters")
        async for message in channel_websocket:
            data = json.loads(message)
            if data["event"] == "connected":
                logger.info("signalwire.call_connected", data=data)
            elif data["event"] == "start":
                logger.info("signalwire.call_started", data=data)
                return map_call_params(data)
            else:
                logger.warning(
                    "signalwire.collect_call_parameters.unexpected_event",
                    payload=data,
                )
        return None

    async def handle_mark_event(
        self, data: Dict[str, Any]
    ) -> Optional[EndConversationAction]:
        marker = call_state.get_marker(data["mark"]["name"])
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
                        logger.debug("signalwire.hangup", marker=marker)
                        return EndConversationAction()
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
            mark_event = await self.handle_mark_event(data)
            if mark_event is not None:
                return mark_event
        else:
            logger.warning("signalwire.unknown_event_received", sw_event=data["event"])
        return ContinueConversationAction()

    def create_output_channel(
        self, voice_websocket: Websocket, tts_engine: TTSEngine
    ) -> VoiceOutputChannel:
        return SignalWireOutputChannel(
            voice_websocket=voice_websocket,
            tts_engine=tts_engine,
            tts_cache=self.tts_cache,
            audio_format=self.audio_format,
        )

    async def interrupt_playback(
        self, ws: Websocket, call_parameters: CallParameters
    ) -> None:
        """Interrupt the current playback of audio."""
        logger.debug("signalwire.interrupt_playback")
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
        """Defines a Sanic blueprint for the SignalWire voice input channel."""
        blueprint = Blueprint("signalwire", __name__)
        self._register_listeners(blueprint)

        @blueprint.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @blueprint.route("/webhook", methods=["GET", "POST"])
        async def webhook(request: Request) -> HTTPResponse:
            call_parameters = _signalwire_call_parameters(request)
            websocket_url = _websocket_stream_url(self.server_url)
            logger.info(
                "signalwire.script_received",
                call_id=call_parameters["call_id"],
                from_=call_parameters["user_phone"],
                to=call_parameters["bot_phone"],
                websocket_url=websocket_url,
            )
            return response.text(
                _signalwire_stream_response(websocket_url, call_parameters),
                content_type="application/xml",
            )

        @blueprint.route("/call_status", methods=["POST"])
        async def call_status(request: Request) -> HTTPResponse:
            logger.debug("signalwire.call_status", request_data=await request.json())
            return response.json({"status": "ok"})

        @blueprint.websocket("/websocket")  # type: ignore
        async def handle_message(request: Request, ws: Websocket) -> None:
            logger.info("signalwire.handle_message")
            try:
                await self.run_audio_streaming(on_new_message, ws)
            except Exception as e:
                logger.error("signalwire.websocket_error", error=str(e))

        return blueprint
