import base64
import json
import structlog
from typing import Any, Awaitable, Callable, Dict, List, Optional, Text
import uuid

from sanic import Blueprint, HTTPResponse, Request, response
from sanic import Websocket  # type: ignore


from rasa.core.channels import UserMessage
from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine
from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.voice_channel import (
    EndConversationAction,
    NewAudioAction,
    VoiceChannelAction,
    ContinueConversationAction,
    VoiceInputChannel,
    VoiceOutputChannel,
)

structlogger = structlog.get_logger()


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

    def channel_bytes_to_messages(
        self, recipient_id: str, channel_bytes: bytes
    ) -> List[Any]:
        message_id = uuid.uuid4().hex
        media_message = json.dumps(
            {
                "event": "media",
                "streamSid": recipient_id,
                "media": {
                    "payload": channel_bytes.decode("utf-8"),
                },
            }
        )
        mark_message = json.dumps(
            {
                "event": "mark",
                "streamSid": recipient_id,
                "mark": {"name": message_id},
            }
        )
        self.latest_message_id = message_id
        return [media_message, mark_message]


class TwilioMediaStreamsInputChannel(VoiceInputChannel):
    @classmethod
    def name(cls) -> str:
        return "twilio_media_streams"

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
    ) -> VoiceChannelAction:
        data = json.loads(message)
        if data["event"] == "media":
            audio_bytes = self.channel_bytes_to_rasa_audio_bytes(
                data["media"]["payload"]
            )
            return NewAudioAction(audio_bytes)
        elif data["event"] == "stop":
            return EndConversationAction()
        elif data["event"] == "mark":
            if data["mark"]["name"] == self.hangup_after:
                structlogger.debug("twilio_streams.hangup", marker=self.hangup_after)
                return EndConversationAction()
        return ContinueConversationAction()

    def create_output_channel(
        self, voice_websocket: Websocket, tts_engine: TTSEngine
    ) -> VoiceOutputChannel:
        return TwilioMediaStreamsOutputChannel(
            voice_websocket,
            tts_engine,
            self.tts_cache,
        )

    def websocket_stream_url(self) -> str:
        """Returns the websocket stream URL."""
        # depending on the config value, the url might contain http as a
        # protocol or not - we'll make sure both work
        if self.server_url.startswith("http"):
            base_url = self.server_url.replace("http", "ws")
        else:
            base_url = f"wss://{self.server_url}"
        return f"{base_url}/webhooks/twilio_media_streams/websocket"

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """Defines a Sanic bluelogger.debug."""
        blueprint = Blueprint("twilio_media_streams", __name__)

        @blueprint.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @blueprint.route("/webhook", methods=["POST"])
        async def receive(request: Request) -> HTTPResponse:
            from twilio.twiml.voice_response import Connect, VoiceResponse

            voice_response = VoiceResponse()
            start = Connect()
            stream = start.stream(url=self.websocket_stream_url())
            # pass information about the call to the webhook - so we can
            # store it in the input channel
            stream.parameter(name="call_id", value=request.form.get("CallSid", None))
            stream.parameter(name="user_phone", value=request.form.get("From", None))
            stream.parameter(name="bot_phone", value=request.form.get("To", None))
            stream.parameter(
                name="direction", value=request.form.get("Direction", None)
            )

            voice_response.append(start)

            return response.text(str(voice_response), content_type="text/xml")

        @blueprint.websocket("/websocket")  # type: ignore
        async def handle_message(request: Request, ws: Websocket) -> None:
            await self.run_audio_streaming(on_new_message, ws)

        return blueprint
