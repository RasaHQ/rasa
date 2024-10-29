import audioop
import structlog
import uuid
from typing import Any, Awaitable, Callable, List, Optional

from sanic import Blueprint, HTTPResponse, Request, response
from sanic import Websocket  # type: ignore


from rasa.core.channels import UserMessage
from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine
from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.voice_channel import (
    NewAudioAction,
    VoiceChannelAction,
    VoiceInputChannel,
    VoiceOutputChannel,
)

structlogger = structlog.get_logger()


class BrowserAudioOutputChannel(VoiceOutputChannel):
    @classmethod
    def name(cls) -> str:
        return "browser_audio"

    def rasa_audio_bytes_to_channel_bytes(
        self, rasa_audio_bytes: RasaAudioBytes
    ) -> bytes:
        return audioop.ulaw2lin(rasa_audio_bytes, 4)

    def channel_bytes_to_messages(
        self, recipient_id: str, channel_bytes: bytes
    ) -> List[Any]:
        return [channel_bytes]


class BrowserAudioInputChannel(VoiceInputChannel):
    @classmethod
    def name(cls) -> str:
        return "browser_audio"

    def channel_bytes_to_rasa_audio_bytes(self, input_bytes: bytes) -> RasaAudioBytes:
        return RasaAudioBytes(audioop.lin2ulaw(input_bytes, 4))

    async def collect_call_parameters(
        self, channel_websocket: Websocket
    ) -> Optional[CallParameters]:
        call_id = f"inspect-{uuid.uuid4()}"
        return CallParameters(call_id, "local", "local", stream_id=call_id)

    def map_input_message(
        self,
        message: Any,
    ) -> VoiceChannelAction:
        audio_bytes = self.channel_bytes_to_rasa_audio_bytes(message)
        return NewAudioAction(audio_bytes)

    def create_output_channel(
        self, voice_websocket: Websocket, tts_engine: TTSEngine
    ) -> VoiceOutputChannel:
        return BrowserAudioOutputChannel(
            voice_websocket,
            tts_engine,
            self.tts_cache,
        )

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """Defines a Sanic bluelogger.debug."""
        blueprint = Blueprint("browser_audio", __name__)

        @blueprint.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @blueprint.websocket("/websocket")  # type: ignore
        async def handle_message(request: Request, ws: Websocket) -> None:
            await self.run_audio_streaming(on_new_message, ws)

        return blueprint
