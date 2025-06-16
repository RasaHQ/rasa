import audioop
import base64
import json
import uuid
from typing import Any, Awaitable, Callable, Optional, Tuple

import structlog
from sanic import (  # type: ignore[attr-defined]
    Blueprint,
    HTTPResponse,
    Request,
    Websocket,
    response,
)

from rasa.core.channels import UserMessage
from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.call_state import call_state
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine
from rasa.core.channels.voice_stream.voice_channel import (
    ContinueConversationAction,
    EndConversationAction,
    NewAudioAction,
    VoiceChannelAction,
    VoiceInputChannel,
    VoiceOutputChannel,
)

logger = structlog.get_logger()


class BrowserAudioOutputChannel(VoiceOutputChannel):
    @classmethod
    def name(cls) -> str:
        return "browser_audio"

    def rasa_audio_bytes_to_channel_bytes(
        self, rasa_audio_bytes: RasaAudioBytes
    ) -> bytes:
        return audioop.ulaw2lin(rasa_audio_bytes, 4)

    def channel_bytes_to_message(self, recipient_id: str, channel_bytes: bytes) -> str:
        return json.dumps({"audio": base64.b64encode(channel_bytes).decode("utf-8")})

    def create_marker_message(self, recipient_id: str) -> Tuple[str, str]:
        message_id = uuid.uuid4().hex
        return json.dumps({"marker": message_id}), message_id


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
        ws: Websocket,
    ) -> VoiceChannelAction:
        data = json.loads(message)
        if "audio" in data:
            channel_bytes = base64.b64decode(data["audio"])
            audio_bytes = self.channel_bytes_to_rasa_audio_bytes(channel_bytes)
            return NewAudioAction(audio_bytes)
        elif "marker" in data:
            if data["marker"] == call_state.latest_bot_audio_id:
                # Just finished streaming last audio bytes
                call_state.is_bot_speaking = False  # type: ignore[attr-defined]
                if call_state.should_hangup:
                    logger.debug(
                        "browser_audio.hangup", marker=call_state.latest_bot_audio_id
                    )
                    return EndConversationAction()
            else:
                call_state.is_bot_speaking = True  # type: ignore[attr-defined]
        return ContinueConversationAction()

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
        """Defines a Sanic blueprint"""
        blueprint = Blueprint("browser_audio", __name__)

        @blueprint.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @blueprint.websocket("/websocket")  # type: ignore
        async def handle_message(request: Request, ws: Websocket) -> None:
            try:
                await self.run_audio_streaming(on_new_message, ws)
            except Exception as e:
                logger.error("browser_audio.handle_message.error", error=e)

        return blueprint
