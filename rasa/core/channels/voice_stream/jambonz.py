from __future__ import annotations

import audioop
import json
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional, Text, Tuple

import structlog
from sanic import (  # type: ignore[attr-defined]
    Blueprint,
    HTTPResponse,
    Request,
    Websocket,
    response,
)

from rasa.core.channels import UserMessage, requires_basic_auth
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

logger = structlog.get_logger()

JAMBONZ_STREAMS_WEBSOCKET_PATH = "webhooks/jambonz_stream/websocket"


def map_call_params(data: Dict[Text, str]) -> CallParameters:
    """Map the Jambonz stream parameters to the CallParameters dataclass."""
    call_sid = data.get("callSid", "None")
    from_number = data.get("from", "Unknown")
    to_number = data.get("to")
    return CallParameters(
        call_id=call_sid,
        user_phone=from_number,
        bot_phone=to_number,
        stream_id=call_sid,
    )


class JambonzStreamOutputChannel(VoiceOutputChannel):
    @classmethod
    def name(cls) -> str:
        return "jambonz_stream"

    async def send_audio_bytes(
        self, recipient_id: str, audio_bytes: RasaAudioBytes
    ) -> None:
        """Overridden to send binary websocket messages for Jambonz.

        Converts 8kHz μ-law to 8kHz L16 PCM for Jambonz streaming.
        """
        pcm = audioop.ulaw2lin(audio_bytes, 2)
        await self.voice_websocket.send(pcm)

    def create_marker_message(self, recipient_id: str) -> Tuple[str, str]:
        """Create a marker message to track audio stream position."""
        marker_id = uuid.uuid4().hex
        return json.dumps({"type": "mark", "data": {"name": marker_id}}), marker_id


class JambonzStreamInputChannel(VoiceInputChannel):
    @classmethod
    def name(cls) -> str:
        return "jambonz_stream"

    def __init__(
        self,
        server_url: str,
        asr_config: Dict,
        tts_config: Dict,
        interruptions: Optional[Dict[str, int]] = None,
        username: Optional[Text] = None,
        password: Optional[Text] = None,
    ) -> None:
        """Initialize the channel.

        Args:
            username: Optional username for basic auth
            password: Optional password for basic auth
        """
        super().__init__(server_url, asr_config, tts_config, interruptions)
        self.username = username
        self.password = password

    @classmethod
    def from_credentials(
        cls, credentials: Optional[Dict[Text, Any]]
    ) -> JambonzStreamInputChannel:
        """Create a channel from credentials dictionary.

        Args:
            credentials: Dictionary containing the required credentials:
                - server_url: URL where the server is hosted
                - asr: ASR engine configuration
                - tts: TTS engine configuration
                - username: Optional username for basic auth
                - password: Optional password for basic auth

        Returns:
            JambonzStreamInputChannel instance
        """
        # Get common credentials from parent
        cls.validate_credentials(credentials)
        new_creds = repack_voice_credentials(credentials)
        return cls(**new_creds)

    @classmethod
    def validate_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> None:
        cls.validate_basic_credentials(credentials)
        # Check optional basic auth credentials
        username = credentials.get("username")  # type: ignore[union-attr]
        password = credentials.get("password")  # type: ignore[union-attr]
        validate_username_password_credentials(username, password, "Jambonz Stream")

    def _websocket_stream_url(self) -> str:
        """Returns the websocket stream URL."""
        # depending on the config value, the url might contain http as a
        # protocol or not - we'll make sure both work
        if self.server_url.startswith("http"):
            base_url = self.server_url.replace("http", "ws")
        else:
            base_url = f"wss://{self.server_url}"
        return f"{base_url}/{JAMBONZ_STREAMS_WEBSOCKET_PATH}"

    def channel_bytes_to_rasa_audio_bytes(self, input_bytes: bytes) -> RasaAudioBytes:
        """Convert Jambonz audio bytes (L16 PCM) to Rasa audio bytes (μ-law)."""
        ulaw = audioop.lin2ulaw(input_bytes, 2)
        return RasaAudioBytes(ulaw)

    async def collect_call_parameters(
        self, channel_websocket: Websocket
    ) -> Optional[CallParameters]:
        # Wait for initial metadata message
        message = await channel_websocket.recv()
        logger.debug("jambonz.collect_call_parameters", message=message)
        metadata = json.loads(message)
        return map_call_params(metadata)

    def map_input_message(self, message: Any, ws: Websocket) -> VoiceChannelAction:
        # Handle binary audio frames
        if isinstance(message, bytes):
            channel_bytes = message
            audio_bytes = self.channel_bytes_to_rasa_audio_bytes(channel_bytes)
            return NewAudioAction(audio_bytes)

        # Handle JSON messages
        data = json.loads(message)
        if data.get("event") == "dtmf":
            return DTMFInputAction(
                digit=data["dtmf"],
            )
        if data.get("type") == "mark":
            if data["data"]["name"] == call_state.latest_bot_audio_id:
                # Just finished streaming last audio bytes
                call_state.is_bot_speaking = False
                if call_state.should_hangup:
                    logger.debug(
                        "jambonz.hangup", marker=call_state.latest_bot_audio_id
                    )
                    return EndConversationAction()
            else:
                call_state.is_bot_speaking = True
        else:
            logger.warning("jambonz.unexpected_message", message=data)

        return ContinueConversationAction()

    def create_output_channel(
        self, voice_websocket: Websocket, tts_engine: TTSEngine
    ) -> VoiceOutputChannel:
        return JambonzStreamOutputChannel(
            voice_websocket,
            tts_engine,
            self.tts_cache,
        )

    async def interrupt_playback(
        self, ws: Websocket, call_parameters: CallParameters
    ) -> None:
        """Interrupt the current playback of audio."""
        logger.debug("jambonz.interrupt_playback")
        await ws.send(json.dumps({"type": "killAudio"}))

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        blueprint = Blueprint("jambonz_stream", __name__)

        @blueprint.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @blueprint.route("/call_status", methods=["POST"])
        @requires_basic_auth(self.username, self.password)
        async def call_status(request: Request) -> HTTPResponse:
            """Handle call status updates from Jambonz."""
            data = request.json
            logger.debug("jambonz.call_status.received", data=data)
            return response.json({"status": "ok"})

        @blueprint.route("/webhook", methods=["POST"])
        @requires_basic_auth(self.username, self.password)
        async def webhook(request: Request) -> HTTPResponse:
            """Handle incoming webhook requests from Jambonz."""
            data = request.json
            logger.debug("jambonz.webhook.received", data=data)
            return response.json(
                [
                    {
                        "verb": "listen",
                        "url": self._websocket_stream_url(),
                        "sampleRate": 8000,
                        "passDtmf": True,
                        "bidirectionalAudio": {
                            "enabled": True,
                            "streaming": True,
                            "sampleRate": 8000,
                        },
                    }
                ]
            )

        @blueprint.websocket("/websocket", subprotocols=["audio.jambonz.org"])  # type: ignore[misc]
        async def handle_message(request: Request, ws: Websocket) -> None:
            try:
                await self.run_audio_streaming(on_new_message, ws)
            except Exception as e:
                logger.error("jambonz.handle_message.error", error=e)

        return blueprint
