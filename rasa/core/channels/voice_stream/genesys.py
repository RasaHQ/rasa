import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, Optional, Text

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
from rasa.core.channels.voice_stream.call_state import (
    call_state,
)
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine
from rasa.core.channels.voice_stream.voice_channel import (
    ContinueConversationAction,
    EndConversationAction,
    NewAudioAction,
    VoiceChannelAction,
    VoiceInputChannel,
    VoiceOutputChannel,
)

"""
Genesys throws a rate limit error with too many audio messages.
To avoid this, we buffer the audio messages and send them in chunks.

- global.inbound.binary.average.rate.per.second: 5
The allowed average rate per second of inbound binary data

- global.inbound.binary.max: 25
The maximum number of inbound binary data messages
that can be sent instantaneously

https://developer.genesys.cloud/organization/organization/limits#audiohook

The maximum binary message size is not mentioned
in the documentation but observed in their example app
https://github.com/GenesysCloudBlueprints/audioconnector-server-reference-implementation
"""
MAXIMUM_BINARY_MESSAGE_SIZE = 64000  # 64KB
logger = structlog.get_logger(__name__)


def map_call_params(data: Dict[Text, Any]) -> CallParameters:
    """Map the twilio stream parameters to the CallParameters dataclass."""
    parameters = data["parameters"]
    participant = parameters["participant"]
    # sent as {"ani": "tel:+491604697810"}
    ani = participant.get("ani", "")
    user_phone = ani.split(":")[-1] if ani else ""

    return CallParameters(
        call_id=parameters.get("conversationId", ""),
        user_phone=user_phone,
        bot_phone=participant.get("dnis", ""),
    )


class GenesysOutputChannel(VoiceOutputChannel):
    @classmethod
    def name(cls) -> str:
        return "genesys"

    async def send_audio_bytes(
        self, recipient_id: str, audio_bytes: RasaAudioBytes
    ) -> None:
        await self.voice_websocket.send(audio_bytes)

    async def send_marker_message(self, recipient_id: str) -> None:
        """
        Send a message that marks positions in the audio stream.
        Genesys does not support this feature, so we do nothing here.
        """
        pass


class GenesysInputChannel(VoiceInputChannel):
    @classmethod
    def name(cls) -> str:
        return "genesys"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _ensure_channel_data_initialized(self) -> None:
        """Initialize Genesys-specific channel data if not already present.

        Genesys requires the server and client each maintain a
        monotonically increasing message sequence number.
        """
        if "server_sequence_number" not in call_state.channel_data:
            call_state.channel_data["server_sequence_number"] = 0
        if "client_sequence_number" not in call_state.channel_data:
            call_state.channel_data["client_sequence_number"] = 0

    def _get_next_sequence(self) -> int:
        """
        Get the next message sequence number
        Rasa == Server
        Genesys == Client

        Genesys requires the server and client each maintain a
        monotonically increasing message sequence number.
        """
        self._ensure_channel_data_initialized()
        call_state.channel_data["server_sequence_number"] += 1
        return call_state.channel_data["server_sequence_number"]

    def _get_last_client_sequence(self) -> int:
        """Get the last client(Genesys) sequence number."""
        self._ensure_channel_data_initialized()
        return call_state.channel_data["client_sequence_number"]

    def _update_client_sequence(self, seq: int) -> None:
        """Update the client(Genesys) sequence number."""
        self._ensure_channel_data_initialized()

        if seq - call_state.channel_data["client_sequence_number"] != 1:
            logger.warning(
                "genesys.update_client_sequence.sequence_gap",
                received_seq=seq,
                last_seq=call_state.channel_data["client_sequence_number"],
            )
        call_state.channel_data["client_sequence_number"] = seq

    def channel_bytes_to_rasa_audio_bytes(self, input_bytes: bytes) -> RasaAudioBytes:
        return RasaAudioBytes(input_bytes)

    async def collect_call_parameters(
        self, channel_websocket: Websocket
    ) -> Optional[CallParameters]:
        """Call Parameters are collected during the open event."""
        async for message in channel_websocket:
            data = json.loads(message)
            self._update_client_sequence(data["seq"])
            if data.get("type") == "open":
                call_params = await self.handle_open(channel_websocket, data)
                return call_params
            else:
                logger.error("genesys.receive.unexpected_initial_message", message=data)

        return None

    def map_input_message(
        self,
        message: Any,
        ws: Websocket,
    ) -> VoiceChannelAction:
        # if message is binary, it's audio
        if isinstance(message, bytes):
            return NewAudioAction(self.channel_bytes_to_rasa_audio_bytes(message))
        else:
            # process text message
            data = json.loads(message)
            self._update_client_sequence(data["seq"])
            msg_type = data.get("type")
            if msg_type == "close":
                logger.info("genesys.handle_close", message=data)
                self.handle_close(ws, data)
                return EndConversationAction()
            elif msg_type == "ping":
                logger.info("genesys.handle_ping", message=data)
                self.handle_ping(ws, data)
            elif msg_type == "playback_started":
                logger.debug("genesys.handle_playback_started", message=data)
                call_state.is_bot_speaking = True  # type: ignore[attr-defined]
            elif msg_type == "playback_completed":
                logger.debug("genesys.handle_playback_completed", message=data)
                call_state.is_bot_speaking = False  # type: ignore[attr-defined]
                if call_state.should_hangup:
                    logger.info("genesys.hangup")
                    self.disconnect(ws, data)
                    # the conversation should continue until
                    # we receive a close message from Genesys
            elif msg_type == "dtmf":
                logger.info("genesys.handle_dtmf", message=data)
            elif msg_type == "error":
                logger.warning("genesys.handle_error", message=data)
            else:
                logger.warning("genesys.map_input_message.unknown_type", message=data)

        return ContinueConversationAction()

    def create_output_channel(
        self, voice_websocket: Websocket, tts_engine: TTSEngine
    ) -> VoiceOutputChannel:
        return GenesysOutputChannel(
            voice_websocket,
            tts_engine,
            self.tts_cache,
            min_buffer_size=MAXIMUM_BINARY_MESSAGE_SIZE // 2,
        )

    async def handle_open(self, ws: Websocket, message: dict) -> CallParameters:
        """Handle initial open transaction from Genesys."""
        call_parameters = map_call_params(message)
        params = message["parameters"]
        media_options = params.get("media", [])

        # Send opened response
        if media_options:
            logger.info("genesys.handle_open", media_parameter=media_options[0])
            response = {
                "version": "2",
                "type": "opened",
                "seq": self._get_next_sequence(),
                "clientseq": self._get_last_client_sequence(),
                "id": message.get("id"),
                "parameters": {"startPaused": False, "media": [media_options[0]]},
            }
            logger.debug("genesys.handle_open.opened", response=response)
            await ws.send(json.dumps(response))
        else:
            logger.warning(
                "genesys.handle_open.no_media_formats", client_message=message
            )
        return call_parameters

    def handle_ping(self, ws: Websocket, message: dict) -> None:
        """Handle ping message from Genesys."""
        response = {
            "version": "2",
            "type": "pong",
            "seq": self._get_next_sequence(),
            "clientseq": message.get("seq"),
            "id": message.get("id"),
            "parameters": {},
        }
        logger.debug("genesys.handle_ping.pong", response=response)
        _schedule_ws_task(ws.send(json.dumps(response)))

    def handle_close(self, ws: Websocket, message: dict) -> None:
        """Handle close message from Genesys."""
        response = {
            "version": "2",
            "type": "closed",
            "seq": self._get_next_sequence(),
            "clientseq": self._get_last_client_sequence(),
            "id": message.get("id"),
            "parameters": message.get("parameters", {}),
        }
        logger.debug("genesys.handle_close.closed", response=response)

        _schedule_ws_task(ws.send(json.dumps(response)))

    def disconnect(self, ws: Websocket, data: dict) -> None:
        """
        Send disconnect message to Genesys.

        https://developer.genesys.cloud/devapps/audiohook/protocol-reference#disconnect
        It should be used to hangup the call.
        Genesys will respond with a "close" message to us
        that is handled by the handle_close method.
        """
        message = {
            "version": "2",
            "type": "disconnect",
            "seq": self._get_next_sequence(),
            "clientseq": self._get_last_client_sequence(),
            "id": data.get("id"),
            "parameters": {
                "reason": "completed",
                # arbitrary values can be sent here
            },
        }
        logger.debug("genesys.disconnect", message=message)
        _schedule_ws_task(ws.send(json.dumps(message)))

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """Defines a Sanic blueprint for the voice input channel."""
        blueprint = Blueprint("genesys", __name__)

        @blueprint.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @blueprint.websocket("/websocket")  # type: ignore[misc]
        async def receive(request: Request, ws: Websocket) -> None:
            logger.debug(
                "genesys.receive",
                audiohook_session_id=request.headers.get("audiohook-session-id"),
            )
            # validate required headers
            required_headers = [
                "audiohook-organization-id",
                "audiohook-correlation-id",
                "audiohook-session-id",
                "x-api-key",
            ]

            for header in required_headers:
                if header not in request.headers:
                    await ws.close(1008, f"Missing required header: {header}")
                    return

            # TODO: validate API key header
            # process audio streaming
            logger.info("genesys.receive", message="Starting audio streaming")
            await self.run_audio_streaming(on_new_message, ws)

        return blueprint


def _schedule_ws_task(coro: Awaitable[Any]) -> None:
    """Helper function to schedule a coroutine in the event loop.

    Args:
        coro: The coroutine to schedule
    """
    loop = asyncio.get_running_loop()
    loop.call_soon_threadsafe(lambda: loop.create_task(coro))
