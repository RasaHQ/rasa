from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
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
from rasa.shared.exceptions import InvalidConfigException

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
HEADER_API_KEY = "X-Api-Key"
logger = structlog.get_logger(__name__)


def map_call_params(data: Dict[Text, Any]) -> CallParameters:
    """Map the Genesys parameters to the CallParameters dataclass."""
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

    def __init__(
        self,
        server_url: str,
        asr_config: Dict,
        tts_config: Dict,
        interruptions: Optional[Dict[str, int]] = None,
        api_key: Optional[Text] = None,
        client_secret: Optional[Text] = None,
    ) -> None:
        super().__init__(server_url, asr_config, tts_config, interruptions)
        self.api_key = api_key
        self.client_secret = client_secret

    @classmethod
    def from_credentials(
        cls,
        credentials: Optional[Dict[str, Any]],
    ) -> GenesysInputChannel:
        """Create a channel from credentials dictionary.

        Args:
            credentials: Dictionary containing the required credentials:
                - server_url: URL where the server is hosted
                - asr: ASR engine configuration
                - tts: TTS engine configuration
                - api_key: Required API key for Genesys authentication
                - client_secret: Optional client secret for signature verification

        Returns:
            GenesysInputChannel instance
        """
        cls.validate_credentials(credentials)
        new_creds = repack_voice_credentials(credentials)
        return cls(**new_creds)

    @classmethod
    def validate_credentials(cls, credentials: Optional[Dict[str, Any]]) -> None:
        """Validate the credentials for the Genesys voice channel."""
        cls.validate_basic_credentials(credentials)
        if not credentials.get("api_key"):  # type: ignore[union-attr]
            raise InvalidConfigException(
                "No API key given for Genesys voice channel (api_key)."
            )

    @staticmethod
    def _ensure_channel_data_initialized() -> None:
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
                call_state.is_bot_speaking = True
            elif msg_type == "playback_completed":
                logger.debug("genesys.handle_playback_completed", message=data)
                call_state.is_bot_speaking = False
                if call_state.should_hangup:
                    logger.info("genesys.hangup")
                    self.disconnect(ws, data)
                    # the conversation should continue until
                    # we receive a close message from Genesys
            elif msg_type == "dtmf":
                return DTMFInputAction(digit=data["parameters"]["digit"])
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
        response: Dict[str, Any] = {
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

    def _calculate_signature(self, request: Request) -> str:
        """Calculate the signature using request data."""
        org_id = request.headers.get("Audiohook-Organization-Id")
        session_id = request.headers.get("Audiohook-Session-Id")
        correlation_id = request.headers.get("Audiohook-Correlation-Id")
        api_key = request.headers.get(HEADER_API_KEY)

        # order of components is important!
        components = [
            ("@request-target", "/webhooks/genesys/websocket"),
            ("audiohook-session-id", session_id),
            ("audiohook-organization-id", org_id),
            ("audiohook-correlation-id", correlation_id),
            (HEADER_API_KEY.lower(), api_key),
            ("@authority", self.server_url),
        ]

        # Create signature base string
        signing_string = ""
        for name, value in components:
            signing_string += f'"{name}": {value}\n'

        # Add @signature-params
        signature_input = request.headers["Signature-Input"]
        _, params_str = signature_input.split("=", 1)
        signing_string += f'"@signature-params": {params_str}'

        # Calculate the HMAC signature
        key_bytes = base64.b64decode(self.client_secret)
        signature = hmac.new(
            key_bytes, signing_string.encode("utf-8"), hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode("utf-8")

    async def _verify_signature(self, request: Request) -> bool:
        """Verify the HTTP message signature from Genesys."""
        if not self.client_secret:
            logger.info(
                "genesys.verify_signature.no_client_secret",
                event_info="Signature verification skipped",
            )
            return True  # Skip verification if no client secret

        signature = request.headers.get("Signature")
        signature_input = request.headers.get("Signature-Input")
        if not signature or not signature_input:
            logger.error("genesys.signature.missing_signature_header")
            return False

        try:
            actual_signature = signature.split("=", 1)[1].strip(':"')
            expected_signature = self._calculate_signature(request)
            return hmac.compare_digest(
                expected_signature.encode("utf-8"), actual_signature.encode("utf-8")
            )
        except Exception as e:
            logger.exception("genesys.signature.verification_error", error=e)
            return False

    def _ensure_required_headers(self, request: Request) -> bool:
        """Ensure required headers are present in the request."""
        required_headers = [
            "Audiohook-Organization-Id",
            "Audiohook-Correlation-Id",
            "Audiohook-Session-Id",
            HEADER_API_KEY,
        ]

        missing_headers = [
            header for header in required_headers if header not in request.headers
        ]

        if missing_headers:
            logger.error(
                "genesys.missing_required_headers",
                missing_headers=missing_headers,
            )
            return False
        return True

    def _ensure_api_key(self, request: Request) -> bool:
        """Ensure the API key is present in the request."""
        api_key = request.headers.get(HEADER_API_KEY)
        if not hmac.compare_digest(str(self.api_key), str(api_key)):
            return False
        return True

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

            # verify signature
            if not await self._verify_signature(request):
                logger.error("genesys.receive.invalid_signature")
                await ws.close(code=1008, reason="Invalid signature")
                return

            # ensure required headers are present
            if not self._ensure_required_headers(request):
                await ws.close(code=1002, reason="Missing required headers")
                return

            # ensure API key is correct
            if not self._ensure_api_key(request):
                logger.error(
                    "genesys.receive.invalid_api_key",
                    invalid_api_key=request.headers.get(HEADER_API_KEY),
                )
                await ws.close(code=1008, reason="Invalid API key")
                return

            # process audio streaming
            logger.info("genesys.receive", message="Starting audio streaming")
            try:
                await self.run_audio_streaming(on_new_message, ws)
            except Exception as e:
                logger.exception(
                    "genesys.receive",
                    message="Error during audio streaming",
                    error=e,
                )
                await ws.close(code=1011, reason="Error during audio streaming")
                raise

        return blueprint


def _schedule_ws_task(coro: Awaitable[Any]) -> None:
    """Helper function to schedule a coroutine in the event loop.

    Args:
        coro: The coroutine to schedule
    """
    loop = asyncio.get_running_loop()
    loop.call_soon_threadsafe(lambda: loop.create_task(coro))
