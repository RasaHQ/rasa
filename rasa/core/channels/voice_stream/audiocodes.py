from __future__ import annotations

import asyncio
import base64
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
from rasa.core.channels.voice_ready.audiocodes import map_call_params
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
from rasa.shared.utils.common import mark_as_beta_feature

logger = structlog.get_logger(__name__)
PREFERRED_AUDIO_FORMAT = "raw/mulaw"


class AudiocodesVoiceOutputChannel(VoiceOutputChannel):
    @classmethod
    def name(cls) -> str:
        return "audiocodes_stream"

    def _ensure_stream_id(self) -> None:
        """Audiocodes requires a stream ID with playStream messages."""
        if "stream_id" not in call_state.channel_data:
            call_state.channel_data["stream_id"] = 0

    def _increment_stream_id(self) -> None:
        self._ensure_stream_id()
        call_state.channel_data["stream_id"] += 1

    def _get_stream_id(self) -> str:
        self._ensure_stream_id()
        return str(call_state.channel_data["stream_id"])

    def rasa_audio_bytes_to_channel_bytes(
        self, rasa_audio_bytes: RasaAudioBytes
    ) -> bytes:
        return base64.b64encode(rasa_audio_bytes)

    def channel_bytes_to_message(self, recipient_id: str, channel_bytes: bytes) -> str:
        media_message = json.dumps(
            {
                "type": "playStream.chunk",
                "streamId": self._get_stream_id(),
                "audioChunk": channel_bytes.decode("utf-8"),
            }
        )
        return media_message

    async def send_start_marker(self, recipient_id: str) -> None:
        """Send playStream.start before first audio chunk."""
        self._increment_stream_id()
        media_message = json.dumps(
            {
                "type": "playStream.start",
                "streamId": self._get_stream_id(),
                "mediaFormat": PREFERRED_AUDIO_FORMAT,
            }
        )
        logger.debug("Sending start marker", stream_id=self._get_stream_id())
        await self.voice_websocket.send(media_message)

        # This should be set when the bot actually starts speaking
        # however, Audiocodes does not have an event to indicate that.
        # This is an approximation, as the bot will be sent the audio chunks next
        # which are played to the user immediately.
        call_state.is_bot_speaking = True
        VoiceInputChannel._cancel_silence_timeout_watcher()

    async def send_intermediate_marker(self, recipient_id: str) -> None:
        """Audiocodes doesn't need intermediate markers, so do nothing."""
        pass

    async def send_end_marker(self, recipient_id: str) -> None:
        """Send playStream.stop after last audio chunk."""
        media_message = json.dumps(
            {
                "type": "playStream.stop",
                "streamId": self._get_stream_id(),
            }
        )
        logger.debug("Sending end marker", stream_id=self._get_stream_id())
        await self.voice_websocket.send(media_message)


class AudiocodesVoiceInputChannel(VoiceInputChannel):
    @classmethod
    def name(cls) -> str:
        return "audiocodes_stream"

    def __init__(
        self,
        server_url: str,
        asr_config: Dict,
        tts_config: Dict,
        interruptions: Optional[Dict[str, int]] = None,
        token: Optional[Text] = None,
    ):
        mark_as_beta_feature("Audiocodes (audiocodes_stream) Channel")
        super().__init__(
            server_url=server_url,
            asr_config=asr_config,
            tts_config=tts_config,
            interruptions=interruptions,
        )
        self.token = token

    @classmethod
    def from_credentials(
        cls,
        credentials: Optional[Dict[str, Any]],
    ) -> AudiocodesVoiceInputChannel:
        cls.validate_basic_credentials(credentials)
        new_creds = repack_voice_credentials(credentials)
        return cls(**new_creds)

    def channel_bytes_to_rasa_audio_bytes(self, input_bytes: bytes) -> RasaAudioBytes:
        return RasaAudioBytes(base64.b64decode(input_bytes))

    async def collect_call_parameters(
        self, channel_websocket: Websocket
    ) -> Optional[CallParameters]:
        async for message in channel_websocket:
            data = json.loads(message)
            if data["type"] == "session.initiate":
                # contains info about mediaformats
                logger.info(
                    "audiocodes_stream.collect_call_parameters.session.initiate",
                    data=data,
                )
                self._send_accepted(channel_websocket, data)
            elif data["type"] == "activities":
                activities = data["activities"]
                for activity in activities:
                    logger.debug(
                        "audiocodes_stream.collect_call_parameters.activity",
                        data=activity,
                    )
                    if activity["name"] == "start":
                        return map_call_params(activity["parameters"])
            elif data["type"] == "connection.validate":
                # not part of call flow; only sent when integration is created
                logger.info(
                    "audiocodes_stream.collect_call_parameters.connection.validate",
                    event_info="received request to validate integration",
                )
                self._send_validated(channel_websocket, data)
            else:
                logger.warning("audiocodes_stream.unknown_message", data=data)
        return None

    def map_input_message(
        self,
        message: Any,
        ws: Websocket,
    ) -> VoiceChannelAction:
        data = json.loads(message)
        if data["type"] == "activities":
            activities = data["activities"]
            for activity in activities:
                if activity["name"] == "start":
                    # handled in collect_call_parameters
                    pass
                elif activity["name"] == "dtmf":
                    return DTMFInputAction(digit=activity["value"])
                elif activity["name"] == "playFinished":
                    logger.debug("audiocodes_stream.playFinished", data=activity)
                    call_state.is_bot_speaking = False
                    if call_state.should_hangup:
                        logger.info("audiocodes_stream.hangup")
                        self._send_hangup(ws, data)
                        # the conversation should continue until
                        # we receive a end message from audiocodes
                else:
                    logger.warning("audiocodes_stream.unknown_activity", data=activity)
        elif data["type"] == "userStream.start":
            logger.debug("audiocodes_stream.userStream.start", data=data)
            self._send_recognition_started(ws, data)
        elif data["type"] == "userStream.chunk":
            audio_bytes = self.channel_bytes_to_rasa_audio_bytes(data["audioChunk"])
            return NewAudioAction(audio_bytes)
        elif data["type"] == "userStream.stop":
            logger.debug("audiocodes_stream.stop_recognition", data=data)
            self._send_recognition_ended(ws, data)
        elif data["type"] == "session.resume":
            logger.debug("audiocodes_stream.resume", data=data)
            self._send_accepted(ws, data)
        elif data["type"] == "session.end":
            logger.debug("audiocodes_stream.end", data=data)
            return EndConversationAction()
        else:
            logger.warning(
                "audiocodes_stream.map_input_message.unknown_message", data=data
            )

        return ContinueConversationAction()

    def _send_accepted(self, ws: Websocket, data: Dict[Text, Any]) -> None:
        supported_formats = data.get("supportedMediaFormats", [])
        preferred_format = PREFERRED_AUDIO_FORMAT

        if preferred_format not in supported_formats:
            logger.warning(
                "audiocodes_stream.format_not_supported",
                supported_formats=supported_formats,
                preferred_format=preferred_format,
            )
            raise

        payload = {
            "type": "session.accepted",
            "mediaFormat": PREFERRED_AUDIO_FORMAT,
        }
        _schedule_async_task(ws.send(json.dumps(payload)))

    def _send_recognition_started(self, ws: Websocket, data: Dict[Text, Any]) -> None:
        payload = {"type": "userStream.started"}
        _schedule_async_task(ws.send(json.dumps(payload)))

    def _send_recognition_ended(self, ws: Websocket, data: Dict[Text, Any]) -> None:
        payload = {"type": "userStream.stopped"}
        _schedule_async_task(ws.send(json.dumps(payload)))

    def _send_hypothesis(self, ws: Websocket, data: Dict[Text, Any]) -> None:
        """
        TODO: The hypothesis message is sent by the bot to provide partial
        recognition results. Using this message is recommended,
        as VAIC relies on it for performing barge-in.
        """
        pass

    def _send_recognition(self, ws: Websocket, data: Dict[Text, Any]) -> None:
        """
        TODO: The recognition message is sent by the bot to provide
        the final recognition result. Using this message is recommended
        mainly for logging purposes.
        """
        pass

    def _send_hangup(self, ws: Websocket, data: Dict[Text, Any]) -> None:
        payload = {
            "conversationId": data["conversationId"],
            "type": "activities",
            "activities": [{"type": "event", "name": "hangup"}],
        }
        _schedule_async_task(ws.send(json.dumps(payload)))

    def _send_validated(self, ws: Websocket, data: Dict[Text, Any]) -> None:
        payload = {
            "type": "connection.validated",
            "success": True,
        }
        _schedule_async_task(ws.send(json.dumps(payload)))

    def create_output_channel(
        self, voice_websocket: Websocket, tts_engine: TTSEngine
    ) -> VoiceOutputChannel:
        return AudiocodesVoiceOutputChannel(
            voice_websocket,
            tts_engine,
            self.tts_cache,
        )

    def _is_token_valid(self, token: Optional[Text]) -> bool:
        # If no token is set, always return True
        if not self.token:
            return True

        # Token is required, but not provided
        if not token:
            return False

        return hmac.compare_digest(str(self.token), str(token))

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """Defines a Sanic blueprint"""
        blueprint = Blueprint("audiocodes_stream", __name__)

        @blueprint.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @blueprint.websocket("/websocket")  # type: ignore
        async def receive(request: Request, ws: Websocket) -> None:
            if not self._is_token_valid(request.token):
                logger.error(
                    "audiocodes_stream.invalid_token",
                    invalid_token=request.token,
                )
                await ws.close(code=1008, reason="Invalid token")
                return

            logger.info(
                "audiocodes_stream.receive", event_info="Started websocket connection"
            )
            try:
                await self.run_audio_streaming(on_new_message, ws)
            except Exception as e:
                logger.exception(
                    "audiocodes_stream.receive",
                    message="Error during audio streaming",
                    error=e,
                )
                await ws.close(code=1011, reason="Error during audio streaming")
                raise

        return blueprint


def _schedule_async_task(coro: Awaitable[Any]) -> None:
    """Helper function to schedule a coroutine in the event loop.

    Args:
        coro: The coroutine to schedule
    """
    loop = asyncio.get_running_loop()
    loop.call_soon_threadsafe(lambda: loop.create_task(coro))
