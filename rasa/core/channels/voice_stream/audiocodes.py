import asyncio
import base64
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

logger = structlog.get_logger(__name__)


def map_call_params(data: Dict[Text, Any]) -> CallParameters:
    """Map the audiocodes stream parameters to the CallParameters dataclass."""
    return CallParameters(
        call_id=data["conversationId"],
        user_phone=data["caller"],
        # Bot phone is not available in the Audiocodes API
        direction="inbound",  # AudioCodes calls are always inbound
    )


class AudiocodesVoiceOutputChannel(VoiceOutputChannel):
    @classmethod
    def name(cls) -> str:
        return "ac_voice"

    def rasa_audio_bytes_to_channel_bytes(
        self, rasa_audio_bytes: RasaAudioBytes
    ) -> bytes:
        return base64.b64encode(rasa_audio_bytes)

    def channel_bytes_to_message(self, recipient_id: str, channel_bytes: bytes) -> str:
        media_message = json.dumps(
            {
                "type": "playStream.chunk",
                "streamId": str(call_state.stream_id),
                "audioChunk": channel_bytes.decode("utf-8"),
            }
        )
        return media_message

    async def send_start_marker(self, recipient_id: str) -> None:
        """Send playStream.start before first audio chunk."""
        call_state.stream_id += 1  # type: ignore[attr-defined]
        media_message = json.dumps(
            {
                "type": "playStream.start",
                "streamId": str(call_state.stream_id),
            }
        )
        logger.debug("Sending start marker", stream_id=call_state.stream_id)
        await self.voice_websocket.send(media_message)

    async def send_intermediate_marker(self, recipient_id: str) -> None:
        """Audiocodes doesn't need intermediate markers, so do nothing."""
        pass

    async def send_end_marker(self, recipient_id: str) -> None:
        """Send playStream.stop after last audio chunk."""
        media_message = json.dumps(
            {
                "type": "playStream.stop",
                "streamId": str(call_state.stream_id),
            }
        )
        logger.debug("Sending end marker", stream_id=call_state.stream_id)
        await self.voice_websocket.send(media_message)


class AudiocodesVoiceInputChannel(VoiceInputChannel):
    @classmethod
    def name(cls) -> str:
        return "ac_voice"

    def channel_bytes_to_rasa_audio_bytes(self, input_bytes: bytes) -> RasaAudioBytes:
        return RasaAudioBytes(base64.b64decode(input_bytes))

    async def collect_call_parameters(
        self, channel_websocket: Websocket
    ) -> Optional[CallParameters]:
        async for message in channel_websocket:
            data = json.loads(message)
            if data["type"] == "session.initiate":
                # retrieve parameters set in the webhook - contains info about the
                # caller
                logger.info("received initiate message", data=data)
                self._send_accepted(channel_websocket, data)
                return map_call_params(data)
            else:
                logger.warning("ac_voice.unknown_message", data=data)
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
                logger.debug("ac_voice.activity", data=activity)
                if activity["name"] == "start":
                    pass
                elif activity["name"] == "dtmf":
                    # TODO: handle DTMF input
                    pass
                elif activity["name"] == "playFinished":
                    logger.debug("ac_voice.playFinished", data=activity)
                    if call_state.should_hangup:
                        logger.info("audiocodes.hangup")
                        self._send_hangup(ws, data)
                        # the conversation should continue until
                        # we receive a end message from audiocodes
                    pass
                else:
                    logger.warning("ac_voice.unknown_activity", data=activity)
        elif data["type"] == "userStream.start":
            logger.debug("ac_voice.userStream.start", data=data)
            self._send_recognition_started(ws, data)
        elif data["type"] == "userStream.chunk":
            audio_bytes = self.channel_bytes_to_rasa_audio_bytes(data["audioChunk"])
            return NewAudioAction(audio_bytes)
        elif data["type"] == "userStream.stop":
            logger.debug("ac_voice.stop_recognition", data=data)
            self._send_recognition_ended(ws, data)
        elif data["type"] == "session.resume":
            logger.debug("ac_voice.resume", data=data)
            self._send_accepted(ws, data)
        elif data["type"] == "session.end":
            logger.debug("ac_voice.end", data=data)
            return EndConversationAction()
        elif data["type"] == "connection.validate":
            # not part of call flow; only sent when integration is created
            self._send_validated(ws, data)
        else:
            logger.warning("ac_voice.unknown_message", data=data)

        return ContinueConversationAction()

    def _send_accepted(self, ws: Websocket, data: Dict[Text, Any]) -> None:
        supported_formats = data.get("supportedMediaFormats", [])
        preferred_format = "raw/mulaw"

        if preferred_format not in supported_formats:
            logger.warning(
                "ac_voice.format_not_supported",
                supported_formats=supported_formats,
                preferred_format=preferred_format,
            )
            raise

        payload = {
            "type": "session.accepted",
            "mediaFormat": "raw/mulaw",
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

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """Defines a Sanic bluelogger.debug."""
        blueprint = Blueprint("ac_voice", __name__)

        @blueprint.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @blueprint.websocket("/websocket")  # type: ignore
        async def receive(request: Request, ws: Websocket) -> None:
            # TODO: validate API key header
            logger.info("audiocodes.receive", message="Starting audio streaming")
            try:
                await self.run_audio_streaming(on_new_message, ws)
            except Exception as e:
                logger.exception(
                    "audiocodes.receive",
                    message="Error during audio streaming",
                    error=e,
                )
                # return 500 error
                raise

        return blueprint


def _schedule_async_task(coro: Awaitable[Any]) -> None:
    """Helper function to schedule a coroutine in the event loop.

    Args:
        coro: The coroutine to schedule
    """
    loop = asyncio.get_running_loop()
    loop.call_soon_threadsafe(lambda: loop.create_task(coro))
