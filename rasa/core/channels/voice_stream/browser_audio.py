from __future__ import annotations

import audioop
import base64
import json
import os
import uuid
import wave
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

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
from rasa.core.channels.voice_stream.util import repack_voice_credentials
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
        marker_data = {"marker": message_id}

        # Include comprehensive latency information if available
        latency_data = {
            "asr_latency_ms": call_state.asr_latency_ms,
            "rasa_processing_latency_ms": call_state.rasa_processing_latency_ms,
            "tts_first_byte_latency_ms": call_state.tts_first_byte_latency_ms,
            "tts_complete_latency_ms": call_state.tts_complete_latency_ms,
        }

        # Filter out None values from latency data
        latency_data = {k: v for k, v in latency_data.items() if v is not None}

        # Add latency data to marker if any metrics are available
        if latency_data:
            marker_data["latency"] = latency_data  # type: ignore[assignment]

        return json.dumps(marker_data), message_id


class BrowserAudioInputChannel(VoiceInputChannel):
    requires_voice_license = False

    def __init__(
        self,
        server_url: str,
        asr_config: Dict[str, Any],
        tts_config: Dict[str, Any],
        recording: bool = False,
        interruptions: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initializes the browser audio input channel."""
        super().__init__(server_url, asr_config, tts_config, interruptions)

        # For debugging, recording of user audio might be useful
        # to identify audio quality issues or transcription errors
        self._recording_enabled = recording
        self._wav_file: Optional[wave.Wave_write] = None

    def _start_recording(self, call_id: str, user_id: str) -> None:
        os.makedirs("recordings", exist_ok=True)
        filename = f"{user_id}_{call_id}.wav"
        file_path = os.path.join("recordings", filename)

        if not self._recording_enabled:
            return

        self._wav_file = wave.open(file_path, "wb")
        self._wav_file.setnchannels(1)  # Mono audio
        self._wav_file.setsampwidth(4)  # 32-bit audio (4 bytes)
        self._wav_file.setframerate(8000)  # 8kHz sample rate
        logger.info("voice_channel.user_audio_recording.started", file_path=file_path)

    def _append_audio_to_recording(self, audio_bytes: bytes) -> None:
        if self._wav_file and self._recording_enabled:
            self._wav_file.writeframes(audio_bytes)

    def _stop_recording(self) -> None:
        """Close the recording file if it's open."""
        if self._wav_file:
            self._wav_file.close()
            self._wav_file = None
            logger.debug("voice_channel.user_audio_recording.stopped")

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

    @classmethod
    def from_credentials(
        cls,
        credentials: Optional[Dict[str, Any]],
    ) -> BrowserAudioInputChannel:
        cls.validate_basic_credentials(credentials)
        new_creds = repack_voice_credentials(credentials or {})
        return cls(**new_creds)

    def map_input_message(
        self,
        message: Any,
        ws: Websocket,
    ) -> VoiceChannelAction:
        data = json.loads(message)
        if "audio" in data:
            channel_bytes = base64.b64decode(data["audio"])
            self._append_audio_to_recording(channel_bytes)
            audio_bytes = self.channel_bytes_to_rasa_audio_bytes(channel_bytes)
            return NewAudioAction(audio_bytes)
        elif "marker" in data:
            if data["marker"] == call_state.latest_bot_audio_id:
                # Just finished streaming last audio bytes
                call_state.is_bot_speaking = False
                if call_state.should_hangup:
                    logger.debug(
                        "browser_audio.hangup", marker=call_state.latest_bot_audio_id
                    )
                    return EndConversationAction()
            else:
                call_state.is_bot_speaking = True
        return ContinueConversationAction()

    async def interrupt_playback(
        self, ws: Websocket, call_parameters: CallParameters
    ) -> None:
        """Interrupt the current playback of audio."""
        logger.debug("browser_audio.interrupt_playback")
        await ws.send(json.dumps({"interruptPlayback": True}))

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
                call_parameters = await self.collect_call_parameters(ws)
                if call_parameters and call_parameters.call_id:
                    self._start_recording(call_parameters.call_id, "local")
                await self.run_audio_streaming(on_new_message, ws)
            except Exception as e:
                logger.error("browser_audio.handle_message.error", error=e)
            finally:
                self._stop_recording()

        return blueprint
