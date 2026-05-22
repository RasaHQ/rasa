from __future__ import annotations

import audioop
import base64
import json
import uuid
from typing import Any, Dict, Optional, Union

import structlog
from pydantic import BaseModel
from sanic import (  # type: ignore[attr-defined]
    Blueprint,
    HTTPResponse,
    Request,
    Websocket,
    response,
)

from rasa.core.channels.channel import RuntimeAgent
from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.audio_debugging import _save_rasa_bytes_to_wav
from rasa.core.channels.voice_stream.call_state import (
    BotIsSpeaking,
    BotStoppedSpeaking,
    Marker,
    MarkerType,
    StepType,
    call_state,
)
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine
from rasa.core.channels.voice_stream.util import repack_voice_credentials
from rasa.core.channels.voice_stream.voice_channel import (
    ContinueConversationAction,
    EndConversationAction,
    MarkerInput,
    MarkerMessageOutput,
    NewAudioAction,
    VoiceChannelAction,
    VoiceInputChannel,
    VoiceOutputChannel,
)

logger = structlog.get_logger()
DEFAULT_SAMPLE_RATE = 48000
_SAMPLE_RATE_TO_FORMAT = {
    8000: MULAW_8KHZ,
    24000: L16_24KHZ,
    48000: L16_48KHZ,
}


class LatencyOutput(BaseModel):
    asr_latency_ms: Optional[float]
    rasa_processing_latency_ms: Optional[float]
    tts_first_byte_latency_ms: Optional[float]
    tts_complete_latency_ms: Optional[float]

    def is_empty(self) -> bool:
        return (
            self.asr_latency_ms is None
            or self.rasa_processing_latency_ms is None
            or self.tts_first_byte_latency_ms is None
            or self.tts_complete_latency_ms is None
        )


class MarkerOutput(BaseModel):
    marker: str
    marker_type: Optional[MarkerType] = None
    step_type: Optional[StepType] = None
    latency: Optional[LatencyOutput] = None

    def serialize_str(self) -> str:
        return self.model_dump_json(
            exclude={"latency"} if not self.latency or self.latency.is_empty() else None
        )

    def to_marker(self) -> Marker:
        return Marker(
            marker_id=self.marker,
            marker_type=self.marker_type,
            step_type=self.step_type,
        )


class BrowserAudioOutputChannel(VoiceOutputChannel):
    @classmethod
    def name(cls) -> str:
        return "browser_audio"

    def rasa_audio_bytes_to_channel_bytes(
        self, rasa_audio_bytes: RasaAudioBytes
    ) -> bytes:
        if self.audio_format == MULAW_8KHZ:
            # Transcode from L16 8kHz to Mulaw 8-bit 8kHz
            return audioop.ulaw2lin(rasa_audio_bytes.data, 2)
        elif self.audio_format in (L16_24KHZ, L16_48KHZ):
            return rasa_audio_bytes.data
        else:
            raise ValueError(f"Unsupported audio format: {self.audio_format}")

    def channel_bytes_to_message(self, recipient_id: str, channel_bytes: bytes) -> str:
        return json.dumps({"audio": base64.b64encode(channel_bytes).decode("utf-8")})

    def create_marker_message(self, marker_input: MarkerInput) -> MarkerMessageOutput:
        marker_output = MarkerOutput(
            marker=uuid.uuid4().hex,
            marker_type=marker_input.marker_type,
            step_type=marker_input.step_type,
            latency=LatencyOutput(
                asr_latency_ms=call_state.asr_latency_ms,
                rasa_processing_latency_ms=call_state.rasa_processing_latency_ms,
                tts_first_byte_latency_ms=call_state.tts_first_byte_latency_ms,
                tts_complete_latency_ms=call_state.tts_complete_latency_ms,
            ),
        )

        if marker_input.marker_type in {MarkerType.START, MarkerType.END}:
            call_state.set_marker(marker_output.to_marker())

        return MarkerMessageOutput(
            message_id=marker_output.marker,
            message=marker_output.serialize_str(),
        )


class BrowserAudioInputChannel(VoiceInputChannel):
    requires_voice_license = False

    def __init__(
        self,
        server_url: str,
        asr_config: Dict[str, Any],
        tts_config: Dict[str, Any],
        recording: bool = False,
        interruptions: Optional[Dict[str, int]] = None,
        silence_timeout: Optional[Union[float, int]] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        """Initializes the browser audio input channel.

        Args:
            server_url: The URL of the Rasa server.
            asr_config: Configuration for the ASR engine.
            tts_config: Configuration for the TTS engine.
            recording: Whether to record user audio for debugging.
            interruptions: Configuration for interruption handling.
            sample_rate: Optional sample rate for the audio.
        """
        super().__init__(server_url, asr_config, tts_config, interruptions)
        self.audio_format = _SAMPLE_RATE_TO_FORMAT[sample_rate]
        self.silence_timeout = silence_timeout

        # For debugging, recording of user audio might be useful
        # to identify audio quality issues or transcription errors
        self._recording_enabled = recording

    def _start_recording(self) -> None:
        if self._recording_enabled:
            self.collected_bytes = RasaAudioBytes(b"", format=self.audio_format)

    def _accumulate_bytes(self, rasa_bytes: RasaAudioBytes) -> None:
        if self._recording_enabled:
            self.collected_bytes += rasa_bytes

    def _stop_recording_and_save_to_wav_file(self) -> None:
        """Save the collected audio bytes to a WAV file for debugging purposes."""
        if self._recording_enabled:
            _save_rasa_bytes_to_wav(self.collected_bytes, "user_audio_recordings")
            logger.info("voice_channel.user_audio_recording.stopped")

    @classmethod
    def name(cls) -> str:
        return "browser_audio"

    def channel_bytes_to_rasa_audio_bytes(self, input_bytes: bytes) -> RasaAudioBytes:
        if self.audio_format == MULAW_8KHZ:
            # Transcode from Mulaw 8-bit 8kHz to L16 8kHz
            transcoded_bytes = audioop.lin2ulaw(input_bytes, 2)
        elif self.audio_format in (L16_24KHZ, L16_48KHZ):
            transcoded_bytes = input_bytes
        else:
            raise ValueError(f"Unsupported audio format: {self.audio_format}")

        return RasaAudioBytes(transcoded_bytes, format=self.audio_format)

    async def collect_call_parameters(
        self,
        channel_websocket: Websocket,
        request: Optional[Request] = None,
    ) -> Optional[CallParameters]:
        call_id = f"inspect-{uuid.uuid4()}"
        self._start_recording()

        language: Optional[str] = None
        if request is not None:
            language = request.args.get("language") or None

        # Channel sends/receives L16 Audio at different sample rates
        # Even Mulaw is sent as L16 8kHz
        await channel_websocket.send(
            json.dumps(
                {
                    "type": "handshake",
                    "sample_rate": self.audio_format.sample_rate,
                }
            )
        )
        logger.info(
            "browser_audio.handshake_sent",
            call_id=call_id,
            sample_rate=self.audio_format.sample_rate,
            audio_format=self.audio_format,
        )
        return CallParameters(
            call_id, "local", "local", stream_id=call_id, language=language
        )

    @classmethod
    def from_credentials(
        cls,
        credentials: Optional[Dict[str, Any]],
    ) -> BrowserAudioInputChannel:
        cls.validate_basic_credentials(credentials)
        new_creds = repack_voice_credentials(credentials or {})
        if (
            new_creds.get("sample_rate") is not None
            and new_creds.get("sample_rate") not in _SAMPLE_RATE_TO_FORMAT
        ):
            raise ValueError(
                f"Unsupported sample rate: {new_creds.get('sample_rate')}. "
                f"Supported rates are: {list(_SAMPLE_RATE_TO_FORMAT.keys())}"
            )
        return cls(**new_creds)

    async def map_input_message(
        self,
        message: Any,
        ws: Websocket,
    ) -> VoiceChannelAction:
        data = json.loads(message)
        if "audio" in data:
            channel_bytes = base64.b64decode(data["audio"])
            audio_bytes = self.channel_bytes_to_rasa_audio_bytes(channel_bytes)
            self._accumulate_bytes(audio_bytes)
            return NewAudioAction(audio_bytes)
        elif "marker" in data:
            marker_message = MarkerOutput.model_validate(data)

            marker = call_state.get_marker(marker_message.marker)

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
                            logger.debug(
                                "browser_audio.hangup",
                                marker=marker,
                            )
                            return EndConversationAction()
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
            voice_websocket=voice_websocket,
            audio_format=self.audio_format,
            tts_engine=tts_engine,
            tts_cache=self.tts_cache,
        )

    def conversation_blueprint(
        self,
        agent: RuntimeAgent,
    ) -> Blueprint:
        """Defines a Sanic blueprint."""
        blueprint = Blueprint("browser_audio", __name__)

        @blueprint.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @blueprint.route("/supported_languages", methods=["GET"])
        async def supported_languages(_: Request) -> HTTPResponse:
            """Return supported languages from the loaded model for voice UI."""
            model_metadata = agent.model_metadata
            language = model_metadata.language if model_metadata else "en"
            additional_languages = (
                model_metadata.additional_languages if model_metadata else []
            )
            return response.json(
                {"languages": [language or "en"] + (additional_languages or [])}
            )

        @blueprint.websocket("/websocket")  # type: ignore
        async def handle_message(request: Request, ws: Websocket) -> None:
            try:
                await self.run_audio_streaming(agent, ws, request=request)
            except Exception as e:
                logger.error("browser_audio.handle_message.error", error=e)
            finally:
                self._stop_recording_and_save_to_wav_file()

        return blueprint
