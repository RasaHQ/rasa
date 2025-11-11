import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import structlog
import websockets
import websockets.exceptions
from websockets.legacy.client import WebSocketClientProtocol

from rasa.core.channels.voice_stream.asr.asr_engine import ASREngine, ASREngineConfig
from rasa.core.channels.voice_stream.asr.asr_event import (
    ASREvent,
    NewTranscript,
    UserIsSpeaking,
)
from rasa.core.channels.voice_stream.audio_bytes import HERTZ, RasaAudioBytes
from rasa.shared.constants import DEEPGRAM_API_KEY_ENV_VAR

logger = structlog.get_logger(__name__)


@dataclass
class DeepgramASRConfig(ASREngineConfig):
    endpoint: Optional[str] = None
    # number of milliseconds of silence to determine end of speech
    endpointing: Optional[int] = None
    language: Optional[str] = None
    model: Optional[str] = None
    smart_format: Optional[bool] = None
    # number of milliseconds of no new transcript to determine end of speech
    # should be at least 1000 according to docs
    utterance_end_ms: Optional[int] = None


class DeepgramASR(ASREngine[DeepgramASRConfig]):
    required_env_vars = (DEEPGRAM_API_KEY_ENV_VAR,)

    def __init__(self, config: Optional[DeepgramASRConfig] = None):
        super().__init__(config)
        self.accumulated_transcript = ""

    async def open_websocket_connection(self) -> WebSocketClientProtocol:
        """Connect to the ASR system."""
        deepgram_api_key = os.environ[DEEPGRAM_API_KEY_ENV_VAR]
        extra_headers = {"Authorization": f"Token {deepgram_api_key}"}
        try:
            return await websockets.connect(  # type: ignore
                self._get_api_url_with_query_params(),
                extra_headers=extra_headers,
            )
        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 401:
                error_msg = "Please make sure your Deepgram API key is correct."
            else:
                error_msg = "Connection to Deepgram failed."
            logger.error(
                "deepgram.connection.failed",
                status_code=e.status_code,
                error=error_msg,
            )
            raise

    def _get_api_url_with_query_params(self) -> str:
        """Combine api url and query params."""
        return self._get_api_url() + self._get_query_params()

    def _get_api_url(self) -> str:
        """Get the api url with the configured endpoint."""
        return f"wss://{self.config.endpoint}/v1/listen?"

    def _get_query_params(self) -> str:
        """Get the configured query parameters for the api."""
        query_params = {
            "encoding": "mulaw",
            "sample_rate": HERTZ,
            "endpointing": self.config.endpointing,
            "vad_events": "true",
            "language": self.config.language,
            "interim_results": "true",
            "model": self.config.model,
            "smart_format": str(self.config.smart_format).lower(),
        }
        if self.config.utterance_end_ms and self.config.utterance_end_ms > 0:
            query_params["utterance_end_ms"] = self.config.utterance_end_ms

        return urlencode(query_params)

    async def signal_audio_done(self) -> None:
        """Signal to the ASR Api that you are done sending data."""
        if self.asr_socket is None:
            raise AttributeError("Websocket not connected.")
        await self.asr_socket.send(json.dumps({"type": "CloseStream"}))

    def rasa_audio_bytes_to_engine_bytes(self, chunk: RasaAudioBytes) -> bytes:
        """Convert RasaAudioBytes to bytes usable by this engine."""
        return chunk

    def engine_event_to_asr_event(self, e: Any) -> Optional[ASREvent]:
        """Translate an engine event to a common ASREvent."""
        data = json.loads(e)
        data_type = data["type"]
        if data_type == "Results":
            transcript_data = data["channel"]["alternatives"][0]
            transcript = transcript_data["transcript"]
            if data["is_final"]:
                if data.get("speech_final"):
                    full_transcript = self.concatenate_transcripts(
                        self.accumulated_transcript, transcript
                    )
                    self.accumulated_transcript = ""
                    if full_transcript:
                        return NewTranscript(full_transcript)
                else:
                    self.accumulated_transcript = self.concatenate_transcripts(
                        self.accumulated_transcript, transcript
                    )
            elif transcript:
                return UserIsSpeaking(transcript)
        # event that comes after utterance_end_ms of no new transcript
        elif data_type == "UtteranceEnd":
            if self.accumulated_transcript:
                transcript = self.accumulated_transcript
                self.accumulated_transcript = ""
                return NewTranscript(transcript)
        return None

    @staticmethod
    def get_default_config() -> DeepgramASRConfig:
        return DeepgramASRConfig(
            endpoint="api.deepgram.com",
            endpointing=400,
            language="en",
            model="nova-2-general",
            smart_format=True,
            utterance_end_ms=1000,
        )

    @classmethod
    def from_config_dict(cls, config: Dict) -> "DeepgramASR":
        return DeepgramASR(DeepgramASRConfig.from_dict(config))

    @staticmethod
    def concatenate_transcripts(t1: str, t2: str) -> str:
        """Concatenate two transcripts making sure there is a space between them."""
        return (t1.strip() + " " + t2.strip()).strip()

    async def send_keep_alive(self) -> None:
        """Send a keep-alive message to the Deepgram websocket connection."""
        if self.asr_socket is None:
            return

        if self.asr_socket.open:
            await self.asr_socket.send(json.dumps({"type": "KeepAlive"}))
