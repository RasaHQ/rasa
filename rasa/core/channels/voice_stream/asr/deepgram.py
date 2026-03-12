import json
import os
from typing import Any, List, Optional
from urllib.parse import urlencode

import structlog
import websockets
import websockets.exceptions
from websockets.legacy.client import WebSocketClientProtocol

from rasa.core.channels.voice_stream.asr.asr_engine import (
    ASREngine,
    ASREngineConfig,
    ASRLanguageMapEntry,
)
from rasa.core.channels.voice_stream.asr.asr_event import (
    ASREvent,
    NewTranscript,
    UserIsSpeaking,
)
from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    AudioFormat,
    RasaAudioBytes,
)
from rasa.shared.constants import DEEPGRAM_API_KEY_ENV_VAR

logger = structlog.get_logger(__name__)

"""
Deepgram STT engine implementation.
Docs: https://developers.deepgram.com/reference/speech-to-text/listen-streaming
Media Input Settings: https://developers.deepgram.com/docs/media-input-settings
"""


class DeepgramASRConfig(ASREngineConfig):
    endpoint: Optional[str] = None
    # number of milliseconds of silence to determine end of speech
    endpointing: Optional[int] = None
    smart_format: Optional[bool] = None
    # number of milliseconds of no new transcript to determine end of speech
    # should be at least 1000 according to docs
    utterance_end_ms: Optional[int] = None


class DeepgramASR(ASREngine[DeepgramASRConfig]):
    required_env_vars = (DEEPGRAM_API_KEY_ENV_VAR,)

    @classmethod
    def name(cls) -> str:
        return "deepgram"

    def __init__(
        self,
        rasa_language: str,
        format: AudioFormat,
        config: Optional[DeepgramASRConfig] = None,
        additional_languages: Optional[List[str]] = None,
    ):
        super().__init__(rasa_language, format, config, additional_languages)
        self.accumulated_transcript = ""

    async def open_websocket_connection(self) -> WebSocketClientProtocol:
        """Connect to the ASR system."""
        deepgram_api_key = os.environ[DEEPGRAM_API_KEY_ENV_VAR]
        extra_headers = {"Authorization": f"Token {deepgram_api_key}"}
        url = self._get_api_url_with_query_params()
        try:
            return await websockets.connect(  # type: ignore
                url,
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
                url=url,
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
        if self.audio_format in (L16_24KHZ, L16_48KHZ):
            encoding = "linear16"
        elif self.audio_format == MULAW_8KHZ:
            encoding = "mulaw"
        else:
            raise ValueError(
                f"Unsupported audio format {self.audio_format} for Deepgram ASR"
            )
        query_params = {
            "encoding": encoding,
            "sample_rate": self.audio_format.sample_rate,
            "endpointing": self.config.endpointing,
            "vad_events": "true",
            "language": self.current_language_config.engine_language_key,
            "interim_results": "true",
            "model": self.current_language_config.model,
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
        return chunk.data

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
    def get_default_config(rasa_language: str) -> DeepgramASRConfig:
        return DeepgramASRConfig(
            endpoint="api.deepgram.com",
            endpointing=400,
            smart_format=True,
            utterance_end_ms=1000,
            language_map={
                rasa_language: ASRLanguageMapEntry(
                    language="en",
                    model="nova-3",
                ),
            },
        )

    @classmethod
    def from_config_dict(
        cls,
        config: Any,
        format: AudioFormat,
        rasa_language: str,
        additional_languages: Optional[List[str]] = None,
    ) -> "DeepgramASR":
        return cls(
            rasa_language=rasa_language,
            format=format,
            config=DeepgramASRConfig.model_validate(config),
            additional_languages=additional_languages,
        )

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
