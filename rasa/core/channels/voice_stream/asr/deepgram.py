from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import os

import websockets
from websockets.legacy.client import WebSocketClientProtocol

from rasa.core.channels.voice_stream.asr.asr_engine import ASREngine, ASREngineConfig
from rasa.core.channels.voice_stream.asr.asr_event import ASREvent, NewTranscript
from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes

DEEPGRAM_API_KEY = "DEEPGRAM_API_KEY"


@dataclass
class DeepgramASRConfig(ASREngineConfig):
    endpoint: Optional[str] = None
    # number of miliseconds of silence to determine end of speech
    endpointing: Optional[int] = None


class DeepgramASR(ASREngine[DeepgramASRConfig]):
    def __init__(self, config: Optional[DeepgramASRConfig] = None):
        super().__init__(config)
        self.accumulated_transcript = ""

    async def open_websocket_connection(self) -> WebSocketClientProtocol:
        """Connect to the ASR system."""
        deepgram_api_key = os.environ.get(DEEPGRAM_API_KEY)
        extra_headers = {"Authorization": f"Token {deepgram_api_key}"}
        api_url = self._get_api_url()
        query_params = self._get_query_params()
        return await websockets.connect(  # type: ignore
            api_url + query_params,
            extra_headers=extra_headers,
        )

    def _get_api_url(self) -> str:
        return f"wss://{self.config.endpoint}/v1/listen?"

    def _get_query_params(self) -> str:
        return (
            f"encoding=mulaw&sample_rate=8000&endpointing={self.config.endpointing}"
            f"&vad_events=true"
        )

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
        if data.get("is_final"):
            transcript = data["channel"]["alternatives"][0]["transcript"]
            if data.get("speech_final"):
                full_transcript = self.accumulated_transcript + transcript
                self.accumulated_transcript = ""
                return NewTranscript(full_transcript)
            else:
                self.accumulated_transcript += transcript
        return None

    @staticmethod
    def get_default_config() -> DeepgramASRConfig:
        return DeepgramASRConfig("api.deepgram.com", 400)

    @classmethod
    def from_config_dict(cls, config: Dict) -> "DeepgramASR":
        return DeepgramASR(DeepgramASRConfig.from_dict(config))
