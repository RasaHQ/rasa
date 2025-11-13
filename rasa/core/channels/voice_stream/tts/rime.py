import base64
import os
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional
from urllib.parse import urlencode

import aiohttp
import structlog
from aiohttp import ClientTimeout, WSMsgType

from rasa.core.channels.voice_stream.audio_bytes import HERTZ, RasaAudioBytes
from rasa.core.channels.voice_stream.tts.tts_engine import (
    TTSEngine,
    TTSEngineConfig,
    TTSError,
)
from rasa.shared.constants import RIME_API_KEY_ENV_VAR
from rasa.shared.exceptions import ConnectionException

structlogger = structlog.get_logger()


@dataclass
class RimeTTSConfig(TTSEngineConfig):
    speaker: Optional[str] = None
    model_id: Optional[str] = None
    endpoint: Optional[str] = None
    speed_alpha: Optional[float] = None
    segment: Optional[str] = None


class RimeTTS(TTSEngine[RimeTTSConfig]):
    session: Optional[aiohttp.ClientSession] = None
    required_env_vars = (RIME_API_KEY_ENV_VAR,)
    ws: Optional[aiohttp.ClientWebSocketResponse] = None
    streaming_input: bool = True

    def __init__(self, config: Optional[RimeTTSConfig] = None):
        super().__init__(config)
        timeout = ClientTimeout(total=self.config.timeout)
        # Have to create this class-shared session lazily at run time otherwise
        # the async event loop doesn't work
        if self.__class__.session is None or self.__class__.session.closed:
            self.__class__.session = aiohttp.ClientSession(timeout=timeout)

    def get_websocket_url(self) -> str:
        """Build WebSocket URL with query parameters for Rime TTS."""
        if self.config.endpoint:
            base_url = self.config.endpoint
        else:
            base_url = "wss://users.rime.ai/ws2"

        # Build query parameters with required audio format for RasaAudioBytes
        # Audio format and sample rate are fixed to match RasaAudioBytes spec:
        # raw wave, 8khz, 8bit, mono channel, mulaw encoding
        query_params = {
            "speaker": self.config.speaker,
            "modelId": self.config.model_id,
            "audioFormat": "mulaw",  # Fixed: required for RasaAudioBytes
            "samplingRate": str(HERTZ),  # Fixed: 8000 Hz required for RasaAudioBytes
        }

        # Add optional parameters
        if self.config.speed_alpha is not None:
            query_params["speedAlpha"] = str(self.config.speed_alpha)

        if self.config.segment is not None:
            query_params["segment"] = self.config.segment

        return f"{base_url}?{urlencode(query_params)}"

    @staticmethod
    def get_request_headers() -> dict[str, str]:
        """Get headers for WebSocket connection."""
        rime_api_key = os.environ[RIME_API_KEY_ENV_VAR]
        return {
            "Authorization": f"Bearer {rime_api_key}",
        }

    async def connect(self, config: Optional[RimeTTSConfig] = None) -> None:
        """Establish WebSocket connection to Rime TTS."""
        headers = self.get_request_headers()
        ws_url = self.get_websocket_url()

        if self.session is None:
            raise ConnectionException("Client session is not initialized")

        try:
            self.ws = await self.session.ws_connect(
                ws_url,
                headers=headers,
                timeout=float(self.config.timeout) if self.config.timeout else 30,
            )
        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                error_msg = "Authentication failed. Please check your Rime API key."
            else:
                error_msg = f"Connection to Rime TTS failed with status {e.status}"
            structlogger.error(
                "rime.connection.failed",
                status_code=e.status,
                error=error_msg,
            )
            raise ConnectionException(error_msg) from e

    async def close_connection(self) -> None:
        """Close WebSocket connection if it exists."""
        if self.ws and not self.ws.closed:
            await self.ws.close()
            self.ws = None

    async def send_text_chunk(self, text: str) -> None:
        """Send text to TTS engine for continuous streaming.

        Rime accepts text messages as JSON objects with a "text" field.
        The API buffers inputs up to punctuation characters (., ?, !).
        """
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        await self.ws.send_json({"text": text})

    async def signal_text_done(self) -> None:
        """Signal TTS engine to process any remaining buffered text.

        Rime uses the "eos" (end of stream) operation to flush remaining
        buffer and close the connection gracefully.
        """
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        # Send EOS operation to flush and signal completion
        await self.ws.send_json({"operation": "eos"})

    async def stream_audio(self) -> AsyncIterator[RasaAudioBytes]:
        """Stream audio output from the TTS engine.

        Rime returns JSON messages with different types:
        - "chunk": Contains base64-encoded audio data
        - "timestamps": Contains word-level timing information
        - "error": Error messages
        """
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        try:
            async for msg in self.ws:
                if msg.type == WSMsgType.TEXT:
                    # All Rime messages are JSON
                    data = msg.json()
                    msg_type = data.get("type")

                    if msg_type == "chunk":
                        # Audio data chunk - decode base64 and yield
                        base64_audio = data.get("data")
                        if base64_audio:
                            audio_bytes = base64.b64decode(base64_audio)
                            yield self.engine_bytes_to_rasa_audio_bytes(audio_bytes)

                    elif msg_type == "error":
                        # Error occurred
                        error_msg = data.get("message", "Unknown error")
                        structlogger.error("rime.stream_audio.error", error=error_msg)
                        raise TTSError(f"Rime TTS error: {error_msg}")

                elif msg.type == WSMsgType.CLOSED:
                    # Connection closed (expected after EOS)
                    structlogger.debug("rime.stream_audio.connection_closed")
                    break

                elif msg.type == WSMsgType.ERROR:
                    structlogger.error("rime.stream_audio.ws_error")
                    raise TTSError("WebSocket error during audio streaming")

        except Exception as e:
            structlogger.error("rime.stream_audio.error", error=str(e))
            raise TTSError(f"Error during audio streaming: {e}")

    async def synthesize(
        self, text: str, config: Optional[RimeTTSConfig] = None
    ) -> AsyncIterator[RasaAudioBytes]:
        """Generate speech from text using a remote TTS system."""
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        await self.send_text_chunk(text)
        await self.signal_text_done()

        async for audio_chunk in self.stream_audio():
            yield audio_chunk

    def engine_bytes_to_rasa_audio_bytes(self, chunk: bytes) -> RasaAudioBytes:
        """Convert the generated TTS audio bytes into rasa audio bytes."""
        return RasaAudioBytes(chunk)

    @staticmethod
    def get_default_config() -> RimeTTSConfig:
        return RimeTTSConfig(
            speaker="cove",
            model_id="mistv2",
            timeout=30,
            endpoint=None,
            speed_alpha=1.0,
            segment="immediate",  # Synthesize immediately for low latency
        )

    @classmethod
    def from_config_dict(cls, config: Dict) -> "RimeTTS":
        return cls(RimeTTSConfig.from_dict(config))
