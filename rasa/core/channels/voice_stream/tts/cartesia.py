import base64
import os
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional

import aiohttp
import structlog
from aiohttp import ClientTimeout, WSMsgType

from rasa.core.channels.voice_stream.audio_bytes import HERTZ, RasaAudioBytes
from rasa.core.channels.voice_stream.tts.tts_engine import (
    TTSEngine,
    TTSEngineConfig,
    TTSError,
)
from rasa.shared.constants import CARTESIA_API_KEY_ENV_VAR
from rasa.shared.exceptions import ConnectionException

structlogger = structlog.get_logger()


@dataclass
class CartesiaTTSConfig(TTSEngineConfig):
    model_id: Optional[str] = None
    version: Optional[str] = None
    endpoint: Optional[str] = None


class CartesiaTTS(TTSEngine[CartesiaTTSConfig]):
    session: Optional[aiohttp.ClientSession] = None
    required_env_vars = (CARTESIA_API_KEY_ENV_VAR,)
    ws: Optional[aiohttp.ClientWebSocketResponse] = None
    streaming_input: bool = True

    def __init__(self, config: Optional[CartesiaTTSConfig] = None):
        super().__init__(config)
        timeout = ClientTimeout(total=self.config.timeout)
        # Have to create this class-shared session lazily at run time otherwise
        # the async event loop doesn't work
        if self.__class__.session is None or self.__class__.session.closed:
            self.__class__.session = aiohttp.ClientSession(timeout=timeout)

    @staticmethod
    def get_request_headers(config: CartesiaTTSConfig) -> dict[str, str]:
        """Get headers for WebSocket connection."""
        cartesia_api_key = os.environ[CARTESIA_API_KEY_ENV_VAR]
        return {
            "Cartesia-Version": str(config.version),
            "X-API-Key": str(cartesia_api_key),
        }

    async def connect(self, config: Optional[CartesiaTTSConfig] = None) -> None:
        """Establish WebSocket connection to Cartesia TTS."""
        headers = self.get_request_headers(self.config)
        ws_url = self.config.endpoint

        if self.session is None:
            raise ConnectionException("Client session is not initialized")

        if not ws_url:
            raise ConnectionException("Cartesia endpoint not configured")

        self.ws = await self.session.ws_connect(
            ws_url,
            headers=headers,
            timeout=float(self.config.timeout) if self.config.timeout else 30,
        )

    async def close_connection(self) -> None:
        """Close WebSocket connection if it exists."""
        if self.ws and not self.ws.closed:
            await self.ws.close()
            self.ws = None

    async def send_text_chunk(self, text: str) -> None:
        """Send text to TTS engine for continuous streaming.

        This sends text to Cartesia but doesn't return anything.
        Audio will be available via stream_audio().
        """
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        await self.ws.send_json(
            {
                "model_id": self.config.model_id,
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": self.config.voice,
                },
                "language": self.config.language,
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_mulaw",
                    "sample_rate": HERTZ,
                },
                "continue": True,  # Indicate more text may follow
            }
        )

    async def signal_text_done(self) -> None:
        """Signal TTS engine to process any remaining buffered text.

        This tells Cartesia that all text has been sent and to finish processing.
        """
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        # Send flush message to signal we're done
        await self.ws.send_json({"flush": True})

    async def stream_audio(self) -> AsyncIterator[RasaAudioBytes]:
        """Stream audio output from the TTS engine.

        This continuously yields audio chunks as they arrive from Cartesia.
        Stops when it receives a done message or flush completion.
        """
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        try:
            async for msg in self.ws:
                if msg.type == WSMsgType.TEXT:
                    # Handle JSON messages
                    data = msg.json()
                    msg_type = data.get("type")

                    if msg_type == "chunk":
                        # Audio data chunk - decode base64 and yield
                        base64_audio = data.get("data")
                        if base64_audio:
                            audio_bytes = base64.b64decode(base64_audio)
                            yield self.engine_bytes_to_rasa_audio_bytes(audio_bytes)

                        # Check if this is the final chunk
                        if data.get("done", False):
                            structlogger.debug("cartesia.stream_audio.done")
                            break

                    elif msg_type == "error":
                        # Error occurred
                        error_msg = data.get("error", "Unknown error")
                        structlogger.error(
                            "cartesia.stream_audio.error", error=error_msg
                        )
                        raise TTSError(f"Cartesia TTS error: {error_msg}")

                elif msg.type == WSMsgType.ERROR:
                    structlogger.error("cartesia.stream_audio.ws_error")
                    raise TTSError("WebSocket error during audio streaming")

        except Exception as e:
            structlogger.error("cartesia.stream_audio.error", error=str(e))
            raise TTSError(f"Error during audio streaming: {e}")

    async def synthesize(
        self, text: str, config: Optional[CartesiaTTSConfig] = None
    ) -> AsyncIterator[RasaAudioBytes]:
        """Generate speech from text using a remote TTS system."""
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        await self.send_text_chunk(text)
        await self.signal_text_done()

        async for audio_chunk in self.stream_audio():
            yield audio_chunk

    def engine_bytes_to_rasa_audio_bytes(self, chunk: bytes) -> RasaAudioBytes:
        """Convert the generated tts audio bytes into rasa audio bytes."""
        return RasaAudioBytes(chunk)

    @staticmethod
    def get_default_config() -> CartesiaTTSConfig:
        return CartesiaTTSConfig(
            language="en",
            voice="248be419-c632-4f23-adf1-5324ed7dbf1d",
            timeout=30,
            model_id="sonic-english",
            version="2024-06-10",
            endpoint="wss://api.cartesia.ai/tts/websocket",
        )

    @classmethod
    def from_config_dict(cls, config: Dict) -> "CartesiaTTS":
        return cls(CartesiaTTSConfig.from_dict(config))
