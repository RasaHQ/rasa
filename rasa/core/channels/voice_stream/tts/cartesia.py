import base64
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

import aiohttp
import structlog
from aiohttp import ClientTimeout

from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    AudioFormat,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.tts.tts_engine import (
    TTSEngine,
    TTSEngineConfig,
    TTSError,
    TTSLanguageMapEntry,
)
from rasa.shared.constants import CARTESIA_API_KEY_ENV_VAR
from rasa.shared.exceptions import ConnectionException

structlogger = structlog.get_logger()

"""
Cartesia TTS engine implementation.
Docs: https://docs.cartesia.ai/api-reference/tts/websocket
Audio Format: https://docs.cartesia.ai/build-with-cartesia/capability-guides/choosing-tts-parameters#reference
"""


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

    # Each synthesis context has a unique ID
    # gets reset in signal_text_done()
    context_id = uuid4().hex

    @classmethod
    def name(cls) -> str:
        """Return the name identifier for this TTS engine."""
        return "cartesia"

    def __init__(
        self,
        rasa_language: str,
        format: AudioFormat,
        config: Optional[CartesiaTTSConfig] = None,
        additional_languages: Optional[List[str]] = None,
    ):
        super().__init__(rasa_language, format, config, additional_languages or [])
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

    async def _send_tts_request(self, text: str, flush: bool) -> None:
        """Send TTS request to Cartesia via WebSocket."""
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        if self.audio_format in (L16_24KHZ, L16_48KHZ):
            encoding = "pcm_s16le"
        elif self.audio_format == MULAW_8KHZ:
            encoding = "pcm_mulaw"
        else:
            raise TTSError(f"Unsupported audio format: {self.audio_format}")

        message: Dict[str, object] = {
            "model_id": self.config.model_id,
            "voice": {
                "mode": "id",
                "id": self.current_language_config.voice,
            },
            "language": self.current_language_config.engine_language_key,
            "output_format": {
                "container": "raw",
                "encoding": encoding,
                "sample_rate": self.audio_format.sample_rate,
            },
            "context_id": self.context_id,
        }

        if flush:
            message["flush"] = True
        else:
            message["transcript"] = text
            message["continue"] = True

        await self.ws.send_json(message)

    async def send_text_chunk(self, text: str) -> None:
        """Send text to TTS engine for continuous streaming.

        This sends text to Cartesia but doesn't return anything.
        Audio will be available via stream_audio().
        """
        await self._send_tts_request(text, flush=False)

    async def signal_text_done(self) -> None:
        """Signal TTS engine to process any remaining buffered text.

        This tells Cartesia that all text has been sent and to finish processing.
        """
        await self._send_tts_request("", flush=True)
        self.context_id = uuid4().hex  # Reset context ID for next synthesis

    async def stream_audio(self) -> AsyncIterator[RasaAudioBytes]:
        """Stream audio output from the TTS engine.

        This continuously yields audio chunks as they arrive from Cartesia.
        Stops when it receives a done message or flush completion.
        """
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        try:
            async for msg in self.ws:
                data = msg.json()
                msg_type = data.get("type")

                if msg_type == "chunk":
                    # Audio data chunk - decode base64 and yield
                    base64_audio = data.get("data")
                    if base64_audio:
                        audio_bytes = base64.b64decode(base64_audio)
                        yield self.engine_bytes_to_rasa_audio_bytes(audio_bytes)

                elif msg_type == "done":
                    # TTS processing is done
                    structlogger.debug("cartesia.stream_audio.done")
                    return

                elif msg_type == "error":
                    # Error occurred
                    error_msg = data.get("error", "Unknown error")
                    structlogger.error("cartesia.stream_audio.error", error=error_msg)
                    raise TTSError(f"Cartesia TTS error: {error_msg}")

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
        return RasaAudioBytes(chunk, format=self.audio_format)

    @staticmethod
    def get_default_config() -> CartesiaTTSConfig:
        return CartesiaTTSConfig(
            timeout=30,
            model_id="sonic-3",
            version="2025-04-16",
            endpoint="wss://api.cartesia.ai/tts/websocket",
            language_map={
                "en": TTSLanguageMapEntry(
                    language="en",
                    voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
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
    ) -> "CartesiaTTS":
        cfg = (
            config
            if type(config).__name__ == "CartesiaTTSConfig"
            else CartesiaTTSConfig.from_dict(config)
        )
        return cls(
            config=cfg,
            format=format,
            rasa_language=rasa_language,
            additional_languages=additional_languages,
        )
