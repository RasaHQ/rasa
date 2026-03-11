import os
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
import orjson
import structlog
from aiohttp import ClientTimeout, WSMsgType

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
from rasa.shared.constants import DEEPGRAM_API_KEY_ENV_VAR
from rasa.shared.exceptions import ConnectionException

structlogger = structlog.get_logger()

"""
Deepgram TTS engine implementation.
Docs: https://developers.deepgram.com/reference/text-to-speech/speak-streaming
Media Input Settings: https://developers.deepgram.com/docs/tts-media-output-settings#audio-format-combinations
"""


@dataclass
class DeepgramTTSConfig(TTSEngineConfig):
    model_id: Optional[str] = None
    endpoint: Optional[str] = None


class DeepgramTTS(TTSEngine[DeepgramTTSConfig]):
    session: Optional[aiohttp.ClientSession] = None
    required_env_vars = (DEEPGRAM_API_KEY_ENV_VAR,)
    ws: Optional[aiohttp.ClientWebSocketResponse] = None
    streaming_input: bool = True

    @classmethod
    def name(cls) -> str:
        """Return the name identifier for this TTS engine."""
        return "deepgram"

    def __init__(
        self,
        rasa_language: str,
        format: AudioFormat,
        config: Optional[DeepgramTTSConfig] = None,
        additional_languages: Optional[List[str]] = None,
    ):
        super().__init__(rasa_language, format, config, additional_languages)
        timeout = ClientTimeout(total=self.config.timeout)
        # Have to create this class-shared session lazily at run time otherwise
        # the async event loop doesn't work
        if self.__class__.session is None or self.__class__.session.closed:
            self.__class__.session = aiohttp.ClientSession(timeout=timeout)

    @staticmethod
    def get_request_headers(config: DeepgramTTSConfig) -> dict[str, str]:
        deepgram_api_key = os.environ[DEEPGRAM_API_KEY_ENV_VAR]
        return {
            "Authorization": f"Token {deepgram_api_key!s}",
        }

    async def connect(self, config: Optional[DeepgramTTSConfig] = None) -> None:
        headers = self.get_request_headers(self.config)
        ws_url = self.get_websocket_url(self.config)

        if self.session is None:
            raise ConnectionException("Client session is not initialized")

        self.ws = await self.session.ws_connect(
            ws_url,
            headers=headers,
            timeout=float(self.config.timeout),
        )

    async def close_connection(self) -> None:
        """Close WebSocket connection if it exists."""
        if self.ws and not self.ws.closed:
            await self.ws.close()
            self.ws = None

    def get_websocket_url(self, config: DeepgramTTSConfig) -> str:
        """Build WebSocket URL with query parameters."""
        if self.audio_format in (L16_24KHZ, L16_48KHZ):
            encoding = "linear16"
        elif self.audio_format == MULAW_8KHZ:
            encoding = "mulaw"
        else:
            raise TTSError(
                f"Unsupported audio format {self.audio_format} for Deepgram TTS"
            )
        base_url = config.endpoint
        query_params = {
            "model": self.current_language_config.model,
            "encoding": encoding,
            "sample_rate": self.audio_format.sample_rate,
        }
        return f"{base_url}?{urlencode(query_params)}"

    async def send_text_chunk(self, text: str) -> None:
        """Send text to TTS engine for continuous streaming.

        This sends text to Deepgram but doesn't return anything.
        Audio will be available via stream_audio().
        """
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        await self.ws.send_json(
            {
                "type": "Speak",
                "text": text,
            }
        )

    async def signal_text_done(self) -> None:
        """Signal TTS engine to flush any buffered text.

        This tells Deepgram to process any remaining text and finish.
        """
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        await self.ws.send_json({"type": "Flush"})

    async def signal_interrupt(self) -> None:
        """Clear the TTS engine buffer."""
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        await self.ws.send_json({"type": "Clear"})
        structlogger.debug("deepgram.tts.clear")

    async def stream_audio(self) -> AsyncIterator[RasaAudioBytes]:
        """Stream audio output from the TTS engine.

        This continuously yields audio chunks as they arrive from Deepgram.
        Stops when it receives a Flushed or Close message.
        """
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        try:
            async for msg in self.ws:
                if msg.type == WSMsgType.BINARY:
                    # Binary data is the raw audio - yield it
                    yield self.engine_bytes_to_rasa_audio_bytes(msg.data)

                elif msg.type == WSMsgType.TEXT:
                    # Handle control messages
                    data = orjson.loads(msg.data)
                    if data.get("type") == "Flushed":
                        # All audio has been sent, stop streaming
                        structlogger.debug("deepgram.stream_audio.flushed")
                        break
                    elif data.get("type") == "Close":
                        # Connection closing
                        structlogger.debug("deepgram.stream_audio.close")
                        break
                    elif data.get("type") == "Cleared":
                        structlogger.debug("deepgram.stream_audio.cleared")
                        break

                elif msg.type == WSMsgType.CLOSED:
                    structlogger.debug("deepgram.stream_audio.ws_closed")
                    break

                elif msg.type == WSMsgType.ERROR:
                    structlogger.error(
                        "deepgram.stream_audio.ws_error", error=str(msg.data)
                    )
                    raise TTSError(f"WebSocket error: {msg.data}")

        except Exception as e:
            structlogger.error("deepgram.stream_audio.error", error=str(e))
            raise TTSError(f"Error during audio streaming: {e}")

    async def synthesize(
        self, text: str, config: Optional[DeepgramTTSConfig] = None
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
        # WebSocket returns raw audio bytes directly
        return RasaAudioBytes(chunk, format=self.audio_format)

    @staticmethod
    def get_default_config() -> DeepgramTTSConfig:
        return DeepgramTTSConfig(
            endpoint="wss://api.deepgram.com/v1/speak",
            timeout=30,
            language_map={
                "en": TTSLanguageMapEntry(
                    model="aura-2-andromeda-en",
                ),
            },
        )

    @classmethod
    def from_config_dict(
        cls,
        config: Dict,
        format: AudioFormat,
        rasa_language: str,
        additional_languages: Optional[List[str]] = None,
    ) -> "DeepgramTTS":
        return cls(
            rasa_language=rasa_language,
            format=format,
            config=DeepgramTTSConfig.from_dict(config),
            additional_languages=additional_languages,
        )

    async def set_language(self, rasa_language: str) -> None:
        """Update the TTS language for next synthesis"""
        await super().set_language(rasa_language)

        # need to reconnect to apply new language
        await self.close_connection()
        await self.connect()
