import base64
import os
from typing import Any, AsyncIterator, List, Optional
from urllib.parse import urlencode
from uuid import uuid4

import aiohttp
import structlog
from aiohttp import ClientTimeout

from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
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
from rasa.shared.constants import RIME_API_KEY_ENV_VAR
from rasa.shared.exceptions import ConnectionException

structlogger = structlog.get_logger()

"""
Rime TTS engine implementation.
Docs: https://docs.rime.ai/api-reference/endpoint/websockets-json#variable-parameters
"""


class RimeTTSConfig(TTSEngineConfig):
    """Rime TTS variable parameters:
    See get_default_config() for default values.
    """

    # Rime TTS calls "voice" as "speaker"
    # the voice used for synthesis
    speaker: Optional[str] = None

    # Model ID to be used for synthesis
    model_id: Optional[str] = None

    # The Rime TTS API endpoint URL
    endpoint: Optional[str] = None

    # Adjusts the speed of speech.
    speed_alpha: Optional[float] = None

    # Controls how text is segmented for synthesis.
    segment: Optional[str] = None

    # Skips text normalization of the input text prior to synthesizing
    # audio. This will reduce latency at the cost of some possible
    # mispronunciation of digits and abbreviations.
    no_text_normalization: Optional[bool] = None


class RimeTTS(TTSEngine[RimeTTSConfig]):
    session: Optional[aiohttp.ClientSession] = None
    required_env_vars = (RIME_API_KEY_ENV_VAR,)
    ws: Optional[aiohttp.ClientWebSocketResponse] = None
    # Indicates if TTS is being used in streaming input mode
    # TODO: Set it to True once Rime TTS streaming input bug
    # is fixed. No timelines yet.
    streaming_input: bool = False

    # Each synthesis context has a unique ID
    # gets reset in signal_text_done()
    context_id = uuid4().hex

    @classmethod
    def name(cls) -> str:
        """Return the name identifier for this TTS engine."""
        return "rime"

    def __init__(
        self,
        rasa_language: str,
        format: AudioFormat,
        config: Optional[RimeTTSConfig] = None,
        additional_languages: Optional[List[str]] = None,
    ):
        super().__init__(rasa_language, format, config, additional_languages or [])
        timeout = ClientTimeout(total=self.config.timeout)
        # Have to create this class-shared session lazily at run time otherwise
        # the async event loop doesn't work
        if self.__class__.session is None or self.__class__.session.closed:
            self.__class__.session = aiohttp.ClientSession(timeout=timeout)

    def get_websocket_url(self) -> str:
        """Build WebSocket URL with query parameters for Rime TTS."""
        base_url = self.config.endpoint

        if self.audio_format is L16_24KHZ:
            encoding = "pcm"
        elif self.audio_format is MULAW_8KHZ:
            encoding = "mulaw"
        else:
            # TODO: log a warning and transcode instead.
            raise TTSError(
                f"Unsupported audio format: {self.audio_format}. Rime does not "
                "support sample rates above 44.1kHz."
            )

        query_params = {
            "speaker": self.current_language_config.voice,
            "modelId": self.config.model_id,
            "lang": self.current_language_config.engine_language_key,
            "audioFormat": encoding,
            "samplingRate": self.audio_format.sample_rate,
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

        await self.ws.send_json({"text": text, "contextId": self.context_id})

    async def signal_text_done(self) -> None:
        """Signal TTS engine to process any remaining buffered text.

        Rime uses the "eos" (end of stream) operation to flush remaining
        buffer and close the connection gracefully.
        """
        if not self.ws or self.ws.closed:
            raise TTSError("WebSocket connection not established")

        await self.ws.send_json({"operation": "flush"})
        self.context_id = uuid4().hex  # Reset context ID for next synthesis

    async def signal_interrupt(self) -> None:
        """Clear the TTS engine buffer."""
        if not self.ws or self.ws.closed:
            return
        await self.ws.send_json({"operation": "clear"})
        structlogger.debug("rime.tts.clear")

    async def stop_streaming(self) -> None:
        await super().stop_streaming()
        if self.stream_state == self.stream_state.SENDING_RESPONSE_CHUNKS:
            self.stream_state = self.stream_state.INTERRUPTED
        elif self.stream_state == self.stream_state.RESPONSE_CHUNKS_SENT:
            self.stream_state = self.stream_state.NO_STREAMING
            await self.signal_interrupt()

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
                data = msg.json()
                msg_type = data.get("type")

                if msg_type == "chunk":
                    # Audio data chunk - decode base64 and yield
                    base64_audio = data.get("data")
                    if base64_audio:
                        audio_bytes = base64.b64decode(base64_audio)
                        yield self.engine_bytes_to_rasa_audio_bytes(audio_bytes)

                elif msg_type == "done":
                    # All audio has been sent, stop streaming
                    break

                elif msg_type == "error":
                    # Error occurred
                    error_msg = data.get("message", "Unknown error")
                    structlogger.error("rime.stream_audio.error", error=error_msg)
                    raise TTSError(f"Rime TTS error: {error_msg}")

                elif msg_type == "timestamps":
                    # Word-level timing information
                    # Rasa doesn't use this yet
                    pass

                else:
                    structlogger.warning(
                        "rime.stream_audio.unknown_message", message=data
                    )
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
        return RasaAudioBytes(chunk, format=self.audio_format)

    @staticmethod
    def get_default_config(rasa_language: str) -> RimeTTSConfig:
        return RimeTTSConfig(
            model_id="mistv2",
            timeout=30,
            endpoint="wss://users.rime.ai/ws2",
            speed_alpha=1.0,
            segment="immediate",  # Synthesize immediately for low latency
            no_text_normalization=False,
            language_map={
                rasa_language: TTSLanguageMapEntry(
                    language="eng",
                    voice="cove",
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
    ) -> "RimeTTS":
        return cls(
            rasa_language=rasa_language,
            format=format,
            config=RimeTTSConfig.model_validate(config),
            additional_languages=additional_languages,
        )

    async def set_language(self, rasa_language: str) -> None:
        """Update the TTS language for next synthesis"""
        await super().set_language(rasa_language)

        # need to reconnect to apply new language
        await self.close_connection()
        await self.connect()
