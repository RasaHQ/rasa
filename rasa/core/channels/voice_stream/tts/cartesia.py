from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional
import os
import aiohttp
import structlog
from aiohttp import ClientConnectorError

from rasa.core.channels.voice_stream.tts.tts_engine import (
    TTSEngineConfig,
)

from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine, TTSError
from rasa.shared.exceptions import ConnectionException

structlogger = structlog.get_logger()

CARTESIA_API_KEY = "CARTESIA_API_KEY"


@dataclass
class CartesiaTTSConfig(TTSEngineConfig):
    model_id: Optional[str] = None
    version: Optional[str] = None


class CartesiaTTS(TTSEngine[CartesiaTTSConfig]):
    session: Optional[aiohttp.ClientSession] = None

    def __init__(self, config: Optional[CartesiaTTSConfig] = None):
        super().__init__(config)
        # Have to create this class-shared session lazily at run time otherwise
        # the async event loop doesn't work
        if self.__class__.session is None or self.__class__.session.closed:
            self.__class__.session = aiohttp.ClientSession()

    @staticmethod
    def get_tts_endpoint() -> str:
        """Create the endpoint string for cartesia."""
        return "https://api.cartesia.ai/tts/bytes"

    @staticmethod
    def get_request_body(text: str, config: CartesiaTTSConfig) -> Dict:
        """Create the request body for cartesia."""
        # more info on payload:
        # https://docs.cartesia.ai/reference/api-reference/rest/stream-speech-bytes
        return {
            "model_id": config.model_id,
            "transcript": text,
            "language": config.language,
            "voice": {
                "mode": "id",
                "id": config.voice,
            },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_mulaw",
                "sample_rate": 8000,
            },
        }

    @staticmethod
    def get_request_headers(config: CartesiaTTSConfig) -> dict[str, str]:
        cartesia_api_key = os.environ.get(CARTESIA_API_KEY)
        return {
            "Cartesia-Version": str(config.version),
            "Content-Type": "application/json",
            "X-API-Key": str(cartesia_api_key),
        }

    async def synthesize(
        self, text: str, config: Optional[CartesiaTTSConfig] = None
    ) -> AsyncIterator[RasaAudioBytes]:
        """Generate speech from text using a remote TTS system."""
        config = self.config.merge(config)
        payload = self.get_request_body(text, config)
        headers = self.get_request_headers(config)
        url = self.get_tts_endpoint()
        if self.session is None:
            raise ConnectionException("Client session is not initialized")
        try:
            async with self.session.post(
                url, headers=headers, json=payload, chunked=True
            ) as response:
                if 200 <= response.status < 300:
                    async for data in response.content.iter_chunked(1024):
                        yield self.engine_bytes_to_rasa_audio_bytes(data)
                    return
                else:
                    structlogger.error(
                        "azure.synthesize.rest.failed",
                        status_code=response.status,
                        msg=response.text(),
                    )
                    raise TTSError(f"TTS failed: {response.text()}")
        except ClientConnectorError as e:
            raise TTSError(e)

    def engine_bytes_to_rasa_audio_bytes(self, chunk: bytes) -> RasaAudioBytes:
        """Convert the generated tts audio bytes into rasa audio bytes."""
        return RasaAudioBytes(chunk)

    @staticmethod
    def get_default_config() -> CartesiaTTSConfig:
        return CartesiaTTSConfig(
            language="en",
            voice="248be419-c632-4f23-adf1-5324ed7dbf1d",
            model_id="sonic-english",
            version="2024-06-10",
        )

    @classmethod
    def from_config_dict(cls, config: Dict) -> "CartesiaTTS":
        return cls(CartesiaTTSConfig.from_dict(config))
