import os
from typing import AsyncIterator, Dict, Optional
from dataclasses import dataclass

import aiohttp
import structlog
from aiohttp import ClientConnectorError

from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.tts.tts_engine import (
    TTSEngine,
    TTSEngineConfig,
    TTSError,
)
from rasa.shared.exceptions import ConnectionException


structlogger = structlog.get_logger()


@dataclass
class AzureTTSConfig(TTSEngineConfig):
    speech_region: Optional[str] = None


class AzureTTS(TTSEngine[AzureTTSConfig]):
    session: Optional[aiohttp.ClientSession] = None

    def __init__(self, config: Optional[AzureTTSConfig] = None):
        super().__init__(config)
        # Have to create this class-shared session lazily at run time otherwise
        # the async event loop doesn't work
        if self.__class__.session is None or self.__class__.session.closed:
            self.__class__.session = aiohttp.ClientSession()

    async def synthesize(
        self, text: str, config: Optional[AzureTTSConfig] = None
    ) -> AsyncIterator[RasaAudioBytes]:
        """Generate speech from text using a remote TTS system."""
        config = self.config.merge(config)
        azure_speech_url = self.get_tts_endpoint(config)
        headers = self.get_request_headers()
        body = self.create_request_body(text, config)
        if self.session is None:
            raise ConnectionException("Client session is not initialized")
        try:
            async with self.session.post(
                azure_speech_url, headers=headers, data=body, chunked=True
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

    @staticmethod
    def get_request_headers() -> dict[str, str]:
        azure_speech_api_key = os.environ["AZURE_SPEECH_API_KEY"]
        return {
            "Ocp-Apim-Subscription-Key": azure_speech_api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "raw-8khz-8bit-mono-mulaw",
        }

    @staticmethod
    def get_tts_endpoint(config: AzureTTSConfig) -> str:
        return f"https://{config.speech_region}.tts.speech.microsoft.com/cognitiveservices/v1"

    @staticmethod
    def create_request_body(text: str, conf: AzureTTSConfig) -> str:
        return f"""
        <speak version='1.0' xml:lang='{conf.language}'>
            <voice xml:lang='{conf.language}' name='{conf.voice}'>
                {text}
            </voice>
        </speak>"""

    def engine_bytes_to_rasa_audio_bytes(self, chunk: bytes) -> RasaAudioBytes:
        """Convert the generated tts audio bytes into rasa audio bytes."""
        return RasaAudioBytes(chunk)

    @staticmethod
    def get_default_config() -> AzureTTSConfig:
        return AzureTTSConfig(
            language="en-US",
            voice="en-US-JennyNeural",
            speech_region="germanywestcentral",
        )

    @classmethod
    def from_config_dict(cls, config: Dict) -> "AzureTTS":
        return cls(AzureTTSConfig.from_dict(config))
