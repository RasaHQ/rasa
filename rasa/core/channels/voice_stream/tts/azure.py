import os
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

import aiohttp
import structlog
from aiohttp import ClientConnectorError, ClientTimeout

from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    AudioFormat,
    CurrentLanguageConfig,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.tts.tts_engine import (
    TTSEngine,
    TTSEngineConfig,
    TTSError,
    TTSLanguageMapEntry,
)
from rasa.shared.constants import AZURE_SPEECH_API_KEY_ENV_VAR
from rasa.shared.exceptions import ConnectionException

structlogger = structlog.get_logger()


@dataclass
class AzureTTSConfig(TTSEngineConfig):
    """Configuration for Azure TTS.

    Attributes:
        speech_region: Azure speech service region.
        endpoint: Custom endpoint URL. If None, constructed from speech_region.
        timeout: Request timeout in seconds.
    """

    speech_region: str = "eastus"
    endpoint: Optional[str] = None
    timeout: int = 10


class AzureTTS(TTSEngine[AzureTTSConfig]):
    session: Optional[aiohttp.ClientSession] = None
    required_env_vars = (AZURE_SPEECH_API_KEY_ENV_VAR,)
    streaming_input: bool = False

    @classmethod
    def name(cls) -> str:
        """Return the name identifier for this TTS engine."""
        return "azure"

    def __init__(
        self,
        rasa_language: str,
        format: AudioFormat,
        config: Optional[AzureTTSConfig] = None,
        additional_languages: Optional[List[str]] = None,
    ):
        super().__init__(rasa_language, format, config, additional_languages)
        timeout = ClientTimeout(total=self.config.timeout)
        # Have to create this class-shared session lazily at run time otherwise
        # the async event loop doesn't work
        if self.__class__.session is None or self.__class__.session.closed:
            self.__class__.session = aiohttp.ClientSession(timeout=timeout)

    async def synthesize(
        self, text: str, config: Optional[AzureTTSConfig] = None
    ) -> AsyncIterator[RasaAudioBytes]:
        """Generate speech from text using a remote TTS system."""
        config = self.config.merge(config)
        azure_speech_url = self.get_tts_endpoint(config)
        headers = self.get_request_headers()
        body = self.create_request_body(text, self.current_language_config)
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
                elif response.status == 401:
                    structlogger.error(
                        "azure.synthesize.rest.authentication_failed",
                        status_code=response.status,
                    )
                    raise TTSError(
                        f"Authentication failed. Please check your API key: {response.status}"  # noqa: E501
                    )
                else:
                    response_text = await response.text()
                    structlogger.error(
                        "azure.synthesize.rest.failed",
                        status_code=response.status,
                        msg=response_text,
                    )
                    raise TTSError(f"TTS failed: {response_text}")
        except ClientConnectorError as e:
            raise TTSError(e)
        except TimeoutError as e:
            raise TTSError(e)

    def get_request_headers(self) -> dict[str, str]:
        _AZURE_OUTPUT_FORMATS: dict[AudioFormat, str] = {
            MULAW_8KHZ: "raw-8khz-8bit-mono-mulaw",
            L16_24KHZ: "raw-24khz-16bit-mono-pcm",
            L16_48KHZ: "raw-48khz-16bit-mono-pcm",
        }
        azure_output_format = _AZURE_OUTPUT_FORMATS.get(self.audio_format)
        if not azure_output_format:
            raise TTSError(
                f"Audio format {self.audio_format} is not supported by Azure TTS."
            )
        azure_speech_api_key = os.environ[AZURE_SPEECH_API_KEY_ENV_VAR]
        return {
            "Ocp-Apim-Subscription-Key": azure_speech_api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": azure_output_format,
        }

    @staticmethod
    def get_tts_endpoint(config: AzureTTSConfig) -> str:
        if config.endpoint is not None:
            return config.endpoint
        else:
            return (
                f"https://{config.speech_region}.tts.speech.microsoft.com/"
                f"cognitiveservices/v1"
            )

    @staticmethod
    def create_request_body(text: str, lang_model: CurrentLanguageConfig) -> str:
        return f"""
        <speak
            version='1.0'
            xml:lang='{lang_model.engine_language_key}'
            xmlns:mstts='http://www.w3.org/2001/mstts'
            xmlns='http://www.w3.org/2001/10/synthesis'
        >
            <voice xml:lang='{lang_model.engine_language_key}'
            name='{lang_model.voice}'>
                {text}
            </voice>
        </speak>"""

    def engine_bytes_to_rasa_audio_bytes(self, chunk: bytes) -> RasaAudioBytes:
        """Convert the generated tts audio bytes into rasa audio bytes."""
        return RasaAudioBytes(chunk, format=self.audio_format)

    @staticmethod
    def get_default_config() -> AzureTTSConfig:
        return AzureTTSConfig(
            language_map={
                "en": TTSLanguageMapEntry(
                    language="en-US",
                    voice="en-US-JennyNeural",
                ),
            }
        )

    @classmethod
    def from_config_dict(
        cls,
        config: Dict,
        format: AudioFormat,
        rasa_language: str,
        additional_languages: Optional[List[str]] = None,
    ) -> "AzureTTS":
        return cls(
            rasa_language=rasa_language,
            format=format,
            config=AzureTTSConfig.from_dict(config),
            additional_languages=additional_languages,
        )
