import asyncio
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp
import azure.cognitiveservices.speech as speechsdk
import structlog
from aiohttp import ClientConnectorError, ClientTimeout
from azure.cognitiveservices.speech import SpeechSynthesisOutputFormat

from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    AudioFormat,
    CurrentLanguageConfig,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.tts.config import StreamingConfig
from rasa.core.channels.voice_stream.tts.tts_engine import (
    TTSEngine,
    TTSEngineConfig,
    TTSError,
    TTSLanguageMapEntry,
)
from rasa.shared.constants import AZURE_SPEECH_API_KEY_ENV_VAR
from rasa.shared.exceptions import ConnectionException

structlogger = structlog.get_logger()


class _AudioOutputCallback(speechsdk.audio.PushAudioOutputStreamCallback):
    """Bridge Azure Speech SDK audio output (SDK thread) to an asyncio queue.

    Azure Speech SDK uses a separate thread to handle audio streaming.
    So in order to push audio bytes back to the async loop in which Azure TTS is running
    we need to encapsulate that loop and queue into one object
    for easier handling and maintenance.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        audio_format: AudioFormat,
    ) -> None:
        super().__init__()
        self._loop = loop
        self.queue = queue  # mutable: swapped per response
        self.audio_format = audio_format

    def write(self, audio_buffer: memoryview) -> int:
        """Write audio buffer to the audio queue.

        Because audio queue is an asyncio.Queue, it must be used in the same async loop
        it was created in.
        We use `call_soon_threadsafe` from the async loop
        (in which async queue was created in) to call
        `put_nowait` method from async queue to put data into the queue.
        """
        audio_bytes = RasaAudioBytes(bytes(audio_buffer), self.audio_format)
        self._loop.call_soon_threadsafe(self.queue.put_nowait, audio_bytes)
        return len(audio_buffer)

    def close(self) -> None:
        # Needed to implement from `PushAudioOutputStreamCallback`.
        pass


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
    ws_endpoint: Optional[str] = None


class AzureTTS(TTSEngine[AzureTTSConfig]):
    session: Optional[aiohttp.ClientSession] = None
    required_env_vars = (AZURE_SPEECH_API_KEY_ENV_VAR,)
    streaming_input: bool = True

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
        if self.__class__.session is None or self.__class__.session.closed:
            self.__class__.session = aiohttp.ClientSession(timeout=timeout)

        self._use_streaming: bool = False
        self._text_buffer: List[str] = []
        self._audio_queue: asyncio.Queue = asyncio.Queue()

        # SDK objects (initialized in connect())
        self._speech_config: Optional[speechsdk.SpeechConfig] = None
        self._synthesizer: Optional[speechsdk.SpeechSynthesizer] = None
        self._callback: Optional[_AudioOutputCallback] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Per-response streaming state
        self._tts_request: Optional[speechsdk.SpeechSynthesisRequest] = None
        self._tts_future: Optional[Any] = None

    # ── Lifecycle ────────────────────────────────────────────────────

    async def prepare_response(
        self, streaming_config: Optional[StreamingConfig] = None
    ) -> None:
        """Set per-response mode based on SSML content and SDK availability."""
        config = streaming_config or StreamingConfig()
        sdk_available = self._synthesizer is not None
        self._use_streaming = not config.response_text_contains_ssml and sdk_available
        self._text_buffer.clear()
        self._tts_request = None
        self._tts_future = None

        if self._use_streaming:
            self._tts_request = speechsdk.SpeechSynthesisRequest(
                input_type=speechsdk.SpeechSynthesisRequestInputType.TextStream
            )
            self._tts_future = (
                self._synthesizer.speak_async(self._tts_request)
                if self._synthesizer
                else None
            )

        structlogger.debug(
            "azure.prepare_response",
            use_streaming=self._use_streaming,
            response_text_contains_ssml=config.response_text_contains_ssml,
            sdk_available=sdk_available,
        )

    async def connect(self, config: Optional[AzureTTSConfig] = None) -> None:
        """Initialize the Azure Speech SDK synthesizer."""
        self._loop = asyncio.get_running_loop()
        cfg = self.config

        endpoint = (
            cfg.ws_endpoint
            or f"wss://{cfg.speech_region}.tts.speech.microsoft.com/"
            f"cognitiveservices/websocket/v2"
        )

        if not self.current_language_config.voice:
            structlogger.warning(
                "azure.sdk.connect_skipped.missing_voice",
                language=self.current_language_config.rasa_language_key,
                hint="Add 'voice' to the language_map entry in your TTS config.",
            )
            return

        try:
            self._speech_config = speechsdk.SpeechConfig(
                subscription=os.environ[AZURE_SPEECH_API_KEY_ENV_VAR],
                endpoint=endpoint,
            )
            self._speech_config.speech_synthesis_voice_name = (
                self.current_language_config.voice
            )
            self._speech_config.set_speech_synthesis_output_format(
                self._get_azure_audio_format()
            )

            # Synthesizer is created once and reused across responses;
            # the callback's queue reference is swapped in prepare_response().
            self._callback = _AudioOutputCallback(
                self._loop, self._audio_queue, self.audio_format
            )
            push_stream = speechsdk.audio.PushAudioOutputStream(self._callback)
            audio_config = speechsdk.audio.AudioOutputConfig(stream=push_stream)
            self._synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self._speech_config, audio_config=audio_config
            )
            structlogger.info(
                "azure.sdk.connected",
                endpoint=endpoint,
                voice=self.current_language_config.voice,
            )
        except Exception as e:
            structlogger.warning("azure.sdk.connect_failed", error=str(e))
            self._synthesizer = None

    async def close_connection(self) -> None:
        """Release SDK resources."""
        self._synthesizer = None
        self._speech_config = None
        self._callback = None

    # ── Streaming pipeline (SDK TextStream) ───────────────────────────

    async def send_text_chunk(self, text: str) -> None:
        """Send text to TTS engine.

        In streaming mode: writes text directly to the SDK TextStream.
        The SDK buffers internally and maintains natural prosody.
        In non-streaming mode: buffers text for later REST synthesis.
        """
        if self._use_streaming and self._tts_request:
            structlogger.debug("azure.send_text_chunk.for_streaming", text=text)
            self._tts_request.input_stream.write(text)
        else:
            self._text_buffer.append(text)

    async def signal_text_done(self) -> None:
        """Signal that no more text chunks will be sent."""
        if self._use_streaming:
            structlogger.debug("azure.signal_text_done.for_streaming")
            self._signal_streaming_done()
        else:
            await self._synthesize_buffered_text()

    def _signal_streaming_done(self) -> None:
        """Close the SDK input stream and await synthesis completion.

        Runs future.get() in a thread pool so the event loop isn't blocked.
        Pushes a None sentinel to the audio queue when synthesis finishes.
        """

        self._tts_request.input_stream.close() if self._tts_request else None
        if not self._tts_future:
            return
        # Capture references for the closure; self._audio_queue and
        # self._tts_future are reset on the next prepare_response().

        def _wait_and_signal() -> None:
            try:
                self._tts_future.get() if self._tts_future else None
            except Exception as e:
                structlogger.error(
                    "azure.signal_text_done.future_error",
                    error=str(e),
                )
            finally:
                self._loop.call_soon_threadsafe(
                    self._audio_queue.put_nowait, None
                ) if self._loop else None

        self._loop.run_in_executor(None, _wait_and_signal) if self._loop else None

    async def _synthesize_buffered_text(self) -> None:
        """Synthesize all buffered text via REST and push to audio queue."""
        full_text = "".join(self._text_buffer)
        structlogger.debug(
            "azure.synthesize_buffered_text.for_streaming", text=full_text
        )
        self._text_buffer.clear()
        if not full_text:
            await self._audio_queue.put(None)
            return
        try:
            async for chunk in self._synthesize_rest(full_text):
                await self._audio_queue.put(chunk)
        except Exception as e:
            structlogger.error("azure.synthesize_buffered_text.error", error=str(e))
        finally:
            await self._audio_queue.put(None)

    async def stream_audio(self) -> AsyncIterator[RasaAudioBytes]:
        """Yield audio chunks until a None sentinel signals completion."""
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                return
            yield chunk

    # ── Non-streaming template responses (send_text_message path) ────

    async def synthesize(
        self, text: str, config: Optional[AzureTTSConfig] = None
    ) -> AsyncIterator[RasaAudioBytes]:
        """Generate speech from text via REST API."""
        async for chunk in self._synthesize_rest(text, config):
            yield chunk

    # ── REST synthesis ────────────────────────────────────────────────

    async def _synthesize_rest(
        self, text: str, config: Optional[AzureTTSConfig] = None
    ) -> AsyncIterator[RasaAudioBytes]:
        """Generate speech via REST API with SSML wrapping."""
        config = self.config.merge(config)
        if not self.current_language_config.voice:
            raise TTSError(
                f"No voice configured for language "
                f"'{self.current_language_config.rasa_language_key}'. "
                f"Add 'voice' to the language_map entry in your TTS config."
            )
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
            },
            speech_region="eastus",
            endpoint=None,
            ws_endpoint=None,
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

    async def stop_streaming(self) -> None:
        """Clear the TTS engine buffer."""
        structlogger.debug("azure_tts.stop_streaming")
        if self._synthesizer:
            structlogger.debug(
                "azure_tts.stop_streaming._synthesizer.stop_speaking_async",
            )
            self._synthesizer.stop_speaking_async()

    def _get_azure_audio_format(self) -> speechsdk.SpeechSynthesisOutputFormat:
        if self.audio_format == MULAW_8KHZ:
            return SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw
        elif self.audio_format == L16_24KHZ:
            return SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
        elif self.audio_format == L16_48KHZ:
            return SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm

        raise ValueError(f"Azure TTS does not support audio format {self.audio_format}")

    async def set_language(self, rasa_language: str) -> None:
        """Update the TTS language for next synthesis"""
        await super().set_language(rasa_language)

        # need to reconnect to apply new language
        await self.close_connection()
        await self.connect()
