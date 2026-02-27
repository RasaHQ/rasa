import asyncio
import os
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

import structlog

from rasa.core.channels.voice_stream.asr.asr_engine import (
    ASREngine,
    ASREngineConfig,
    ASRLanguageMapEntry,
)
from rasa.core.channels.voice_stream.asr.asr_event import (
    ASREvent,
    NewTranscript,
    UserIsSpeaking,
)
from rasa.core.channels.voice_stream.audio_bytes import (
    AudioFormat,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.tts.tts_engine import TTSConfigError
from rasa.shared.constants import AZURE_SPEECH_API_KEY_ENV_VAR
from rasa.shared.exceptions import ConnectionException

if TYPE_CHECKING:
    from azure.cognitiveservices.speech import SpeechRecognitionEventArgs

logger = structlog.get_logger(__name__)


class AzureASRConfig(ASREngineConfig):
    """Configuration for Azure ASR.

    Attributes:
        language: ASR language code (typically set from language_map).
        speech_region: Azure speech service region.
        speech_host: Custom host URL.
        speech_endpoint: Custom endpoint URL.
        model: Optional model identifier.
    """

    language: Optional[str] = None
    model: Optional[str] = None
    speech_region: Optional[str] = None
    speech_host: Optional[str] = None
    speech_endpoint: Optional[str] = None


class AzureASR(ASREngine[AzureASRConfig]):
    required_env_vars = (AZURE_SPEECH_API_KEY_ENV_VAR,)
    required_packages = ("azure.cognitiveservices.speech",)

    @classmethod
    def name(cls) -> str:
        return "azure"

    def __init__(
        self,
        rasa_language: str,
        format: AudioFormat,
        config: Optional[AzureASRConfig] = None,
        additional_languages: Optional[List[str]] = None,
    ):
        super().__init__(rasa_language, format, config, additional_languages)

        import azure.cognitiveservices.speech as speechsdk

        self.speech_recognizer: Optional[speechsdk.SpeechRecognizer] = None
        self.stream: Optional[speechsdk.audio.PushAudioInputStream] = None
        self.is_recognizing = False
        self.queue: asyncio.Queue[speechsdk.SpeechRecognitionEventArgs] = (
            asyncio.Queue()
        )
        self.main_loop = asyncio.get_running_loop()

    def signal_user_is_speaking(self, event: "SpeechRecognitionEventArgs") -> None:
        """Replace the azure event with a generic is speaking event."""
        self.fill_queue(UserIsSpeaking(event.result.text))

    def fill_queue(self, event: Any) -> None:
        """Either puts the event or a dedicated ASR Event into the queue."""
        # This function is used by call backs of the azure speech library
        # which seems to run separate threads/processes
        # To properly wake up the task waiting at queue.get, we need to
        # put to the queue in the same event loop
        self.main_loop.call_soon_threadsafe(self.queue.put_nowait, event)

    async def connect(self) -> None:
        """Connect to Azure ASR service and set up callbacks."""

        # Validate that at least one of the connection parameters is set
        if not (
            self.config.speech_region
            or self.config.speech_host
            or self.config.speech_endpoint
        ):
            raise TTSConfigError(
                f"Azure ASR config requires at least one of speech_region "
                f"({self.config.speech_region}), "
                f"speech_host ({self.config.speech_host}), "
                f"or speech_endpoint ({self.config.speech_endpoint}) to be set."
            )

        # Set up Azure Speech SDK configuration
        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ[AZURE_SPEECH_API_KEY_ENV_VAR],
            region=self.config.speech_region,
            endpoint=self.config.speech_endpoint,
            host=self.config.speech_host,
        )
        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self.audio_format.sample_rate,
            bits_per_sample=self.audio_format.bit_depth,
            channels=self.audio_format.channels,
            wave_stream_format=speechsdk.AudioStreamWaveFormat.MULAW,
        )
        self.stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=self.stream)
        self.speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            language=self.current_language_config.engine_language_key,
            audio_config=audio_config,
        )
        self.speech_recognizer.recognized.connect(self.fill_queue)
        self.speech_recognizer.recognizing.connect(self.signal_user_is_speaking)
        self.speech_recognizer.start_continuous_recognition_async()
        self.is_recognizing = True

    async def close_connection(self) -> None:
        if self.speech_recognizer is None:
            raise ConnectionException("Websocket not connected.")
        self.speech_recognizer.stop_continuous_recognition_async()

    async def signal_audio_done(self) -> None:
        """Signal to the ASR Api that you are done sending data."""
        self.is_recognizing = False

    def rasa_audio_bytes_to_engine_bytes(self, chunk: RasaAudioBytes) -> bytes:
        """Convert RasaAudioBytes to bytes usable by this engine."""
        return chunk.data

    async def send_audio_chunks(self, chunk: RasaAudioBytes) -> None:
        """Send audio chunks to the ASR system via the websocket."""
        if self.speech_recognizer is None or self.stream is None:
            raise ConnectionException("ASR not connected.")
        engine_bytes = self.rasa_audio_bytes_to_engine_bytes(chunk)
        self.stream.write(engine_bytes)

    async def stream_asr_events(self) -> AsyncIterator[ASREvent]:
        """Stream the events returned by the ASR system as it is fed audio bytes."""
        if self.speech_recognizer is None:
            raise ConnectionException("Websocket not connected.")
        while self.is_recognizing or not self.queue.empty():
            try:
                message = await asyncio.wait_for(self.queue.get(), timeout=2)
                asr_event = self.engine_event_to_asr_event(message)
                if asr_event:
                    yield asr_event
            except asyncio.TimeoutError:
                pass

    def engine_event_to_asr_event(self, e: Any) -> Optional[ASREvent]:
        """Translate an engine event to a common ASREvent."""
        import azure.cognitiveservices.speech as speechsdk

        if isinstance(e, speechsdk.SpeechRecognitionEventArgs) and isinstance(
            e.result, speechsdk.SpeechRecognitionResult
        ):
            return NewTranscript(e.result.text)
        if isinstance(e, ASREvent):
            # transformation happened before
            return e

        return None

    @staticmethod
    def get_default_config() -> AzureASRConfig:
        return AzureASRConfig(
            language_map={
                "en": ASRLanguageMapEntry(
                    language="en-US",
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
    ) -> "AzureASR":
        return cls(
            rasa_language=rasa_language,
            format=format,
            config=AzureASRConfig(**config),
            additional_languages=additional_languages,
        )
