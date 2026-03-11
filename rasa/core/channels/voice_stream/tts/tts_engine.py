from dataclasses import dataclass
from typing import AsyncIterator, Dict, Generic, List, Optional, Tuple, Type, TypeVar

import structlog

from rasa.core.channels.voice_stream.audio_bytes import (
    AudioFormat,
    CurrentLanguageConfig,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.tts.config import StreamingConfig
from rasa.core.channels.voice_stream.util import MergeableConfig
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.common import validate_environment

logger = structlog.get_logger(__name__)


class TTSError(RasaException):
    pass


class TTSConfigError(RasaException):
    """Error raised when TTS configuration is invalid."""

    pass


T = TypeVar("T", bound="TTSEngineConfig")
E = TypeVar("E", bound="TTSEngine")
L = TypeVar("L", bound="TTSLanguageMapEntry")


@dataclass
class TTSLanguageMapEntry:
    """Entry in the language_map mapping Rasa language to TTS settings.
    Usually a TTS Engine will require at least a language code and voice
    identifier to be able to synthesize speech. This class can be extended with
    additional fields as needed for specific engines.

    Attributes:
        language: Optional TTS-specific language code (e.g., 'en-US' for Azure).
        voice: Optional TTS-specific voice identifier.
        model: Optional model identifier for engines that support multiple models.
    """

    language: Optional[str] = None
    voice: Optional[str] = None
    model: Optional[str] = None


@dataclass
class TTSEngineConfig(MergeableConfig):
    """Base configuration for TTS engines.

    Attributes:
        language: (deprecated) TTS language code.
        voice: (deprecated) TTS voice identifier.
        timeout: Request timeout in seconds.
        language_map: Maps Rasa language codes to TTS-specific settings.
            Each TTS engine should provide sensible defaults.
    """

    language: Optional[str] = None
    voice: Optional[str] = None
    timeout: int = 30
    language_map: Optional[Dict[str, TTSLanguageMapEntry]] = None

    @classmethod
    def from_dict(cls: Type["TTSEngineConfig"], data: dict) -> "TTSEngineConfig":
        """Create config from dict, converting language_map entries."""
        if data.get("language_map"):
            data = {**data}  # shallow copy to avoid mutating the original
            data["language_map"] = {
                k: TTSLanguageMapEntry(**v) if isinstance(v, dict) else v
                for k, v in data["language_map"].items()
            }
        return cls(**data)

    def validate_language_map_keys(
        self,
        rasa_language: Optional[str],
        additional_languages: Optional[List[str]] = None,
    ) -> None:
        """Validate that language_map is configured and contains valid languages.

        Checks that:
        1. language_map is not empty
        2. rasa_language is provided
        3. rasa_language exists in language_map
        4. All language_map keys are in the set of allowed languages
           (rasa_language + additional_languages)
        """
        if not self.language_map:
            raise TTSConfigError(
                "TTS configuration requires 'language_map' to be set. "
                "The language_map should map Rasa language codes to TTS settings."
            )

        if rasa_language is None:
            raise TTSConfigError(
                "A language key must be provided in config.yml, "
                "this key is used to determine which language and "
                "voice to use from the language_map."
            )

        if rasa_language not in self.language_map:
            available_languages = list(self.language_map.keys())
            raise TTSConfigError(
                f"Language '{rasa_language}' not found in language_map. "
                f"Available languages: {available_languages}. "
                f"Please add '{rasa_language}' to the language_map configuration."
            )

        # Validate that language_map keys are subset of allowed languages
        allowed_languages = {rasa_language}
        if additional_languages:
            allowed_languages.update(additional_languages)

        language_map_keys = set(self.language_map.keys())
        invalid_keys = language_map_keys - allowed_languages
        if invalid_keys:
            raise TTSConfigError(
                f"language_map contains invalid language keys: {sorted(invalid_keys)}. "
                f"Allowed languages are: {sorted(allowed_languages)}. "
                f"These must match 'language' and 'additional_languages' in config.yml."
            )


class TTSEngine(Generic[T]):
    required_env_vars: Tuple[str, ...] = ()
    required_packages: Tuple[str, ...] = ()

    # If TTS supports input text streaming
    streaming_input: bool = False

    # Runtime language/model state
    current_language_config: CurrentLanguageConfig

    @classmethod
    def name(cls) -> str:
        raise NotImplementedError(
            "Subclasses must implement name() method to return engine name."
        )

    def __init__(
        self,
        rasa_language: str,
        format: AudioFormat,
        config: Optional[T] = None,
        additional_languages: Optional[List[str]] = None,
    ):
        self.audio_format = format
        self.config = self.get_default_config().merge(config)
        self.config.validate_language_map_keys(rasa_language, additional_languages)
        validate_environment(
            self.required_env_vars,
            self.required_packages,
            f"TTS Engine {self.__class__.__name__}",
        )
        self._set_current_language_config(rasa_language)

    async def prepare_response(
        self, streaming_config: Optional[StreamingConfig] = None
    ) -> None:
        """Called before a streaming response begins.

        Engines can override to adjust per-response behavior
        (e.g., switching between streaming and non-streaming modes).
        """
        pass

    async def connect(self, config: Optional[T] = None) -> None:
        """Establish connection to the TTS engine if necessary."""
        return

    async def close_connection(self) -> None:
        """Cleanup the connection if necessary."""
        return

    async def send_text_chunk(self, text: str) -> None:
        """Send text chunks to the TTS system.

        This method sends text to the TTS engine but does not return audio.
        Audio should be retrieved via stream_audio().

        Subclasses should implement this to send text to their specific engine.
        """
        pass

    async def signal_text_done(self) -> None:
        """Signal TTS engine to process any remaining buffered text.

        This tells the TTS engine that all text has been sent and to finish
        processing and prepare to end the stream.

        Returns:
            bool: Whether the engine supports streaming responses.
        """
        pass

    async def stream_audio(self) -> AsyncIterator[RasaAudioBytes]:
        """Stream audio output from the TTS engine.

        This continuously yields audio chunks as they are produced by the engine.
        Used in conjunction with send_text_chunk() for streaming responses.
        """
        yield RasaAudioBytes(b"", format=self.audio_format)

    async def synthesize(
        self, text: str, config: Optional[T] = None
    ) -> AsyncIterator[RasaAudioBytes]:
        """Generate speech from text using a remote TTS system."""
        yield RasaAudioBytes(b"", format=self.audio_format)

    def engine_bytes_to_rasa_audio_bytes(self, chunk: bytes) -> RasaAudioBytes:
        """Convert the generated TTS audio bytes into rasa audio bytes."""
        raise NotImplementedError

    async def signal_interrupt(self) -> None:
        """Cancel the TTS engine buffer."""
        pass

    @staticmethod
    def get_default_config() -> T:
        """Get the default config for this component."""
        raise NotImplementedError

    @classmethod
    def from_config_dict(
        cls: Type[E],
        config: Dict,
        format: AudioFormat,
        rasa_language: str,
        additional_languages: Optional[List[str]] = None,
    ) -> E:
        raise NotImplementedError

    async def set_language(self, rasa_language: str) -> None:
        """Update the TTS language for subsequent synthesis calls.

        Called by the voice channel when the language slot changes.

        Args:
            rasa_language: Value of Rasa's language slot.
        """
        if self.current_language_config.is_same_language(rasa_language):
            return

        old_language_config = self.current_language_config
        self._set_current_language_config(rasa_language)
        logger.info(
            f"tts.{self.name()}.language_changed",
            before=old_language_config,
            after=self.current_language_config,
        )

    def _set_current_language_config(self, rasa_language: str) -> None:
        """Helper method to set the current language configuration."""
        assert (
            self.config.language_map is not None
        ), "language_map must be set in config"
        try:
            entry = self.config.language_map[rasa_language]
        except KeyError:
            logger.error(
                f"tts.{self.name()}.language_not_in_map",
                language=rasa_language,
                available_languages=list(self.config.language_map.keys()),
            )
            return
        self.current_language_config = CurrentLanguageConfig(
            rasa_language_key=rasa_language,
            engine_language_key=entry.language,
            voice=entry.voice,
            model=entry.model,
        )

    async def stop_streaming(self) -> None:
        """Clear the TTS engine buffer."""
        pass
