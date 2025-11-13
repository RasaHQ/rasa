from dataclasses import dataclass
from typing import AsyncIterator, Dict, Generic, Optional, Tuple, Type, TypeVar

from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.util import MergeableConfig
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.common import validate_environment


class TTSError(RasaException):
    pass


T = TypeVar("T", bound="TTSEngineConfig")
E = TypeVar("E", bound="TTSEngine")


@dataclass
class TTSEngineConfig(MergeableConfig):
    language: Optional[str] = None
    voice: Optional[str] = None
    timeout: Optional[int] = None


class TTSEngine(Generic[T]):
    required_env_vars: Tuple[str, ...] = ()
    required_packages: Tuple[str, ...] = ()
    streaming_input: bool = False

    def __init__(self, config: Optional[T] = None):
        self.config = self.get_default_config().merge(config)
        validate_environment(
            self.required_env_vars,
            self.required_packages,
            f"TTS Engine {self.__class__.__name__}",
        )

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
        yield RasaAudioBytes(b"")

    async def synthesize(
        self, text: str, config: Optional[T] = None
    ) -> AsyncIterator[RasaAudioBytes]:
        """Generate speech from text using a remote TTS system."""
        yield RasaAudioBytes(b"")

    def engine_bytes_to_rasa_audio_bytes(self, chunk: bytes) -> RasaAudioBytes:
        """Convert the generated TTS audio bytes into rasa audio bytes."""
        raise NotImplementedError

    @staticmethod
    def get_default_config() -> T:
        """Get the default config for this component."""
        raise NotImplementedError

    @classmethod
    def from_config_dict(cls: Type[E], config: Dict) -> E:
        raise NotImplementedError
