from typing import AsyncIterator, Dict, Generic, Optional, Type, TypeVar
from dataclasses import dataclass

from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.util import MergeableConfig
from rasa.shared.exceptions import RasaException


class TTSError(RasaException):
    pass


T = TypeVar("T", bound="TTSEngineConfig")
E = TypeVar("E", bound="TTSEngine")


@dataclass
class TTSEngineConfig(MergeableConfig):
    language: Optional[str] = None
    voice: Optional[str] = None


class TTSEngine(Generic[T]):
    def __init__(self, config: Optional[T] = None):
        self.config = self.get_default_config().merge(config)

    async def close_connection(self) -> None:
        """Cleanup the connection if necessary."""
        return

    async def synthesize(
        self, text: str, config: Optional[T] = None
    ) -> AsyncIterator[RasaAudioBytes]:
        """Generate speech from text using a remote TTS system."""
        yield RasaAudioBytes(b"")

    def engine_bytes_to_rasa_audio_bytes(self, chunk: bytes) -> RasaAudioBytes:
        """Convert the generated tts audio bytes into rasa audio bytes."""
        raise NotImplementedError

    @staticmethod
    def get_default_config() -> T:
        """Get the default config for this component."""
        raise NotImplementedError

    @classmethod
    def from_config_dict(cls: Type[E], config: Dict) -> E:
        raise NotImplementedError
