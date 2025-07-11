from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import structlog
from websockets.legacy.client import WebSocketClientProtocol

from rasa.core.channels.voice_stream.asr.asr_event import ASREvent
from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.util import MergeableConfig
from rasa.shared.exceptions import ConnectionException
from rasa.shared.utils.common import validate_environment

T = TypeVar("T", bound="ASREngineConfig")
E = TypeVar("E", bound="ASREngine")
logger = structlog.get_logger(__name__)


@dataclass
class ASREngineConfig(MergeableConfig):
    keep_alive_interval: int = 5  # seconds


class ASREngine(Generic[T]):
    required_env_vars: Tuple[str, ...] = ()
    required_packages: Tuple[str, ...] = ()

    def __init__(self, config: Optional[T] = None):
        self.config = self.get_default_config().merge(config)
        self.asr_socket: Optional[WebSocketClientProtocol] = None
        validate_environment(
            self.required_env_vars,
            self.required_packages,
            f"ASR Engine {self.__class__.__name__}",
        )

    async def connect(self) -> None:
        self.asr_socket = await self.open_websocket_connection()

    async def open_websocket_connection(self) -> WebSocketClientProtocol:
        """Connect to the ASR system."""
        raise NotImplementedError

    @classmethod
    def from_config_dict(cls: Type[E], config: Dict) -> E:
        raise NotImplementedError

    async def close_connection(self) -> None:
        if self.asr_socket:
            await self.asr_socket.close()

    async def signal_audio_done(self) -> None:
        """Signal to the ASR Api that you are done sending data."""
        raise NotImplementedError

    async def send_audio_chunks(self, chunk: RasaAudioBytes) -> None:
        """Send audio chunks to the ASR system via the websocket."""
        if self.asr_socket is None:
            raise ConnectionException("Websocket not connected.")
        engine_bytes = self.rasa_audio_bytes_to_engine_bytes(chunk)
        await self.asr_socket.send(engine_bytes)

    def rasa_audio_bytes_to_engine_bytes(self, chunk: RasaAudioBytes) -> bytes:
        """Convert RasaAudioBytes to bytes usable by this engine."""
        raise NotImplementedError

    async def stream_asr_events(self) -> AsyncIterator[ASREvent]:
        """Stream the events returned by the ASR system as it is fed audio bytes."""
        if self.asr_socket is None:
            raise ConnectionException("Websocket not connected.")

        try:
            async for message in self.asr_socket:
                asr_event = self.engine_event_to_asr_event(message)
                if asr_event:
                    yield asr_event
        except Exception as e:
            logger.warning(f"Error while streaming ASR events: {e}")

    def engine_event_to_asr_event(self, e: Any) -> Optional[ASREvent]:
        """Translate an engine event to a common ASREvent."""
        raise NotImplementedError

    @staticmethod
    def get_default_config() -> T:
        """Get the default config for this component."""
        raise NotImplementedError

    async def send_keep_alive(self) -> None:
        """Send a keep-alive message to the ASR system if supported."""
        pass
