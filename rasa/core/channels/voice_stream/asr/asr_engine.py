import warnings
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import structlog
from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator
from websockets.legacy.client import WebSocketClientProtocol

from rasa.core.channels.voice_stream.asr.asr_event import ASREvent
from rasa.core.channels.voice_stream.audio_bytes import (
    AudioFormat,
    CurrentLanguageConfig,
    RasaAudioBytes,
)
from rasa.shared.exceptions import ConnectionException, RasaException
from rasa.shared.utils.common import validate_environment

T = TypeVar("T", bound="ASREngineConfig")
E = TypeVar("E", bound="ASREngine")
L = TypeVar("L", bound="ASRLanguageMapEntry")
logger = structlog.get_logger(__name__)


class ASRConfigError(RasaException):
    """Error raised when ASR configuration is invalid."""

    pass


class ASRLanguageMapEntry(BaseModel):
    """Entry in the language_map mapping Rasa language to ASR settings.

    Attributes:
        language: ASR-specific language code (e.g., 'en-US' for Azure).
        model: Optional model identifier for engines that support multiple models.
    """

    language: str
    model: Optional[str] = None


class ASREngineConfig(BaseModel):
    """Base configuration for ASR engines.

    Attributes:
        keep_alive_interval: Interval in seconds for keep-alive messages.
        language_map: Maps Rasa language codes to ASR-specific settings.
            Each ASR engine should provide sensible defaults via get_default_config().
        language: Deprecated. Use ``language_map`` instead.
        model: Deprecated. Use ``language_map`` instead.
    """

    # Deprecated: set these inside language_map instead
    language: Optional[str] = None
    model: Optional[str] = None
    keep_alive_interval: int = 5
    language_map: Optional[Dict[str, ASRLanguageMapEntry]] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_language_fields(self, info: ValidationInfo) -> "ASREngineConfig":
        """Validate mutual exclusivity of deprecated fields and language_map.

        Skipped during merge(), which intentionally combines a user config
        (potentially with deprecated fields) with a default config (which always
        has language_map). apply_deprecated_fields() folds them together after.
        """
        if info.context and info.context.get("merging"):
            return self

        has_deprecated = self.language is not None or self.model is not None
        has_language_map = self.language_map is not None

        if has_deprecated and has_language_map:
            raise ValueError(
                "Cannot specify both top-level 'language'/'model' and 'language_map'. "
                "Remove 'language' and 'model' and configure 'language_map' only."
            )

        if has_deprecated:
            used_fields = sorted(
                f for f in ("language", "model") if getattr(self, f) is not None
            )
            field_list = ", ".join(f"'{f}'" for f in used_fields)
            warnings.warn(
                f"Top-level ASR config field(s) {field_list} are deprecated. "
                "Use 'language_map' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        return self

    def merge(self: T, other: Optional[T]) -> T:
        """Merges two configs while dropping None values of the second config."""
        if other is None:
            return self
        other_dict = other.model_dump()
        other_dict_clean = {k: v for k, v in other_dict.items() if v is not None}
        merged = {**self.model_dump(), **other_dict_clean}
        return self.__class__.model_validate(merged, context={"merging": True})

    def apply_deprecated_fields(self, rasa_language: str) -> "ASREngineConfig":
        """Fold top-level deprecated fields into language_map[rasa_language].

        Moves ``language`` or ``model`` values set at the top level into the
        language_map entry for ``rasa_language``. The DeprecationWarning is
        emitted earlier, during model validation.
        """
        updates = {
            k: v
            for k, v in [("language", self.language), ("model", self.model)]
            if v is not None
        }
        if not updates:
            # No deprecated fields to apply, return self unmodified
            return self

        existing_map = dict(self.language_map or {})
        existing_entry = existing_map.get(rasa_language)
        if existing_entry is not None:
            existing_map[rasa_language] = existing_entry.model_copy(update=updates)
        else:
            # If ASR doesn't have a language map in default config
            existing_map[rasa_language] = ASRLanguageMapEntry(**updates)

        return self.model_copy(
            update={"language": None, "model": None, "language_map": existing_map}
        )

    def validate_language_map_keys(
        self,
        rasa_language: Optional[str],
        additional_languages: Optional[List[str]] = None,
    ) -> None:
        """Validate that rasa_language and allowed languages match language_map keys.

        Checks that:
        1. rasa_language is provided
        2. rasa_language exists in language_map
        3. All language_map keys are in the set of allowed languages
           (rasa_language + additional_languages)
        """
        if rasa_language is None:
            raise ASRConfigError(
                "A language key must be provided in config.yml, "
                "this key is used to determine which language "
                "to use from the language_map."
            )

        assert self.language_map is not None, "language_map must be set in config"
        if rasa_language not in self.language_map:
            available_languages = (
                list(self.language_map.keys()) if self.language_map else []
            )
            raise ASRConfigError(
                f"Language '{rasa_language}' not found in language_map. "
                f"Available languages: {available_languages}. "
                f"Please add '{rasa_language}' to the language_map configuration."
            )

        # Validate that language_map keys are subset of allowed languages
        allowed_languages = {rasa_language}
        if additional_languages:
            allowed_languages.update(additional_languages)

        language_map_keys = (
            set(self.language_map.keys()) if self.language_map else set()
        )
        invalid_keys = language_map_keys - allowed_languages
        if invalid_keys:
            raise ASRConfigError(
                f"language_map contains invalid language keys: {sorted(invalid_keys)}. "
                f"Allowed languages are: {sorted(allowed_languages)}. "
                f"These must match 'language' and 'additional_languages' in config.yml."
            )


class ASREngine(Generic[T]):
    config: T
    required_env_vars: Tuple[str, ...] = ()
    required_packages: Tuple[str, ...] = ()

    # runtime language config
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
        self.config = self.get_default_config(rasa_language).merge(config)
        self.config = cast(T, self.config.apply_deprecated_fields(rasa_language))
        self.config.validate_language_map_keys(rasa_language, additional_languages)
        self.asr_socket: Optional[WebSocketClientProtocol] = None
        validate_environment(
            self.required_env_vars,
            self.required_packages,
            f"ASR Engine {self.__class__.__name__}",
        )
        self._set_current_language_config(rasa_language)

    async def connect(self) -> None:
        self.asr_socket = await self.open_websocket_connection()

    async def open_websocket_connection(self) -> WebSocketClientProtocol:
        """Connect to the ASR system."""
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
    def get_default_config(rasa_language: str) -> T:
        """Get the default config for this component."""
        raise NotImplementedError

    async def send_keep_alive(self) -> None:
        """Send a keep-alive message to the ASR system if supported."""
        pass

    async def set_language(self, rasa_language: str) -> None:
        """Update the ASR language for the current session.

        Called by the voice channel when the language slot changes.
        Subclasses should implement engine-specific language switching logic.

        Args:
            rasa_language: Value of Rasa's language slot.
        """
        if self.current_language_config.is_same_language(rasa_language):
            return

        old_language_config = self.current_language_config
        self._set_current_language_config(rasa_language)
        logger.info(
            f"asr.{self.name()}.language_changed",
            before=old_language_config,
            after=self.current_language_config,
        )

        # reconnect asr with new language config
        await self.close_connection()
        await self.connect()

    def _set_current_language_config(self, rasa_language: str) -> None:
        """Helper method to set the current language configuration."""
        assert (
            self.config.language_map is not None
        ), "language_map must be set in config"
        try:
            entry = self.config.language_map[rasa_language]
        except KeyError:
            logger.error(
                f"asr.{self.name()}.set_language.language_not_found",
                rasa_language=rasa_language,
                available_languages=list(self.config.language_map.keys()),
            )
            return

        self.current_language_config = CurrentLanguageConfig(
            rasa_language_key=rasa_language,
            engine_language_key=entry.language,
            model=entry.model,
        )
