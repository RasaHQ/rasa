from rasa.core.channels.voice_stream.tts.tts_engine import (
    TTSLanguageMapEntry,
    TTSEngine,
    TTSEngineConfig,
)
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class CustomTTSConfig(TTSEngineConfig):
    server_url: Optional[str] = None

class CustomTTSEngine(TTSEngine[CustomTTSConfig]):
    def __init__(
        self,
        rasa_language: str,
        config: Optional[CustomTTSConfig] = None,
        additional_languages: Optional[List[str]] = None,
    ) -> None:
        super().__init__(rasa_language, config, additional_languages)

    @staticmethod
    def get_default_config() -> CustomTTSConfig:
        return CustomTTSConfig(
            server_url="http://localhost:5000",
            language_map={
                "en": TTSLanguageMapEntry(
                    language="en",
                    voice="nova-2-general",
                ),
            },
        )

    @classmethod
    def from_config_dict(
        cls,
        config: Dict,
        rasa_language: str,
        additional_languages: Optional[List[str]] = None,
    ) -> "CustomTTSEngine":
        return CustomTTSEngine(
            rasa_language,
            CustomTTSConfig.from_dict(config),
            additional_languages,
        )
