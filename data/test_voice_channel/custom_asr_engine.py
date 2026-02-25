from rasa.core.channels.voice_stream.asr.asr_engine import (
    ASREngine,
    ASREngineConfig,
    ASRLanguageMapEntry,
)
from typing import Optional, Dict, List

class CustomASRConfig(ASREngineConfig):
    endpoint: Optional[str] = None
    language: Optional[str] = None

class CustomASREngine(ASREngine[CustomASRConfig]):
    def __init__(
        self,
        rasa_language: str,
        config: Optional[CustomASRConfig] = None,
        additional_languages: Optional[List[str]] = None,
    ) -> None:
        super().__init__(rasa_language, config, additional_languages)

    @staticmethod
    def get_default_config() -> CustomASRConfig:
        return CustomASRConfig(
            endpoint="en",
            language_map={
                "en": ASRLanguageMapEntry(
                    language="en",
                ),
            },
        )

    @classmethod
    def from_config_dict(
        cls,
        config: Dict,
        rasa_language: str,
        additional_languages: Optional[List[str]] = None,
    ) -> "CustomASREngine":
        return CustomASREngine(
            rasa_language,
            CustomASRConfig(**config),
            additional_languages,
        )
