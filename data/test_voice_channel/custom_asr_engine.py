from rasa.core.channels.voice_stream.asr.asr_engine import (
    ASREngine,
    ASREngineConfig,
    ASRLanguageMapEntry,
)
from rasa.core.channels.voice_stream.audio_bytes import AudioFormat
from typing import Optional, Dict, List

class CustomASRConfig(ASREngineConfig):
    endpoint: Optional[str] = None
    language: Optional[str] = None

class CustomASREngine(ASREngine[CustomASRConfig]):
    def __init__(
        self,
        rasa_language: str,
        format: AudioFormat,
        config: Optional[CustomASRConfig] = None,
        additional_languages: Optional[List[str]] = None,
    ) -> None:
        super().__init__(rasa_language=rasa_language, format=format, config=config, additional_languages=additional_languages)

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
        format: AudioFormat,
        rasa_language: str,
        additional_languages: Optional[List[str]] = None,
    ) -> "CustomASREngine":
        return CustomASREngine(
            rasa_language=rasa_language,
            format=format,
            config=CustomASRConfig(**config),
            additional_languages=additional_languages,
        )
