from rasa.core.channels.voice_stream.asr.asr_engine import ASREngine, ASREngineConfig
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class CustomASRConfig(ASREngineConfig):
    endpoint: Optional[str] = None
    language: Optional[str] = None

class CustomASREngine(ASREngine[CustomASRConfig]):
    def __init__(self, config: CustomASRConfig) -> None:
        super().__init__(config)

    @staticmethod
    def get_default_config() -> CustomASRConfig:
        return CustomASRConfig(endpoint="en")

    @classmethod
    def from_config_dict(cls, config: Dict) -> "CustomASREngine":
        return CustomASREngine(CustomASRConfig.from_dict(config))
