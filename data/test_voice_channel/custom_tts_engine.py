from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine, TTSEngineConfig
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class CustomTTSConfig(TTSEngineConfig):
    server_url: Optional[str] = None

class CustomTTSEngine(TTSEngine[CustomTTSConfig]):
    def __init__(self, config: CustomTTSConfig) -> None:
        super().__init__(config)

    @staticmethod
    def get_default_config() -> CustomTTSConfig:
        return CustomTTSConfig(language="en", voice="nova-2-general", server_url="http://localhost:5000")

    @classmethod
    def from_config_dict(cls, config: Dict) -> "CustomTTSEngine":
        return CustomTTSEngine(CustomTTSConfig.from_dict(config))
