from dataclasses import dataclass
from typing import NewType, Optional


# Used for runtime language/model state in ASR/TTS engines
@dataclass
class CurrentLanguageConfig:
    # The Rasa language code currently being processed (e.g., 'en', 'de').
    # These are defined by the builder in Rasa Config
    rasa_language_key: str

    # Vendor specific language/model settings currently being used
    # These are typically set from the language_map based on rasa_language_key
    engine_language_key: Optional[str] = None
    voice: Optional[str] = None
    model: Optional[str] = None

    def is_same_language(self, rasa_language: str) -> bool:
        """Check if the given Rasa language is the same as the current one."""
        return self.rasa_language_key == rasa_language

    def __repr__(self) -> str:
        return (
            f"CurrentLanguageConfig(rasa_language_key='{self.rasa_language_key}'"
            + (
                f", engine_language_key='{self.engine_language_key}'"
                if self.engine_language_key
                else ""
            )
            + (f", voice='{self.voice}'" if self.voice else "")
            + (f", model='{self.model}'" if self.model else "")
            + ")"
        )


# a common intermediate audio byte format that acts as a common data format,
# to prevent quadratic complexity between formats of channels, asr engines,
# and tts engines
# currently corresponds to raw wave, 8khz, 8bit, mono channel, mulaw encoding
RasaAudioBytes = NewType("RasaAudioBytes", bytes)
HERTZ = 8000
