from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class AudioEncoding(Enum):
    # PCM (Pulse Code Modulation) reprents raw audio data
    # Each sample can be encoded linearly (e.g., 16-bit) or
    # with a companding algorithm like μ-law or A-law

    LINEAR = "linear"
    MULAW = "mulaw"
    # A-Law isn't currently supported.


@dataclass(frozen=True)
class AudioFormat:
    encoding: AudioEncoding
    bit_depth: int  # bits per sample
    sample_rate: int  # Hertz
    channels: int = 1  # default to mono

    @property
    def bytes_per_second(self) -> int:
        """For uncompressed formats: usable for duration math and rate signalling."""
        return self.sample_rate * (self.bit_depth // 8)


# Rasa supported Audio Formats
# MULAW_8KHZ aka G.711 μ-law is raw wave, 8kHz, 8bit, mono channel, mulaw encoding
MULAW_8KHZ = AudioFormat(AudioEncoding.MULAW, sample_rate=8000, bit_depth=8)
# L16 or Linear 16 is a general term for Linear PCM 16-bit Encoding
L16_24KHZ = AudioFormat(AudioEncoding.LINEAR, sample_rate=24000, bit_depth=16)
L16_48KHZ = AudioFormat(AudioEncoding.LINEAR, sample_rate=48000, bit_depth=16)


@dataclass
class RasaAudioBytes:
    data: bytes
    format: AudioFormat

    def full_seconds(self) -> float:
        """Calculate the duration of this audio chunk in seconds."""
        return len(self.data) / self.format.bytes_per_second

    def __add__(self, other: "RasaAudioBytes") -> "RasaAudioBytes":
        """Combine two RasaAudioBytes as long as their formats match."""
        if not isinstance(other, RasaAudioBytes):
            raise ValueError("Can only add RasaAudioBytes to RasaAudioBytes.")
        if self.format != other.format:
            raise ValueError("Cannot add RasaAudioBytes with different formats.")
        combined_data = self.data + other.data
        return RasaAudioBytes(combined_data, format=self.format)

    def __len__(self) -> int:
        """Return the length of the audio data in bytes."""
        return len(self.data)

    def __getitem__(self, key: Any) -> "RasaAudioBytes":
        """Allow slicing and indexing"""
        return RasaAudioBytes(self.data[key], format=self.format)


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
