import audioop
import wave
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Type, TypeVar

import structlog

from rasa.core.channels.voice_stream.audio_bytes import HERTZ, RasaAudioBytes
from rasa.shared.exceptions import RasaException

structlogger = structlog.get_logger()


def read_wav_to_rasa_audio_bytes(file_name: str) -> Optional[RasaAudioBytes]:
    """Reads rasa audio bytes from a file."""
    if not file_name.endswith(".wav"):
        raise RasaException("Should only read .wav files with this method.")
    wave_object = wave.open(file_name, "rb")
    wave_data = wave_object.readframes(wave_object.getnframes())
    if wave_object.getnchannels() != 1:
        wave_data = audioop.tomono(wave_data, wave_object.getsampwidth(), 1, 1)
    if wave_object.getsampwidth() != 1:
        wave_data = audioop.lin2lin(wave_data, wave_object.getsampwidth(), 1)
        # 8 bit is unsigned
        # wave_data = audioop.bias(wave_data, 1, 128)
    if wave_object.getframerate() != HERTZ:
        wave_data, _ = audioop.ratecv(
            wave_data, 1, 1, wave_object.getframerate(), HERTZ, None
        )
    wave_data = audioop.lin2ulaw(wave_data, 1)
    return RasaAudioBytes(wave_data)


def generate_silence(length_in_seconds: float = 1.0) -> RasaAudioBytes:
    return RasaAudioBytes(b"\00" * int(length_in_seconds * HERTZ))


T = TypeVar("T", bound="MergeableConfig")


@dataclass
class MergeableConfig:
    def __init__(self) -> None:
        pass

    def merge(self: T, other: Optional[T]) -> T:
        """Merges two configs while dropping None values of the second config."""
        if other is None:
            return self
        other_dict = asdict(other)
        other_dict_clean = {k: v for k, v in other_dict.items() if v is not None}
        merged = {**asdict(self), **other_dict_clean}
        return self.from_dict(merged)

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Optional[str]]) -> T:
        return cls(**data)


def repack_voice_credentials(
    credentials: Dict[str, str],
) -> Dict[str, str]:
    """Repack voice credentials to ensure they are in the correct format."""
    new_creds = {**credentials}
    new_creds["asr_config"] = new_creds.pop("asr", None)
    new_creds["tts_config"] = new_creds.pop("tts", None)
    return new_creds
