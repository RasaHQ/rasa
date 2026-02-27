from collections import OrderedDict
from typing import Optional

from rasa.core.channels.voice_stream.audio_bytes import AudioFormat, RasaAudioBytes


class TTSCache:
    """An LRU Cache for TTS based on pythons OrderedDict."""

    def __init__(self, max_size: int):
        self.cache: OrderedDict[tuple[str, AudioFormat], RasaAudioBytes] = OrderedDict()
        self.max_size = max_size

    def _key(self, text: str, fmt: AudioFormat) -> tuple[str, AudioFormat]:
        return (text, fmt)

    def get(self, text: str, fmt: AudioFormat) -> Optional[RasaAudioBytes]:
        key = self._key(text, fmt)
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, text: str, bytes: RasaAudioBytes) -> None:
        key = self._key(text, bytes.format)
        self.cache[key] = bytes
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
