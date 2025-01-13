import logging
from collections import OrderedDict
from typing import Optional

from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes

logger = logging.getLogger(__name__)


class TTSCache:
    """An LRU Cache for TTS based on pythons OrderedDict."""

    def __init__(self, max_size: int):
        self.cache: OrderedDict[str, RasaAudioBytes] = OrderedDict()
        self.max_size = max_size

    def get(self, text: str) -> Optional[RasaAudioBytes]:
        if text not in self.cache:
            return None
        else:
            self.cache.move_to_end(text)
            return self.cache[text]

    def put(self, text: str, audio_bytes: RasaAudioBytes) -> None:
        self.cache[text] = audio_bytes
        self.cache.move_to_end(text)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
