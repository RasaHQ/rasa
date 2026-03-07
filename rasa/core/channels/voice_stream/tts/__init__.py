from typing import Callable, Dict, Type

from rasa.core.channels.voice_stream.tts.tts_cache import TTSCache
from rasa.core.channels.voice_stream.tts.tts_engine import (
    TTSConfigError,
    TTSEngine,
    TTSEngineConfig,
    TTSError,
)

__all__ = ["TTSConfigError", "TTSEngine", "TTSEngineConfig", "TTSError", "TTSCache"]


def load_azure() -> Type[TTSEngine]:
    from rasa.core.channels.voice_stream.tts.azure import AzureTTS

    return AzureTTS


def load_cartesia() -> Type[TTSEngine]:
    from rasa.core.channels.voice_stream.tts.cartesia import CartesiaTTS

    return CartesiaTTS


def load_deepgram() -> Type[TTSEngine]:
    from rasa.core.channels.voice_stream.tts.deepgram import DeepgramTTS

    return DeepgramTTS


def load_rime() -> Type[TTSEngine]:
    from rasa.core.channels.voice_stream.tts.rime import RimeTTS

    return RimeTTS


BUILT_IN_TTS_ENGINES: Dict[str, Callable[..., Type[TTSEngine[TTSEngineConfig]]]] = {
    "azure": load_azure,
    "cartesia": load_cartesia,
    "deepgram": load_deepgram,
    "rime": load_rime,
}
