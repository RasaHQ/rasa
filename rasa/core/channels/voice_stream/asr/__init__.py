from typing import Callable, Dict, Type

from rasa.core.channels.voice_stream.asr.asr_engine import ASREngine


def load_azure() -> Type[ASREngine]:
    from rasa.core.channels.voice_stream.asr.azure import AzureASR

    return AzureASR


def load_deepgram() -> Type[ASREngine]:
    from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR

    return DeepgramASR


BUILT_IN_ASR_ENGINES: Dict[str, Callable[..., Type[ASREngine]]] = {
    "azure": load_azure,
    "deepgram": load_deepgram,
}
