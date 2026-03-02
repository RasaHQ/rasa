import pytest

from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR
from rasa.core.channels.voice_stream.audio_bytes import L16_24KHZ, L16_48KHZ, MULAW_8KHZ
from rasa.core.channels.voice_stream.tts.cartesia import CartesiaTTS
from tests.core.channels.voice_stream.tts.test_tts import (
    run_single_utterance_through_tts_and_asr,
)


@pytest.mark.parametrize(
    "format",
    [
        MULAW_8KHZ,
        L16_24KHZ,
        L16_48KHZ,
    ],
)
async def test_synthesis_with_asr(format):
    tts_engine = CartesiaTTS(rasa_language="en", format=format)
    text = "hello my name is Edgar"
    asr_engine = DeepgramASR(rasa_language="en", format=format)

    await run_single_utterance_through_tts_and_asr(text, asr_engine, tts_engine, format)
