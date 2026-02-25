from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR
from rasa.core.channels.voice_stream.tts.cartesia import CartesiaTTS
from tests.core.channels.voice_stream.tts.test_tts import (
    run_single_utterance_through_tts_and_asr,
)


async def test_synthesis_with_asr():
    tts_engine = CartesiaTTS(rasa_language="en")
    text = "hello my name is Edgar"
    asr_engine = DeepgramASR(rasa_language="en")

    await run_single_utterance_through_tts_and_asr(text, asr_engine, tts_engine)
