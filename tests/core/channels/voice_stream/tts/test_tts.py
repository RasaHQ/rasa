import difflib

from rasa.core.channels.voice_stream.asr.asr_engine import ASREngine
from rasa.core.channels.voice_stream.asr.asr_event import NewTranscript
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine
from rasa.core.channels.voice_stream.util import generate_silence


async def run_single_utterance_through_tts_and_asr(
    text: str, asr_engine: ASREngine, tts_engine: TTSEngine, match_ratio: float = 0.75
):
    await asr_engine.connect()

    async for chunk in tts_engine.synthesize(text):
        await asr_engine.send_audio_chunks(chunk)
    await asr_engine.send_audio_chunks(generate_silence())
    await asr_engine.signal_audio_done()

    events = []
    async for event in asr_engine.stream_asr_events():
        events.append(event)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, NewTranscript)
    match = difflib.SequenceMatcher(None, event.text, text)
    assert match.ratio() > match_ratio
