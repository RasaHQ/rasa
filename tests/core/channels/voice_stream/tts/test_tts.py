import difflib

from rasa.core.channels.voice_stream.asr.asr_engine import ASREngine
from rasa.core.channels.voice_stream.asr.asr_event import (
    NewTranscript,
    UserIsSpeaking,
)
from rasa.core.channels.voice_stream.tts.tts_engine import TTSEngine
from rasa.core.channels.voice_stream.util import generate_silence


async def run_single_utterance_through_tts_and_asr(
    text: str, asr_engine: ASREngine, tts_engine: TTSEngine, match_ratio: float = 0.75
):
    await asr_engine.connect()
    async for chunk in tts_engine.synthesize(text):
        await asr_engine.send_audio_chunks(chunk)
    offset = 0
    step_size = 1024
    silence = generate_silence(2.5)
    while offset < len(silence):
        await asr_engine.send_audio_chunks(silence[offset : offset + step_size])
        offset += step_size
    await asr_engine.signal_audio_done()

    events = []
    async for event in asr_engine.stream_asr_events():
        events.append(event)

    assert len(events) == 2
    assert isinstance(events[0], UserIsSpeaking)
    assert isinstance(events[1], NewTranscript)
    match = difflib.SequenceMatcher(None, events[1].text, text)
    assert match.ratio() > 0.75
