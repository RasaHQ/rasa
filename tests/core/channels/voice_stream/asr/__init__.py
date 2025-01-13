import asyncio
import difflib
from typing import List

from rasa.core.channels.voice_stream.asr.asr_engine import ASREngine
from rasa.core.channels.voice_stream.asr.asr_event import (
    ASREvent,
    NewTranscript,
    UserIsSpeaking,
)
from rasa.core.channels.voice_stream.audio_bytes import HERTZ
from rasa.core.channels.voice_stream.util import read_wav_to_rasa_audio_bytes


async def run_transcription(audio_path: str, asr_engine: ASREngine) -> List[ASREvent]:
    rasa_audio_bytes = read_wav_to_rasa_audio_bytes(audio_path)
    step_size = 1024
    await asr_engine.connect()
    offset = 0
    while offset < len(rasa_audio_bytes):
        await asr_engine.send_audio_chunks(
            rasa_audio_bytes[offset : offset + step_size]
        )
        offset += step_size
        await asyncio.sleep(step_size / HERTZ)
    await asr_engine.signal_audio_done()

    events = []
    async for event in asr_engine.stream_asr_events():
        events.append(event)
    return events


async def run_single_utterance_transcription(
    audio_path: str, transcript: str, asr_engine: ASREngine
):
    events = await run_transcription(audio_path, asr_engine)

    assert len(events) > 2
    assert all([isinstance(event, UserIsSpeaking) for event in events[:-1]])
    assert isinstance(events[-1], NewTranscript)
    match = difflib.SequenceMatcher(None, events[-1].text, transcript)
    assert match.ratio() > 0.75
