#!/usr/bin/env python3
"""Test script to send text to Deepgram TTS and save synthesized audio."""

import pytest

from rasa.core.channels.voice_stream.audio_bytes import MULAW_8KHZ, RasaAudioBytes
from rasa.core.channels.voice_stream.tts.deepgram import DeepgramTTS


@pytest.mark.asyncio
async def test_deepgram_tts(tmp_path):
    output_path = tmp_path / "output.wav"
    tts_engine = DeepgramTTS("en", MULAW_8KHZ)
    text = "hello my name is Edgar"
    audio_bytes = RasaAudioBytes(b"", format=MULAW_8KHZ)
    try:
        await tts_engine.connect()
        async for chunk in tts_engine.synthesize(text):
            audio_bytes += chunk
    finally:
        await tts_engine.close_connection()
    output_path.write_bytes(audio_bytes.data)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
