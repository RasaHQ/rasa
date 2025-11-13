#!/usr/bin/env python3
"""Test script to send text to Deepgram TTS and save synthesized audio."""

import pytest

from rasa.core.channels.voice_stream.tts.deepgram import DeepgramTTS


@pytest.mark.asyncio
async def test_deepgram_tts(tmp_path):
    output_path = tmp_path / "output.wav"
    tts_engine = DeepgramTTS()
    text = "hello my name is Edgar"
    audio_bytes = b""
    async for chunk in tts_engine.synthesize(text):
        audio_bytes += chunk
    output_path.write_bytes(audio_bytes)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
