#!/usr/bin/env python3
"""Test script to send text to Azure TTS and save synthesized audio."""

import pytest

from rasa.core.channels.voice_stream.tts.azure import AzureTTS


@pytest.mark.asyncio
async def test_azure_tts(tmp_path):
    output_path = tmp_path / "output.wav"
    tts_engine = AzureTTS()
    text = "hello my name is Edgar"
    audio_bytes = b""
    async for chunk in tts_engine.synthesize(text):
        audio_bytes += chunk
    output_path.write_bytes(audio_bytes)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
