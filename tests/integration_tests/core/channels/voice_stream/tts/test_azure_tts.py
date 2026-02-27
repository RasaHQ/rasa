#!/usr/bin/env python3
"""Test script to send text to Azure TTS and save synthesized audio."""

import pytest

from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.tts.azure import AzureTTS, AzureTTSConfig


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "format",
    [
        MULAW_8KHZ,
        L16_24KHZ,
        L16_48KHZ,
    ],
)
async def test_azure_tts(tmp_path, format):
    output_path = tmp_path / "output.wav"
    tts_engine = AzureTTS(
        rasa_language="en",
        config=AzureTTSConfig(
            speech_region="germanywestcentral",
        ),
        format=format,
    )
    text = "hello my name is Edgar"
    audio_bytes = RasaAudioBytes(b"", format=format)
    async for chunk in tts_engine.synthesize(text):
        audio_bytes += chunk
    output_path.write_bytes(audio_bytes.data)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
