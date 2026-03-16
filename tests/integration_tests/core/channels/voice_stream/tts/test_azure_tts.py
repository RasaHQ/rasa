#!/usr/bin/env python3
"""Integration tests for Azure TTS engine."""

import asyncio

import pytest

from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.tts.azure import AzureTTS, AzureTTSConfig

_SPEECH_REGION = "germanywestcentral"
_INTERRUPT_TEST_TEXT = (
    "Hello, I am a conversational voice assistant and I can help you today."
)
_NEXT_MESSAGE_TEXT = "How can I help you?"


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
            speech_region=_SPEECH_REGION,
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


@pytest.mark.parametrize(
    "format",
    [
        MULAW_8KHZ,
        L16_24KHZ,
        L16_48KHZ,
    ],
)
async def test_interruption_during_sending_response_chunks(format):
    """Test interruption when stream_state is SENDING_RESPONSE_CHUNKS.

    Azure's interrupt mechanism differs from WebSocket-based engines: there is no
    cancel message sent to a remote API. Instead, stop_streaming() calls
    stop_speaking_async() on the SDK synthesizer, which halts audio generation.
    The SDK thread then pushes a None sentinel to the audio queue, causing
    stream_audio() to exit cleanly.

    Azure also does not manage stream_state internally — the voice channel layer
    is responsible for that transition.

    Verifies that:
    - stop_streaming_output_audio_chunks is True (audio stops being forwarded)
    - stop_speaking_async() terminates the SDK, ending the audio queue stream
    - The engine can synthesize the next message normally (bot moves on)
    """
    tts_engine = AzureTTS(
        rasa_language="en",
        config=AzureTTSConfig(speech_region=_SPEECH_REGION),
        format=format,
    )

    await tts_engine.connect()
    try:
        await tts_engine.prepare_response()

        first_chunk_event = asyncio.Event()
        received_chunks: list[RasaAudioBytes] = []

        async def consume_audio() -> None:
            await tts_engine.send_text_chunk(_INTERRUPT_TEST_TEXT)
            await tts_engine.signal_text_done()
            async for chunk in tts_engine.stream_audio():
                received_chunks.append(chunk)
                first_chunk_event.set()

        consume_task = asyncio.create_task(consume_audio())
        await asyncio.wait_for(first_chunk_event.wait(), timeout=15.0)

        await tts_engine.stop_streaming()
        await asyncio.wait_for(consume_task, timeout=15.0)

        assert tts_engine.stop_streaming_output_audio_chunks is True
        assert len(received_chunks) > 0

        # Verify the bot can move to the next message after the interrupt
        tts_engine.stop_streaming_output_audio_chunks = False

        next_message_chunks: list[RasaAudioBytes] = []
        async for chunk in tts_engine.synthesize(_NEXT_MESSAGE_TEXT):
            next_message_chunks.append(chunk)
        assert len(next_message_chunks) > 0
    finally:
        await tts_engine.close_connection()


@pytest.mark.parametrize(
    "format",
    [
        MULAW_8KHZ,
        L16_24KHZ,
        L16_48KHZ,
    ],
)
async def test_interruption_after_response_chunks_sent(format):
    """Test interruption when stream_state is RESPONSE_CHUNKS_SENT.

    Azure's interrupt mechanism differs from WebSocket-based engines: there is no
    cancel message sent to a remote API. Instead, stop_streaming() calls
    stop_speaking_async() on the SDK synthesizer, which halts audio generation.
    The SDK thread then pushes a None sentinel to the audio queue, causing
    stream_audio() to exit cleanly.

    Azure also does not manage stream_state internally — the voice channel layer
    is responsible for that transition.

    Verifies that:
    - stop_streaming_output_audio_chunks is True (audio stops being forwarded)
    - stop_speaking_async() terminates the SDK mid-generation, ending the stream
    - The engine can synthesize the next message normally (bot moves on)
    """
    tts_engine = AzureTTS(
        rasa_language="en",
        config=AzureTTSConfig(speech_region=_SPEECH_REGION),
        format=format,
    )

    await tts_engine.connect()
    try:
        await tts_engine.prepare_response()

        first_chunk_event = asyncio.Event()
        received_chunks: list[RasaAudioBytes] = []

        async def consume_audio() -> None:
            await tts_engine.send_text_chunk(_INTERRUPT_TEST_TEXT)
            await tts_engine.signal_text_done()
            async for chunk in tts_engine.stream_audio():
                received_chunks.append(chunk)
                first_chunk_event.set()

        consume_task = asyncio.create_task(consume_audio())
        # Wait until at least one audio chunk has been received, confirming that
        # signal_text_done() was already called and the SDK is mid-generation —
        # the realistic RESPONSE_CHUNKS_SENT condition.
        await asyncio.wait_for(first_chunk_event.wait(), timeout=15.0)

        await tts_engine.stop_streaming()
        await asyncio.wait_for(consume_task, timeout=15.0)

        assert tts_engine.stop_streaming_output_audio_chunks is True
        assert len(received_chunks) > 0

        # Verify the bot can move to the next message after the interrupt
        tts_engine.stop_streaming_output_audio_chunks = False

        next_message_chunks: list[RasaAudioBytes] = []
        async for chunk in tts_engine.synthesize(_NEXT_MESSAGE_TEXT):
            next_message_chunks.append(chunk)
        assert len(next_message_chunks) > 0
    finally:
        await tts_engine.close_connection()
