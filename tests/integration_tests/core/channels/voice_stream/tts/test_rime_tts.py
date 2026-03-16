#!/usr/bin/env python3
"""Integration tests for Rime TTS engine."""

import asyncio

import pytest

from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    MULAW_8KHZ,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.tts.rime import RimeTTS
from rasa.core.channels.voice_stream.tts.tts_engine import StreamState

_INTERRUPT_TEST_TEXT = (
    "Hello, I am a conversational voice assistant and I can help you today."
)
_NEXT_MESSAGE_TEXT = "How can I help you?"


@pytest.mark.parametrize(
    "format",
    [
        MULAW_8KHZ,
        L16_24KHZ,
    ],
)
async def test_rime_tts(tmp_path, format):
    """Test basic Rime TTS synthesis and verify audio is produced."""
    output_path = tmp_path / "output.wav"
    tts_engine = RimeTTS(rasa_language="en", format=format)
    text = "hello my name is Edgar"
    audio_bytes = RasaAudioBytes(b"", format=format)
    try:
        await tts_engine.connect()
        async for chunk in tts_engine.synthesize(text):
            audio_bytes += chunk
    finally:
        await tts_engine.close_connection()
    output_path.write_bytes(audio_bytes.data)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.parametrize(
    "format",
    [
        MULAW_8KHZ,
        L16_24KHZ,
    ],
)
async def test_interruption_during_sending_response_chunks(format):
    """Test interruption when stream_state is SENDING_RESPONSE_CHUNKS.

    Simulates an interrupt arriving while the TTS engine is actively streaming
    audio chunks. Verifies that:
    - Audio stops being forwarded to the channel
        (stop_streaming_output_audio_chunks=True)
    - stream_state transitions to INTERRUPTED
    - The engine can synthesize the next message normally (bot moves on)
    """
    tts_engine = RimeTTS(rasa_language="en", format=format)

    await tts_engine.connect()
    try:
        tts_engine.stream_state = StreamState.SENDING_RESPONSE_CHUNKS

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
        assert tts_engine.stream_state == StreamState.INTERRUPTED
        assert len(received_chunks) > 0

        # Verify the bot can move to the next message after the interrupt
        await tts_engine.close_connection()
        await tts_engine.connect()
        tts_engine.stop_streaming_output_audio_chunks = False
        tts_engine.stream_state = StreamState.NO_STREAMING

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
    ],
)
async def test_interruption_after_response_chunks_sent(format):
    """Test interruption when stream_state is RESPONSE_CHUNKS_SENT.

    Simulates an interrupt arriving after all text chunks have been sent and
    audio is actively streaming. Waiting for the first audio chunk before
    triggering the interrupt ensures that stream_audio() is already consuming
    the WebSocket — the same condition that holds in production. This means
    Rime finishes sending the "done" for the active request before any
    response to the clear can arrive, so stream_audio() exits cleanly.
    Verifies that:
    - Audio stops being forwarded to the channel
        (stop_streaming_output_audio_chunks=True)
    - stream_state transitions to NO_STREAMING
    - A clear signal is sent to Rime to stop further audio generation
    - The engine can synthesize the next message normally (bot moves on)
    """
    tts_engine = RimeTTS(rasa_language="en", format=format)

    await tts_engine.connect()
    try:
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

        # At this point signal_text_done() has already been called and at least
        # one audio chunk is in flight — the realistic RESPONSE_CHUNKS_SENT state.
        tts_engine.stream_state = StreamState.RESPONSE_CHUNKS_SENT
        await tts_engine.stop_streaming()
        await asyncio.wait_for(consume_task, timeout=15.0)

        assert tts_engine.stop_streaming_output_audio_chunks is True
        assert tts_engine.stream_state == StreamState.NO_STREAMING
        assert len(received_chunks) > 0

        # Verify the bot can move to the next message after the interrupt
        await tts_engine.close_connection()
        await tts_engine.connect()
        tts_engine.stop_streaming_output_audio_chunks = False

        next_message_chunks: list[RasaAudioBytes] = []
        async for chunk in tts_engine.synthesize(_NEXT_MESSAGE_TEXT):
            next_message_chunks.append(chunk)
        assert len(next_message_chunks) > 0
    finally:
        await tts_engine.close_connection()
