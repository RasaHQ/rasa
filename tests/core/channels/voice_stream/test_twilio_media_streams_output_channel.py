import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from sanic.exceptions import WebsocketClosed

from rasa.core.channels.voice_stream.audio_bytes import (
    MULAW_8KHZ,
    AudioFormat,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.call_state import CallState, _call_state
from rasa.core.channels.voice_stream.tts.azure import AzureTTS, AzureTTSConfig
from rasa.core.channels.voice_stream.tts.tts_cache import TTSCache
from rasa.core.channels.voice_stream.twilio_media_streams import (
    TwilioMediaStreamsOutputChannel,
)
from rasa.shared.constants import AZURE_SPEECH_API_KEY_ENV_VAR


@pytest.fixture(autouse=True)
def mock_azure_speech_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a dummy Azure Speech API key so AzureTTS can be instantiated
    in tests without a real credential present in the environment."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "test-key")


@pytest.fixture
async def mock_websocket() -> AsyncMock:
    """Mock websocket for testing."""
    return AsyncMock()


@pytest.fixture
def tts_cache() -> TTSCache:
    """Mock TTS cache for testing."""
    return TTSCache(1)


@pytest.fixture
def azure_tts_config() -> AzureTTSConfig:
    """TTS config for testing."""
    return AzureTTSConfig(
        speech_region="germanywestcentral",
        language="en-US",
    )


@pytest.fixture
async def tts_engine(
    azure_tts_config: AzureTTSConfig, monkeypatch: pytest.MonkeyPatch
) -> AzureTTS:
    """TTS engine for testing.

    synthesize() is patched to return a single silent audio chunk so tests
    never make real HTTP calls to Azure.
    """
    engine = AzureTTS("en", MULAW_8KHZ, azure_tts_config)
    monkeypatch.setattr(
        engine, "synthesize", MagicMock(return_value=_single_chunk_async_gen())
    )
    return engine


def ensure_call_state_context():
    _call_state.set(CallState())


def check_media_message(message: str, recipient_id: str):
    data = json.loads(message)
    assert data["event"] == "media"
    assert data["streamSid"] == recipient_id
    assert len(data["media"]["payload"]) > 0


def check_mark_message(message: str, recipient_id: str):
    data = json.loads(message)
    assert data["event"] == "mark"
    assert data["streamSid"] == recipient_id
    assert len(data["mark"]["name"]) > 0


async def test_twilio_media_streams_output_channel_send(
    tts_engine: AzureTTS,
    tts_cache: TTSCache,
    mock_websocket: AsyncMock,
):
    ensure_call_state_context()
    recipient_id = "test_id"
    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, MULAW_8KHZ
    )
    await output_channel.send_text_message(recipient_id, "Hi There.")
    messages = mock_websocket.send.call_args_list
    # receiving chunked messages
    assert len(messages) > 1
    for i in range(len(messages)):
        message = messages[i][0][0]
        if "media" in message:
            check_media_message(message, recipient_id)
        else:
            check_mark_message(message, recipient_id)

    # last message should be mark
    check_mark_message(messages[-1][0][0], recipient_id)


async def test_twilio_media_streams_output_channel_caching(
    tts_engine: AzureTTS,
    tts_cache: TTSCache,
    mock_websocket: AsyncMock,
):
    ensure_call_state_context()
    recipient_id = "test_id"
    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, MULAW_8KHZ
    )
    await output_channel.send_text_message(recipient_id, "Hi There.")

    websocket_2 = AsyncMock()
    output_channel_2 = TwilioMediaStreamsOutputChannel(
        websocket_2, tts_engine, tts_cache, MULAW_8KHZ
    )
    await output_channel_2.send_text_message(recipient_id, "Hi There.")
    messages = websocket_2.send.call_args_list

    # second time also media and mark messages
    assert len(messages) > 1
    for _message in messages:
        message = _message[0][0]
        if "media" in message:
            check_media_message(message, recipient_id)
        else:
            check_mark_message(message, recipient_id)

    # last message should be mark
    check_mark_message(messages[-1][0][0], recipient_id)


async def test_twilio_media_streams_output_channel_send_when_client_closed(
    mock_websocket: AsyncMock,
    tts_cache: TTSCache,
    tts_engine: AzureTTS,
):
    ensure_call_state_context()
    mock_websocket.send.side_effect = WebsocketClosed()
    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, MULAW_8KHZ
    )
    recipient_id = "test_id"
    await output_channel.send_text_message(recipient_id, "Hi There.")


def test_rasa_audio_bytes_to_channel_bytes(
    mock_websocket: AsyncMock,
    tts_cache: TTSCache,
    tts_engine: AzureTTS,
    mulaw_format: AudioFormat,
):
    rasa_audio_bytes = RasaAudioBytes(b"\00", format=mulaw_format)
    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, mulaw_format
    )
    channel_bytes = output_channel.rasa_audio_bytes_to_channel_bytes(rasa_audio_bytes)
    assert channel_bytes == base64.b64encode(rasa_audio_bytes.data)


def test_channel_bytes_to_message(
    mock_websocket: AsyncMock,
    tts_cache: TTSCache,
    tts_engine: AzureTTS,
    mulaw_format: AudioFormat,
):
    rasa_audio_bytes = RasaAudioBytes(b"\00", format=mulaw_format)
    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, mulaw_format
    )
    channel_bytes = output_channel.rasa_audio_bytes_to_channel_bytes(rasa_audio_bytes)
    recipient_id = "test_id"
    message = output_channel.channel_bytes_to_message(recipient_id, channel_bytes)
    check_media_message(message, recipient_id)


def test_twilio_media_streams_output_channel_name() -> None:
    assert TwilioMediaStreamsOutputChannel.name() == "twilio_media_streams"


def _make_streaming_tts_engine() -> AsyncMock:
    """Return a mock TTS engine that advertises streaming_input=True."""
    engine = AsyncMock()
    engine.streaming_input = True
    engine.signal_text_done = AsyncMock()
    # stream_audio() is called without `await` and must return an async iterator
    # directly, so use MagicMock with side_effect (not AsyncMock).
    engine.stream_audio = MagicMock(side_effect=_empty_async_gen)
    engine.set_language = AsyncMock()
    return engine


async def _empty_async_gen():
    """Async generator that yields nothing (simulates a finished TTS stream)."""
    return
    yield  # make this an async generator


async def test_send_response_chunk_end_retains_accumulated_text(
    tts_cache: TTSCache,
    mock_websocket: AsyncMock,
) -> None:
    """send_response_chunk_end() does NOT clear accumulated streaming text, so
    a subsequent send_text_message with identical content is detected as a
    duplicate and skipped.
    """
    ensure_call_state_context()
    recipient_id = "test_id"
    tts_engine = _make_streaming_tts_engine()

    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, MULAW_8KHZ
    )
    output_channel.audio_sender_task = None

    await output_channel.send_response_chunk_start(recipient_id)
    await output_channel.send_response_chunk(recipient_id, "Hello there")
    await output_channel.send_response_chunk_end(recipient_id)

    # Accumulated text must still be available after chunk_end for dedup.
    assert output_channel._accumulated_streaming_text == "Hello there"


def _make_non_streaming_tts_engine() -> AsyncMock:
    """Return a mock TTS engine with streaming_input=False (non-streaming).

    Synthesize returns an async generator yielding a single audio chunk so
    that send_text_message can complete without hitting a real TTS service.
    """
    engine = AsyncMock()
    engine.streaming_input = False
    engine.set_language = AsyncMock()
    engine.synthesize = MagicMock(return_value=_single_chunk_async_gen())
    return engine


async def _single_chunk_async_gen():
    """Async generator yielding one silent audio chunk (simulates TTS synthesis)."""
    yield RasaAudioBytes(b"\x00" * 160, format=MULAW_8KHZ)


async def test_send_text_message_skipped_when_duplicate_of_streamed_response(
    tts_cache: TTSCache,
    mock_websocket: AsyncMock,
) -> None:
    """send_text_message() is a no-op when its text matches the last streamed
    response, preventing double delivery of the same content."""
    ensure_call_state_context()
    recipient_id = "test_id"
    tts_engine = _make_non_streaming_tts_engine()

    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, MULAW_8KHZ
    )

    # Simulate a completed streaming session that delivered "Hello there"
    await output_channel.send_response_chunk_start(recipient_id)
    await output_channel.send_response_chunk(recipient_id, "Hello there")
    await output_channel.send_response_chunk_end(recipient_id)
    mock_websocket.reset_mock()

    # The identical text arriving via send_text_message must be dropped
    await output_channel.send_text_message(recipient_id, "Hello there")

    mock_websocket.send.assert_not_called()


async def test_send_text_message_different_text_is_not_skipped(
    tts_cache: TTSCache,
    mock_websocket: AsyncMock,
) -> None:
    """send_text_message() is delivered normally when its text differs from the
    last streamed response (e.g. the MCP agent case where send_text_message is
    never called after streaming, so the next turn's distinct message must go
    through)."""
    ensure_call_state_context()
    recipient_id = "test_id"
    tts_engine = _make_non_streaming_tts_engine()

    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, MULAW_8KHZ
    )

    # Simulate a completed streaming session that delivered one response
    await output_channel.send_response_chunk_start(recipient_id)
    await output_channel.send_response_chunk(recipient_id, "First streamed response")
    await output_channel.send_response_chunk_end(recipient_id)
    mock_websocket.reset_mock()

    # A different message on the next turn must NOT be suppressed
    await output_channel.send_text_message(recipient_id, "Next turn response")

    assert mock_websocket.send.call_count >= 1
