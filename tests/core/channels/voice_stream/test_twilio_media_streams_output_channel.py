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


def _make_streaming_tts_engine() -> MagicMock:
    """Return a mock TTS engine that advertises streaming_input=True."""
    engine = MagicMock()
    engine.streaming_input = True
    engine.signal_text_done = AsyncMock()
    engine.stream_audio = AsyncMock(return_value=_empty_async_gen())
    engine.set_language = AsyncMock()
    return engine


async def _empty_async_gen():
    """Async generator that yields nothing (simulates a finished TTS stream)."""
    return
    yield  # make this an async generator


async def test_send_response_chunk_end_default_sets_streaming_flag(
    tts_cache: TTSCache,
    mock_websocket: AsyncMock,
) -> None:
    """send_response_chunk_end() with default is_intermediate=False sets
    streaming_response_sent=True, causing a subsequent send_text_message to be skipped.
    """
    ensure_call_state_context()
    recipient_id = "test_id"
    tts_engine = _make_streaming_tts_engine()

    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, MULAW_8KHZ
    )

    # Simulate a completed streaming session (no background task running)
    output_channel.audio_sender_task = None

    await output_channel.send_response_chunk_end(recipient_id)

    assert output_channel.streaming_response_sent is True


async def test_send_response_chunk_end_intermediate_resets_streaming_flag(
    tts_cache: TTSCache,
    mock_websocket: AsyncMock,
) -> None:
    """send_response_chunk_end(is_intermediate=True) does not set
    streaming_response_sent to True, so when the flag was False
    (e.g. no prior full streaming response), it remains False and a subsequent
    send_text_message is not skipped."""
    ensure_call_state_context()
    recipient_id = "test_id"
    tts_engine = _make_streaming_tts_engine()

    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, MULAW_8KHZ
    )
    output_channel.audio_sender_task = None

    # Flag is False (no prior full streaming response)
    assert output_channel.streaming_response_sent is False

    await output_channel.send_response_chunk_end(recipient_id, is_intermediate=True)

    # Intermediate chunk end does not set the flag to True
    assert output_channel.streaming_response_sent is False


def _make_non_streaming_tts_engine() -> MagicMock:
    """Return a mock TTS engine with streaming_input=False (non-streaming).

    Synthesize returns an async generator yielding a single audio chunk so
    that send_text_message can complete without hitting a real TTS service.
    """
    engine = MagicMock()
    engine.streaming_input = False
    engine.set_language = AsyncMock()
    engine.synthesize = MagicMock(return_value=_single_chunk_async_gen())
    return engine


async def _single_chunk_async_gen():
    """Async generator yielding one silent audio chunk (simulates TTS synthesis)."""
    yield RasaAudioBytes(b"\x00" * 160, format=MULAW_8KHZ)


async def test_send_text_message_skipped_after_non_intermediate_chunk_end(
    tts_cache: TTSCache,
    mock_websocket: AsyncMock,
) -> None:
    """After send_response_chunk_end() with default is_intermediate=False,
    send_text_message() must be a no-op (no audio sent to websocket)."""
    ensure_call_state_context()
    recipient_id = "test_id"
    tts_engine = _make_non_streaming_tts_engine()

    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, MULAW_8KHZ
    )

    # Manually set the flag as send_response_chunk_end would in a streaming session
    output_channel.streaming_response_sent = True

    await output_channel.send_text_message(recipient_id, "This should be skipped.")

    # No audio or marker messages should have been sent
    mock_websocket.send.assert_not_called()
    # Flag must be reset for the next response
    assert output_channel.streaming_response_sent is False


async def test_send_text_message_not_skipped_after_intermediate_chunk_end(
    tts_cache: TTSCache,
    mock_websocket: AsyncMock,
) -> None:
    """When the only chunk end was intermediate (flag never set to True),
    send_text_message() must proceed normally and send audio to the websocket."""
    ensure_call_state_context()
    recipient_id = "test_id"
    # Use streaming engine so send_response_chunk_end runs the is_intermediate branch
    tts_engine = _make_streaming_tts_engine()
    tts_engine.synthesize = MagicMock(return_value=_single_chunk_async_gen())

    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, MULAW_8KHZ
    )
    output_channel.audio_sender_task = None

    # No prior full streaming response — only an intermediate chunk end
    await output_channel.send_response_chunk_end(recipient_id, is_intermediate=True)

    await output_channel.send_text_message(recipient_id, "This should be sent.")

    # At least a start marker and end marker should have been sent
    assert mock_websocket.send.call_count >= 1


async def test_send_text_message_flag_reset_allows_subsequent_messages(
    tts_cache: TTSCache,
    mock_websocket: AsyncMock,
) -> None:
    """Verify that streaming_response_sent is reset to False after being consumed
    by send_text_message, so the next message is not inadvertently skipped."""
    ensure_call_state_context()
    recipient_id = "test_id"
    tts_engine = _make_non_streaming_tts_engine()

    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache, MULAW_8KHZ
    )

    # First call: flag is set — message should be skipped
    output_channel.streaming_response_sent = True
    await output_channel.send_text_message(recipient_id, "Skipped message.")
    assert mock_websocket.send.call_count == 0
    assert output_channel.streaming_response_sent is False

    # Second call: flag was reset — message should be sent normally
    await output_channel.send_text_message(recipient_id, "Sent message.")
    assert mock_websocket.send.call_count >= 1
