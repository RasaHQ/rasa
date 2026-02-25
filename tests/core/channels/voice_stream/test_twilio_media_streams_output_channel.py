import base64
import json
from unittest.mock import AsyncMock

import pytest
from sanic.exceptions import WebsocketClosed

from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.call_state import CallState, _call_state
from rasa.core.channels.voice_stream.tts.azure import AzureTTS, AzureTTSConfig
from rasa.core.channels.voice_stream.tts.tts_cache import TTSCache
from rasa.core.channels.voice_stream.twilio_media_streams import (
    TwilioMediaStreamsOutputChannel,
)


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
async def tts_engine(azure_tts_config: AzureTTSConfig) -> AzureTTS:
    """TTS engine for testing."""
    return AzureTTS("en", azure_tts_config)


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
        mock_websocket, tts_engine, tts_cache
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
        mock_websocket, tts_engine, tts_cache
    )
    await output_channel.send_text_message(recipient_id, "Hi There.")

    websocket_2 = AsyncMock()
    output_channel_2 = TwilioMediaStreamsOutputChannel(
        websocket_2, tts_engine, tts_cache
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
    mock_websocket: AsyncMock, tts_cache: TTSCache, tts_engine: AzureTTS
):
    ensure_call_state_context()
    mock_websocket.send.side_effect = WebsocketClosed()
    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache
    )
    recipient_id = "test_id"
    await output_channel.send_text_message(recipient_id, "Hi There.")


def test_rasa_audio_bytes_to_channel_bytes(
    mock_websocket: AsyncMock, tts_cache: TTSCache, tts_engine: AzureTTS
):
    rasa_audio_bytes = RasaAudioBytes(b"\00")
    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache
    )
    channel_bytes = output_channel.rasa_audio_bytes_to_channel_bytes(rasa_audio_bytes)
    assert channel_bytes == base64.b64encode(rasa_audio_bytes)


def test_channel_bytes_to_message(
    mock_websocket: AsyncMock, tts_cache: TTSCache, tts_engine: AzureTTS
):
    rasa_audio_bytes = RasaAudioBytes(b"\00")
    output_channel = TwilioMediaStreamsOutputChannel(
        mock_websocket, tts_engine, tts_cache
    )
    channel_bytes = output_channel.rasa_audio_bytes_to_channel_bytes(rasa_audio_bytes)
    recipient_id = "test_id"
    message = output_channel.channel_bytes_to_message(recipient_id, channel_bytes)
    check_media_message(message, recipient_id)


def test_twilio_media_streams_output_channel_name() -> None:
    assert TwilioMediaStreamsOutputChannel.name() == "twilio_media_streams"
