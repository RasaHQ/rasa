import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.call_state import CallState, _call_state
from rasa.core.channels.voice_stream.jambonz import (
    JambonzStreamInputChannel,
    JambonzStreamOutputChannel,
    map_call_params,
)
from rasa.core.channels.voice_stream.voice_channel import (
    ContinueConversationAction,
    EndConversationAction,
    NewAudioAction,
)


@pytest.fixture
def input_channel() -> JambonzStreamInputChannel:
    server_url = "localhost"
    asr_config = {"name": "azure"}
    tts_config = {"name": "azure"}
    return JambonzStreamInputChannel(server_url, asr_config, tts_config)


@pytest.fixture
def call_metadata():
    return {
        "sampleRate": 16000,
        "mixType": "mono",
        "callSid": "d5d1dffb-36fc-4314-81ef-054d41f06b8b",
        "direction": "inbound",
        "from": "+491234567890",
        "to": "+13234866320",
        "callId": "cf85413f-c6ed-123e-64b4-021b968ebcc3",
        "sbcCallid": "09ff53ed-2b3d-4e48-82b7-b0f8af80dd8d",
        "sipStatus": 200,
        "sipReason": "OK",
        "callStatus": "in-progress",
        "accountSid": "d88c9c57-3365-451c-98ae-a6a3cfdb8165",
        "traceId": "5322da038c1a38997eaa7f437c9f451c",
        "applicationSid": "3d1d5ceb-d371-4b2f-8e0e-35536e22c422",
        "fsSipAddress": "10.0.45.157:5070",
        "originatingSipIp": "99.77.253.45",
        "originatingSipTrunkName": "AWS Connect",
        "apiBaseUrl": "http://52.10.150.71/v1",
    }


@pytest.fixture
def mock_tts_engine():
    engine = MagicMock()
    engine.generate_audio.return_value = b"dummy_audio"
    return engine


@pytest.fixture
def mock_websocket(call_metadata):
    ws = AsyncMock()
    ws.recv.side_effect = [
        json.dumps(call_metadata),
    ]
    return ws


@pytest.fixture
def sample_audio_bytes():
    # Create 1 second of silence at 8kHz
    return bytes([0xFF] * 8000)


@pytest.fixture
def setup_call_state():
    """Setup and teardown call state for tests."""
    # Initialize a new call state
    _call_state.set(CallState())
    yield
    # Cleanup after test
    _call_state.set(None)


def test_map_call_params(call_metadata):
    """Test mapping of call parameters from metadata."""
    params = map_call_params(call_metadata)
    assert params.call_id == "d5d1dffb-36fc-4314-81ef-054d41f06b8b"
    assert params.user_phone == "+491234567890"
    assert params.bot_phone == "+13234866320"
    assert params.stream_id == "d5d1dffb-36fc-4314-81ef-054d41f06b8b"


async def test_collect_call_parameters(input_channel, mock_websocket):
    """Test collection of call parameters from websocket."""
    params = await input_channel.collect_call_parameters(mock_websocket)
    assert params is not None
    assert params.call_id == "d5d1dffb-36fc-4314-81ef-054d41f06b8b"


def test_channel_bytes_conversion(input_channel, sample_audio_bytes):
    """Test audio format conversion."""
    # Convert L16 PCM 16kHz to μ-law 8kHz
    result = input_channel.channel_bytes_to_rasa_audio_bytes(sample_audio_bytes)
    assert isinstance(result, bytes)
    # L16 PCM is 2 bytes per sample, μ-law is 1 byte per sample
    assert len(result) == len(sample_audio_bytes) // 2


def test_map_input_message_bytes(input_channel, sample_audio_bytes, mock_websocket):
    """Test handling of binary audio input."""
    action = input_channel.map_input_message(sample_audio_bytes, mock_websocket)
    assert isinstance(action, NewAudioAction)


def test_map_input_message_mark(input_channel, mock_websocket, setup_call_state):
    """Test handling of mark messages."""
    _call_state.get().latest_bot_audio_id = "1234"
    mark_message = {"type": "mark", "data": {"name": "1234"}}
    action = input_channel.map_input_message(json.dumps(mark_message), mock_websocket)
    assert isinstance(action, ContinueConversationAction)

    # Test mark message with hangup flag
    _call_state.get().should_hangup = True
    action = input_channel.map_input_message(json.dumps(mark_message), mock_websocket)
    assert isinstance(action, EndConversationAction)


async def test_output_channel_audio_sending(mock_websocket, mock_tts_engine):
    """Test audio sending through output channel."""
    output_channel = JambonzStreamOutputChannel(mock_websocket, mock_tts_engine, {})
    audio_bytes = RasaAudioBytes(b"test_audio")

    await output_channel.send_audio_bytes("test_recipient", audio_bytes)
    assert mock_websocket.send.called

    # Output is L16 PCM (2 bytes per sample)
    # Input is μ-law (1 byte per sample)
    sent_bytes = mock_websocket.send.call_args[0][0]
    assert len(sent_bytes) == len(audio_bytes) * 2


def test_create_marker_message(mock_websocket, mock_tts_engine):
    """Test marker message creation."""
    output_channel = JambonzStreamOutputChannel(mock_websocket, mock_tts_engine, {})
    message, marker_id = output_channel.create_marker_message("test_recipient")

    assert isinstance(message, str)
    parsed = json.loads(message)
    assert parsed["type"] == "mark"
    assert "name" in parsed["data"]
    assert len(marker_id) > 0


async def test_blueprint_health_endpoint(input_channel):
    """Test health check endpoint."""
    from sanic import Sanic

    app = Sanic("test_app")
    blueprint = input_channel.blueprint(AsyncMock())
    app.blueprint(blueprint)
    routes = [route.uri for route in blueprint.routes]
    prefix = "/rasa.core.channels.voice_stream.jambonz"
    assert prefix in routes  # for / endpoint
    assert prefix + "/webhook" in routes
    assert prefix + "/call_status" in routes
    assert prefix + "/websocket" in routes
