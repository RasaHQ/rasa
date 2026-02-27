import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.call_state import _call_state
from rasa.core.channels.voice_stream.jambonz import (
    JAMBONZ_STREAMS_WEBSOCKET_PATH,
    JambonzStreamInputChannel,
    JambonzStreamOutputChannel,
    map_call_params,
)
from rasa.core.channels.voice_stream.voice_channel import (
    ContinueConversationAction,
    DTMFInputAction,
    EndConversationAction,
    NewAudioAction,
)
from rasa.shared.exceptions import InvalidConfigException


@pytest.fixture
def input_channel() -> JambonzStreamInputChannel:
    server_url = "localhost"
    asr_config = {"name": "azure"}
    tts_config = {"name": "azure"}
    return JambonzStreamInputChannel(
        server_url=server_url,
        asr_config=asr_config,
        tts_config=tts_config,
        username=None,
        password=None,
    )


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
def jambonz_output_channel(mock_websocket, mock_tts_engine, mulaw_format):
    return JambonzStreamOutputChannel(mock_websocket, mock_tts_engine, {}, mulaw_format)


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
    assert isinstance(result, RasaAudioBytes)
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


async def test_output_channel_audio_sending(jambonz_output_channel, mock_websocket):
    """Test audio sending through output channel."""
    format = jambonz_output_channel.audio_format
    audio_bytes = RasaAudioBytes(b"test_audio", format=format)

    await jambonz_output_channel.send_audio_bytes("test_recipient", audio_bytes)
    assert mock_websocket.send.called

    # Output is L16 PCM (2 bytes per sample)
    # Input is μ-law (1 byte per sample)
    sent_bytes = mock_websocket.send.call_args[0][0]
    assert len(sent_bytes) == len(audio_bytes) * 2


def test_create_marker_message(jambonz_output_channel):
    """Test marker message creation."""
    message, marker_id = jambonz_output_channel.create_marker_message("test_recipient")

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


@pytest.mark.parametrize(
    "credentials",
    [
        {"server_url": "example.com"},  # Missing ASR and TTS
        {"server_url": "example.com", "asr": {"name": "azure"}},  # Missing TTS
        {
            "server_url": "example.com",
            "asr": {"name": "azure"},
            "tts": {"name": "azure"},
            "username": "test_user",
        },  # Missing password
        {
            "server_url": "example.com",
            "asr": {"name": "azure"},
            "tts": {"name": "azure"},
            "password": "test_pass",
        },  # Missing username
    ],
)
def test_from_credentials_validation(credentials):
    """Test validation of credentials when creating channel from config."""
    with pytest.raises(InvalidConfigException):
        JambonzStreamInputChannel.from_credentials(credentials)


def test_from_credentials_success():
    """Test successful creation of channel from valid credentials."""
    credentials = {
        "server_url": "example.com",
        "asr": {"name": "azure"},
        "tts": {"name": "azure"},
        "username": "test_user",
        "password": "test_pass",
    }

    channel = JambonzStreamInputChannel.from_credentials(credentials)

    assert isinstance(channel, JambonzStreamInputChannel)
    assert channel.server_url == "example.com"
    assert channel.asr_config == {"name": "azure"}
    assert channel.tts_config == {"name": "azure"}
    assert channel.username == "test_user"
    assert channel.password == "test_pass"


@pytest.mark.parametrize(
    "server_url,expected_ws_url",
    [
        (
            "example.com",
            f"wss://example.com/{JAMBONZ_STREAMS_WEBSOCKET_PATH}",
        ),
        (
            "http://example.com",
            f"ws://example.com/{JAMBONZ_STREAMS_WEBSOCKET_PATH}",
        ),
        (
            "https://example.com",
            f"wss://example.com/{JAMBONZ_STREAMS_WEBSOCKET_PATH}",
        ),
        (
            "http://example.com:8080",
            f"ws://example.com:8080/{JAMBONZ_STREAMS_WEBSOCKET_PATH}",
        ),
        (
            "example.com:8080",
            f"wss://example.com:8080/{JAMBONZ_STREAMS_WEBSOCKET_PATH}",
        ),
    ],
)
def test_websocket_stream_url(server_url: str, expected_ws_url: str):
    """Test websocket URL generation with different server URL formats."""
    channel = JambonzStreamInputChannel(
        server_url=server_url,
        asr_config={"name": "azure"},
        tts_config={"name": "azure"},
    )
    assert channel._websocket_stream_url() == expected_ws_url


def test_map_input_message_dtmf(input_channel, mock_websocket):
    """Test handling of DTMF input messages."""
    dtmf_message = {"event": "dtmf", "dtmf": "5", "duration": "1600"}
    action = input_channel.map_input_message(json.dumps(dtmf_message), mock_websocket)

    assert isinstance(action, DTMFInputAction)
    assert action.digit == "5"


@pytest.mark.parametrize(
    "dtmf_digit",
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "#", "*"],
)
def test_map_input_message_dtmf_all_digits(input_channel, mock_websocket, dtmf_digit):
    """Test handling of all valid DTMF digits."""
    dtmf_message = {"event": "dtmf", "dtmf": dtmf_digit, "duration": "1600"}
    action = input_channel.map_input_message(json.dumps(dtmf_message), mock_websocket)

    assert isinstance(action, DTMFInputAction)
    assert action.digit == dtmf_digit


def test_map_input_message_unknown_event(input_channel, mock_websocket):
    """Test handling of unknown event types."""
    unknown_message = {"event": "unknown_event", "data": "something"}
    action = input_channel.map_input_message(
        json.dumps(unknown_message), mock_websocket
    )

    assert isinstance(action, ContinueConversationAction)
