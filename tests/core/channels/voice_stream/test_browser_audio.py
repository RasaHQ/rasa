import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.browser_audio import (
    BrowserAudioInputChannel,
    BrowserAudioOutputChannel,
)
from rasa.core.channels.voice_stream.voice_channel import (
    ContinueConversationAction,
    EndConversationAction,
    NewAudioAction,
)


@pytest.fixture
def input_channel():
    return BrowserAudioInputChannel(
        server_url="localhost",
        asr_config={"name": "dummy_asr"},
        tts_config={"name": "dummy_tts"},
        recording=False,
    )


@pytest.fixture
def output_channel(mulaw_format):
    return BrowserAudioOutputChannel(
        MagicMock(), MagicMock(), {}, audio_format=mulaw_format
    )


@pytest.fixture
def mock_websocket():
    ws = AsyncMock()
    return ws


@pytest.fixture
def sample_audio_bytes():
    # 1 second of silence, 16-bit (2 bytes/sample)
    return bytes([0x00] * 16000)


def test_channel_bytes_to_rasa_audio_bytes_mulaw_8khz(
    input_channel, sample_audio_bytes
):
    input_channel.audio_format = MULAW_8KHZ
    rasa_audio = input_channel.channel_bytes_to_rasa_audio_bytes(sample_audio_bytes)
    assert isinstance(rasa_audio, RasaAudioBytes)
    # Mulaw 8kHz should result in half the number of samples when converted to L16
    assert len(rasa_audio) == len(sample_audio_bytes) // 2


def test_channel_bytes_to_rasa_audio_bytes_l16_24khz(input_channel, sample_audio_bytes):
    input_channel.audio_format = L16_24KHZ
    rasa_audio = input_channel.channel_bytes_to_rasa_audio_bytes(sample_audio_bytes)
    assert isinstance(rasa_audio, RasaAudioBytes)
    # L16 24kHz should have the same number of bytes as the input
    assert len(rasa_audio) == len(sample_audio_bytes)


def test_channel_bytes_to_rasa_audio_bytes_l16_48khz(input_channel, sample_audio_bytes):
    input_channel.audio_format = L16_48KHZ
    rasa_audio = input_channel.channel_bytes_to_rasa_audio_bytes(sample_audio_bytes)
    assert isinstance(rasa_audio, RasaAudioBytes)
    # L16 48kHz should have the same number of bytes as the input
    assert len(rasa_audio) == len(sample_audio_bytes)


def test_rasa_audio_bytes_to_channel_bytes(
    output_channel, sample_audio_bytes, mulaw_format
):
    rasa_audio = RasaAudioBytes(sample_audio_bytes[:8000], format=mulaw_format)
    channel_bytes = output_channel.rasa_audio_bytes_to_channel_bytes(rasa_audio)
    assert isinstance(channel_bytes, bytes)


def test_channel_bytes_to_message(output_channel, sample_audio_bytes):
    msg = output_channel.channel_bytes_to_message(
        "recipient", sample_audio_bytes[:8000]
    )
    data = json.loads(msg)
    assert "audio" in data
    decoded = base64.b64decode(data["audio"])
    assert decoded == sample_audio_bytes[:8000]


def test_map_input_message_audio(input_channel, sample_audio_bytes, mock_websocket):
    audio_b64 = base64.b64encode(sample_audio_bytes[:8000]).decode("utf-8")
    msg = json.dumps({"audio": audio_b64})
    action = input_channel.map_input_message(msg, mock_websocket)
    assert isinstance(action, NewAudioAction)


def test_map_input_message_marker(input_channel, mock_websocket, setup_call_state):
    from rasa.core.channels.voice_stream.call_state import call_state

    call_state.latest_bot_audio_id = "abc123"
    call_state.should_hangup = False
    msg = json.dumps({"marker": "abc123"})
    action = input_channel.map_input_message(msg, mock_websocket)
    assert isinstance(action, ContinueConversationAction)
    call_state.should_hangup = True
    action = input_channel.map_input_message(msg, mock_websocket)
    assert isinstance(action, EndConversationAction)


def test_map_input_message_unknown(input_channel, mock_websocket):
    msg = json.dumps({"foo": "bar"})
    action = input_channel.map_input_message(msg, mock_websocket)
    assert isinstance(action, ContinueConversationAction)


@pytest.mark.asyncio
async def test_collect_call_parameters(input_channel, mock_websocket):
    params = await input_channel.collect_call_parameters(mock_websocket)
    assert isinstance(params, CallParameters)
    assert params.call_id.startswith("inspect-")


@pytest.mark.asyncio
async def test_interrupt_playback(input_channel, mock_websocket):
    await input_channel.interrupt_playback(
        mock_websocket, CallParameters("cid", "u", "b")
    )
    mock_websocket.send.assert_called()


@pytest.mark.asyncio
async def test_blueprint_health_endpoint(input_channel):
    from sanic import Sanic

    app = Sanic("test_app")
    blueprint = input_channel.blueprint(AsyncMock())
    app.blueprint(blueprint)
    routes = [route.uri for route in blueprint.routes]
    prefix = "/rasa.core.channels.voice_stream.browser_audio"
    assert f"{prefix}/websocket" in routes


def test_from_credentials_success():
    credentials = {
        "server_url": "localhost",
        "asr": {"name": "dummy_asr"},
        "tts": {"name": "dummy_tts"},
    }
    channel = BrowserAudioInputChannel.from_credentials(credentials)
    assert isinstance(channel, BrowserAudioInputChannel)
    assert channel.server_url == "localhost"
    assert channel.asr_config == {"name": "dummy_asr"}
    assert channel.tts_config == {"name": "dummy_tts"}


def test_from_credentials_validation():
    with pytest.raises(Exception):
        BrowserAudioInputChannel.from_credentials(None)
