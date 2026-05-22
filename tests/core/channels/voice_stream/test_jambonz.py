import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from rasa.core.channels.voice_stream.audio_bytes import AudioFormat, RasaAudioBytes
from rasa.core.channels.voice_stream.call_state import (
    BotIsSpeaking,
    BotStoppedSpeaking,
    Marker,
    MarkerType,
    StepType,
)
from rasa.core.channels.voice_stream.jambonz import (
    JAMBONZ_STREAMS_WEBSOCKET_PATH,
    JambonzStreamInputChannel,
    JambonzStreamOutputChannel,
    map_call_params,
)
from rasa.core.channels.voice_stream.tts import TTSCache
from rasa.core.channels.voice_stream.voice_channel import (
    ContinueConversationAction,
    DTMFInputAction,
    EndConversationAction,
    MarkerInput,
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
def call_metadata() -> Dict[str, Any]:
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
def mock_tts_engine() -> MagicMock:
    engine = MagicMock()
    engine.generate_audio.return_value = b"dummy_audio"
    return engine


@pytest.fixture
def mock_websocket(call_metadata) -> AsyncMock:
    ws = AsyncMock()
    ws.recv.side_effect = [
        json.dumps(call_metadata),
    ]
    return ws


@pytest.fixture
def sample_audio_bytes() -> bytes:
    # Create 1 second of silence at 8kHz
    return bytes([0xFF] * 8000)


@pytest.fixture
def jambonz_output_channel(
    mock_websocket: AsyncMock, mock_tts_engine: MagicMock, mulaw_format: AudioFormat
) -> JambonzStreamOutputChannel:
    return JambonzStreamOutputChannel(
        mock_websocket, mock_tts_engine, TTSCache(max_size=2000), mulaw_format
    )


def test_map_call_params(call_metadata: Dict[str, Any]):
    """Test mapping of call parameters from metadata."""
    params = map_call_params(call_metadata)
    assert params.call_id == "d5d1dffb-36fc-4314-81ef-054d41f06b8b"
    assert params.user_phone == "+491234567890"
    assert params.bot_phone == "+13234866320"
    assert params.stream_id == "d5d1dffb-36fc-4314-81ef-054d41f06b8b"


async def test_collect_call_parameters(
    input_channel: JambonzStreamInputChannel, mock_websocket: AsyncMock
):
    """Test collection of call parameters from websocket."""
    params = await input_channel.collect_call_parameters(mock_websocket)
    assert params is not None
    assert params.call_id == "d5d1dffb-36fc-4314-81ef-054d41f06b8b"


def test_channel_bytes_conversion(
    input_channel: JambonzStreamInputChannel, sample_audio_bytes: bytes
):
    """Test that there's no audio format conversion"""
    result = input_channel.channel_bytes_to_rasa_audio_bytes(sample_audio_bytes)
    assert isinstance(result, RasaAudioBytes)
    assert len(result) == len(result)


###############################################################
# Map Input Message
###############################################################


async def test_map_input_message_bytes(
    input_channel: JambonzStreamInputChannel,
    sample_audio_bytes: bytes,
    mock_websocket: AsyncMock,
):
    """Test handling of binary audio input."""
    action = await input_channel.map_input_message(sample_audio_bytes, mock_websocket)
    assert isinstance(action, NewAudioAction)


@pytest.mark.parametrize(
    "bot_utterance_type", [StepType.COLLECT, StepType.REGULAR_UTTER, None]
)
@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_mark_with_no_registered_marker(
    input_channel: JambonzStreamInputChannel,
    mock_websocket: AsyncMock,
    bot_utterance_type: Optional[StepType],
):
    """Test mark message when no marker is registered with that ID."""
    from rasa.core.channels.voice_stream.call_state import call_state

    mark_message = {"type": "mark", "data": {"name": "unregistered_marker_id"}}
    call_state.current_bot_utterance_type = bot_utterance_type
    previous_event_count = call_state.internal_queue.qsize()

    action = await input_channel.map_input_message(
        json.dumps(mark_message), mock_websocket
    )

    assert isinstance(action, ContinueConversationAction)
    # bot_utterance_type should be unchanged
    assert call_state.current_bot_utterance_type == bot_utterance_type
    # No new events are queued
    assert call_state.internal_queue.qsize() == previous_event_count


@pytest.mark.parametrize("step_type", [StepType.COLLECT, StepType.REGULAR_UTTER])
@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_mark_with_start_marker(
    input_channel: JambonzStreamInputChannel,
    mock_websocket: AsyncMock,
    step_type: StepType,
):
    """Test that a START marker updates current_bot_utterance_type,
    sets is_bot_speaking to True and removes the marker."""
    from rasa.core.channels.voice_stream.call_state import call_state

    marker_id = "marker_start_1"
    marker = Marker(
        marker_id=marker_id, marker_type=MarkerType.START, step_type=step_type
    )
    call_state.set_marker(marker)
    mark_message = {"type": "mark", "data": {"name": marker_id}}

    action = await input_channel.map_input_message(
        json.dumps(mark_message), mock_websocket
    )

    assert isinstance(action, ContinueConversationAction)

    assert call_state.internal_queue.qsize() == 1
    event = call_state.internal_queue.get_nowait()
    assert isinstance(event, BotIsSpeaking)

    assert call_state.current_bot_utterance_type == step_type
    assert call_state.get_marker(marker_id) is None


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_mark_with_end_marker(
    input_channel: JambonzStreamInputChannel, mock_websocket: AsyncMock
):
    """Test that an END marker clears current_bot_utterance_type,
    queues BotStoppedSpeaking and removes the marker."""
    from rasa.core.channels.voice_stream.call_state import call_state

    marker_id = "marker_end_1"
    marker = Marker(
        marker_id=marker_id,
        marker_type=MarkerType.END,
        step_type=StepType.REGULAR_UTTER,
    )
    call_state.set_marker(marker)
    call_state.current_bot_utterance_type = StepType.REGULAR_UTTER
    mark_message = {"type": "mark", "data": {"name": marker_id}}

    action = await input_channel.map_input_message(
        json.dumps(mark_message), mock_websocket
    )

    assert isinstance(action, ContinueConversationAction)
    assert call_state.current_bot_utterance_type is None

    assert call_state.internal_queue.qsize() == 1
    event = call_state.internal_queue.get_nowait()
    assert isinstance(event, BotStoppedSpeaking)

    assert call_state.get_marker(marker_id) is None


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_mark_with_end_marker_should_hangup(
    input_channel: JambonzStreamInputChannel, mock_websocket: AsyncMock
):
    """Test that an END marker clears current_bot_utterance_type,
    queues BotStoppedSpeaking and removes the marker.

    EndConversationAction is returned as a result.
    """
    from rasa.core.channels.voice_stream.call_state import call_state

    marker_id = "marker_end_1"
    marker = Marker(
        marker_id=marker_id,
        marker_type=MarkerType.END,
        step_type=StepType.REGULAR_UTTER,
    )
    call_state.set_marker(marker)
    call_state.current_bot_utterance_type = StepType.REGULAR_UTTER
    mark_message = {"type": "mark", "data": {"name": marker_id}}

    call_state.should_hangup = True
    action = await input_channel.map_input_message(
        json.dumps(mark_message), mock_websocket
    )

    assert isinstance(action, EndConversationAction)

    assert call_state.internal_queue.qsize() == 1
    event = call_state.internal_queue.get_nowait()
    assert isinstance(event, BotStoppedSpeaking)

    assert call_state.current_bot_utterance_type is None
    assert call_state.get_marker(marker_id) is None


@pytest.mark.parametrize(
    "current_bot_utterance_type", [StepType.REGULAR_UTTER, StepType.COLLECT, None]
)
@pytest.mark.parametrize("marker_type", [MarkerType.START, MarkerType.END])
@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_mark_with_marker_no_step_type(
    input_channel: JambonzStreamInputChannel,
    mock_websocket: AsyncMock,
    marker_type: MarkerType,
    current_bot_utterance_type: Optional[StepType],
):
    """Test that a marker without step_type is removed, that no events are
    queued and that bot utterance type is not affected."""
    from rasa.core.channels.voice_stream.call_state import call_state

    marker_id = "marker_no_step"
    marker = Marker(marker_id=marker_id, marker_type=marker_type, step_type=None)
    call_state.set_marker(marker)
    mark_message = {"type": "mark", "data": {"name": marker_id}}

    previous_event_count = call_state.internal_queue.qsize()
    call_state.current_bot_utterance_type = current_bot_utterance_type

    action = await input_channel.map_input_message(
        json.dumps(mark_message), mock_websocket
    )

    assert isinstance(action, ContinueConversationAction)
    # Current bot utterance is not affected
    assert call_state.current_bot_utterance_type == current_bot_utterance_type
    # No new events are queued
    assert call_state.internal_queue.qsize() == previous_event_count
    # Marker is removed
    assert call_state.get_marker(marker_id) is None


@pytest.mark.parametrize(
    "current_bot_utterance_type", [StepType.REGULAR_UTTER, StepType.COLLECT, None]
)
@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_mark_with_intermediate_marker(
    input_channel: JambonzStreamInputChannel,
    mock_websocket: AsyncMock,
    current_bot_utterance_type: Optional[StepType],
):
    """Test that an INTERMEDIATE marker leaves current_bot_utterance_type unchanged."""
    from rasa.core.channels.voice_stream.call_state import call_state

    marker_id = "marker_intermediate"
    marker = Marker(
        marker_id=marker_id,
        marker_type=MarkerType.INTERMEDIATE,
        step_type=StepType.COLLECT,
    )
    call_state.set_marker(marker)
    mark_message = {"type": "mark", "data": {"name": marker_id}}
    previous_event_count = call_state.internal_queue.qsize()
    call_state.current_bot_utterance_type = current_bot_utterance_type

    action = await input_channel.map_input_message(
        json.dumps(mark_message), mock_websocket
    )

    assert isinstance(action, ContinueConversationAction)
    # No new events are queued
    assert call_state.internal_queue.qsize() == previous_event_count
    # Current bot utterance is not affected
    assert call_state.current_bot_utterance_type == current_bot_utterance_type
    # Marker is removed
    assert call_state.get_marker(marker_id) is None


async def test_output_channel_audio_sending(
    jambonz_output_channel: JambonzStreamOutputChannel, mock_websocket: AsyncMock
) -> None:
    """Test audio sending through output channel."""
    format = jambonz_output_channel.audio_format
    audio_bytes = RasaAudioBytes(b"test_audio", format=format)

    await jambonz_output_channel.send_audio_bytes("test_recipient", audio_bytes)
    assert mock_websocket.send.called

    # Both input and output are same format
    sent_bytes = mock_websocket.send.call_args[0][0]
    assert len(sent_bytes) == len(audio_bytes)


@pytest.mark.parametrize(
    "marker_type", [MarkerType.START, MarkerType.INTERMEDIATE, MarkerType.END]
)
@pytest.mark.parametrize("step_type", [StepType.COLLECT, StepType.REGULAR_UTTER, None])
@pytest.mark.usefixtures("setup_call_state")
def test_create_marker_message(
    mock_websocket: AsyncMock,
    mock_tts_engine: MagicMock,
    marker_type: MarkerType,
    step_type: Optional[StepType],
    jambonz_output_channel: JambonzStreamOutputChannel,
):
    """Test marker message creation."""
    marker_message = jambonz_output_channel.create_marker_message(
        marker_input=MarkerInput(
            recipient_id="recipient_id", marker_type=marker_type, step_type=step_type
        )
    )

    assert isinstance(marker_message.message, str)
    parsed = json.loads(marker_message.message)
    assert parsed["type"] == "mark"
    assert "name" in parsed["data"]
    assert len(marker_message.message_id) > 0


async def test_blueprint_health_endpoint(input_channel: JambonzStreamInputChannel):
    """Test health check endpoint."""
    from sanic import Sanic

    app = Sanic("test_app")
    blueprint = input_channel.conversation_blueprint(AsyncMock())
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
def test_from_credentials_validation(credentials: Dict[str, str]) -> None:
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
def test_websocket_stream_url(server_url: str, expected_ws_url: str) -> None:
    """Test websocket URL generation with different server URL formats."""
    channel = JambonzStreamInputChannel(
        server_url=server_url,
        asr_config={"name": "azure"},
        tts_config={"name": "azure"},
    )
    assert channel._websocket_stream_url() == expected_ws_url


async def test_map_input_message_dtmf(
    input_channel: JambonzStreamInputChannel, mock_websocket: AsyncMock
) -> None:
    """Test handling of DTMF input messages."""
    dtmf_message = {"event": "dtmf", "dtmf": "5", "duration": "1600"}
    action = await input_channel.map_input_message(
        json.dumps(dtmf_message), mock_websocket
    )

    assert isinstance(action, DTMFInputAction)
    assert action.digit == "5"


@pytest.mark.parametrize(
    "dtmf_digit",
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "#", "*"],
)
async def test_map_input_message_dtmf_all_digits(
    input_channel: JambonzStreamInputChannel, mock_websocket: AsyncMock, dtmf_digit: str
) -> None:
    """Test handling of all valid DTMF digits."""
    dtmf_message = {"event": "dtmf", "dtmf": dtmf_digit, "duration": "1600"}
    action = await input_channel.map_input_message(
        json.dumps(dtmf_message), mock_websocket
    )

    assert isinstance(action, DTMFInputAction)
    assert action.digit == dtmf_digit


async def test_map_input_message_unknown_event(
    input_channel: JambonzStreamInputChannel, mock_websocket: AsyncMock
):
    """Test handling of unknown event types."""
    unknown_message = {"event": "unknown_event", "data": "something"}
    action = await input_channel.map_input_message(
        json.dumps(unknown_message), mock_websocket
    )

    assert isinstance(action, ContinueConversationAction)
