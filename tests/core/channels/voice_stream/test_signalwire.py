import base64
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from sanic import Request

from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.audio_bytes import L16_24KHZ, RasaAudioBytes
from rasa.core.channels.voice_stream.call_state import (
    BotIsSpeaking,
    BotStoppedSpeaking,
    Marker,
    MarkerType,
    StepType,
)
from rasa.core.channels.voice_stream.signalwire import (
    CALL_SID_REQUEST_KEY,
    DIRECTION_REQUEST_KEY,
    FROM_NUMBER_REQUEST_KEY,
    SIGNALWIRE_WEBSOCKET_PATH,
    TO_NUMBER_REQUEST_KEY,
    SignalWireInputChannel,
    SignalWireOutputChannel,
    _signalwire_call_parameters,
    _websocket_stream_url,
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


class _AsyncWebsocketMessages:
    """Minimal async iterator that yields websocket message strings."""

    def __init__(self, messages: List[str]) -> None:
        self._messages = messages

    def __aiter__(self) -> "_AsyncWebsocketMessages":
        self._iter = iter(self._messages)
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@pytest.fixture
def input_channel() -> SignalWireInputChannel:
    return SignalWireInputChannel(
        server_url="localhost",
        asr_config={"name": "azure"},
        tts_config={"name": "azure"},
        username=None,
        password=None,
    )


@pytest.fixture
def start_event_payload() -> Dict[str, Any]:
    return {
        "event": "start",
        "start": {
            "streamSid": "MZaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "customParameters": {
                "call_id": "CAbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "user_phone": "+491234567890",
                "bot_phone": "+13234866320",
                "direction": "inbound",
            },
        },
    }


@pytest.fixture
def mock_tts_engine() -> MagicMock:
    engine = MagicMock()
    engine.generate_audio.return_value = b"dummy_audio"
    return engine


@pytest.fixture
def async_ws() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def sample_audio_bytes() -> bytes:
    return bytes([0xFF] * 4800)


@pytest.fixture
def signalwire_output_channel(
    mock_tts_engine: MagicMock,
) -> SignalWireOutputChannel:
    ws = AsyncMock()
    return SignalWireOutputChannel(
        ws,
        mock_tts_engine,
        TTSCache(max_size=2000),
        L16_24KHZ,
    )


def test_map_call_params(start_event_payload: Dict[str, Any]) -> None:
    params = map_call_params(start_event_payload)
    assert params.call_id == "CAbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    assert params.user_phone == "+491234567890"
    assert params.bot_phone == "+13234866320"
    assert params.direction == "inbound"
    assert params.stream_id == "MZaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


async def test_collect_call_parameters(
    input_channel: SignalWireInputChannel,
    start_event_payload: Dict[str, Any],
) -> None:
    connected = json.dumps({"event": "connected", "protocol": "foo"})
    start = json.dumps(start_event_payload)
    ws = _AsyncWebsocketMessages([connected, start])

    params = await input_channel.collect_call_parameters(ws)

    assert params is not None
    assert params.call_id == "CAbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    assert params.stream_id == "MZaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


async def test_collect_call_parameters_returns_none_when_no_start(
    input_channel: SignalWireInputChannel,
) -> None:
    ws = _AsyncWebsocketMessages([])
    assert await input_channel.collect_call_parameters(ws) is None


def test_get_sender_id_uses_stream_id(input_channel: SignalWireInputChannel) -> None:
    params = CallParameters(
        call_id="call-id",
        user_phone="+1",
        bot_phone="+2",
        stream_id="stream-sid-value",
    )
    assert input_channel.get_sender_id(params) == "stream-sid-value"


def test_channel_bytes_conversion(
    input_channel: SignalWireInputChannel, sample_audio_bytes: bytes
) -> None:
    encoded = base64.b64encode(sample_audio_bytes)
    result = input_channel.channel_bytes_to_rasa_audio_bytes(encoded)
    assert isinstance(result, RasaAudioBytes)
    assert result.data == sample_audio_bytes
    assert result.format == L16_24KHZ


@pytest.mark.parametrize(
    "request_data, expected",
    [
        (
            {
                CALL_SID_REQUEST_KEY: "test-call-sid-001",
                FROM_NUMBER_REQUEST_KEY: "+15550001111",
                TO_NUMBER_REQUEST_KEY: "+15550002222",
                DIRECTION_REQUEST_KEY: "inbound",
            },
            {
                "call_id": "test-call-sid-001",
                "user_phone": "+15550001111",
                "bot_phone": "+15550002222",
                "direction": "inbound",
            },
        ),
        ({}, {"call_id": "", "user_phone": "", "bot_phone": "", "direction": ""}),
    ],
)
def test_signalwire_call_parameters_from_form(
    request_data: Dict[str, str], expected: Dict[str, str]
) -> None:
    request_mock = MagicMock(spec=Request)
    request_mock.form = request_data
    request_mock.args = {}
    request_mock.headers = {}
    request_mock.json = None

    assert _signalwire_call_parameters(request_mock) == expected


def test_signalwire_call_parameters_from_query_args() -> None:
    request_mock = MagicMock(spec=Request)
    request_mock.form = {}
    request_mock.args = {
        CALL_SID_REQUEST_KEY: "sid-from-query",
        FROM_NUMBER_REQUEST_KEY: "+10001",
        TO_NUMBER_REQUEST_KEY: "+10002",
        DIRECTION_REQUEST_KEY: "outbound",
    }
    request_mock.headers = {}
    request_mock.json = None

    assert _signalwire_call_parameters(request_mock) == {
        "call_id": "sid-from-query",
        "user_phone": "+10001",
        "bot_phone": "+10002",
        "direction": "outbound",
    }


def test_signalwire_call_parameters_json_body_overrides_when_form_empty() -> None:
    request_mock = MagicMock(spec=Request)
    request_mock.form = {}
    request_mock.args = {}
    request_mock.headers = {"content-type": "application/json"}
    request_mock.json = {
        CALL_SID_REQUEST_KEY: "sid-json",
        FROM_NUMBER_REQUEST_KEY: "+20001",
        TO_NUMBER_REQUEST_KEY: "+20002",
        DIRECTION_REQUEST_KEY: "inbound",
    }

    assert _signalwire_call_parameters(request_mock) == {
        "call_id": "sid-json",
        "user_phone": "+20001",
        "bot_phone": "+20002",
        "direction": "inbound",
    }


###############################################################
# Map Input Message
###############################################################


async def test_map_input_message_media(
    input_channel: SignalWireInputChannel,
    async_ws: AsyncMock,
    sample_audio_bytes: bytes,
) -> None:
    payload_b64 = base64.b64encode(sample_audio_bytes).decode("ascii")
    media_message = json.dumps({"event": "media", "media": {"payload": payload_b64}})
    action = await input_channel.map_input_message(media_message, async_ws)
    assert isinstance(action, NewAudioAction)


async def test_map_input_message_stop(
    input_channel: SignalWireInputChannel, async_ws: AsyncMock
) -> None:
    action = await input_channel.map_input_message(
        json.dumps({"event": "stop"}), async_ws
    )
    assert isinstance(action, EndConversationAction)


@pytest.mark.parametrize(
    "bot_utterance_type", [StepType.COLLECT, StepType.REGULAR_UTTER, None]
)
@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_mark_with_no_registered_marker(
    input_channel: SignalWireInputChannel,
    async_ws: AsyncMock,
    bot_utterance_type: Optional[StepType],
) -> None:
    from rasa.core.channels.voice_stream.call_state import call_state

    mark_message = json.dumps(
        {"event": "mark", "mark": {"name": "unregistered_marker_id"}}
    )
    call_state.current_bot_utterance_type = bot_utterance_type
    previous_event_count = call_state.internal_queue.qsize()

    action = await input_channel.map_input_message(mark_message, async_ws)

    assert isinstance(action, ContinueConversationAction)
    assert call_state.current_bot_utterance_type == bot_utterance_type
    assert call_state.internal_queue.qsize() == previous_event_count


@pytest.mark.parametrize("step_type", [StepType.COLLECT, StepType.REGULAR_UTTER])
@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_mark_with_start_marker(
    input_channel: SignalWireInputChannel,
    async_ws: AsyncMock,
    step_type: StepType,
) -> None:
    from rasa.core.channels.voice_stream.call_state import call_state

    marker_id = "marker_start_1"
    marker = Marker(
        marker_id=marker_id, marker_type=MarkerType.START, step_type=step_type
    )
    call_state.set_marker(marker)
    mark_message = json.dumps({"event": "mark", "mark": {"name": marker_id}})

    action = await input_channel.map_input_message(mark_message, async_ws)

    assert isinstance(action, ContinueConversationAction)

    assert call_state.internal_queue.qsize() == 1
    event = call_state.internal_queue.get_nowait()
    assert isinstance(event, BotIsSpeaking)

    assert call_state.current_bot_utterance_type == step_type
    assert call_state.get_marker(marker_id) is None


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_mark_with_end_marker(
    input_channel: SignalWireInputChannel, async_ws: AsyncMock
) -> None:
    from rasa.core.channels.voice_stream.call_state import call_state

    marker_id = "marker_end_1"
    marker = Marker(
        marker_id=marker_id,
        marker_type=MarkerType.END,
        step_type=StepType.REGULAR_UTTER,
    )
    call_state.set_marker(marker)
    call_state.current_bot_utterance_type = StepType.REGULAR_UTTER
    mark_message = json.dumps({"event": "mark", "mark": {"name": marker_id}})

    action = await input_channel.map_input_message(mark_message, async_ws)

    assert isinstance(action, ContinueConversationAction)
    assert call_state.current_bot_utterance_type is None

    assert call_state.internal_queue.qsize() == 1
    event = call_state.internal_queue.get_nowait()
    assert isinstance(event, BotStoppedSpeaking)

    assert call_state.get_marker(marker_id) is None


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_mark_with_end_marker_should_hangup(
    input_channel: SignalWireInputChannel, async_ws: AsyncMock
) -> None:
    from rasa.core.channels.voice_stream.call_state import call_state

    marker_id = "marker_end_1"
    marker = Marker(
        marker_id=marker_id,
        marker_type=MarkerType.END,
        step_type=StepType.REGULAR_UTTER,
    )
    call_state.set_marker(marker)
    call_state.current_bot_utterance_type = StepType.REGULAR_UTTER
    mark_message = json.dumps({"event": "mark", "mark": {"name": marker_id}})

    call_state.should_hangup = True
    action = await input_channel.map_input_message(mark_message, async_ws)

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
    input_channel: SignalWireInputChannel,
    async_ws: AsyncMock,
    marker_type: MarkerType,
    current_bot_utterance_type: Optional[StepType],
) -> None:
    from rasa.core.channels.voice_stream.call_state import call_state

    marker_id = "marker_no_step"
    marker = Marker(marker_id=marker_id, marker_type=marker_type, step_type=None)
    call_state.set_marker(marker)
    mark_message = json.dumps({"event": "mark", "mark": {"name": marker_id}})

    previous_event_count = call_state.internal_queue.qsize()
    call_state.current_bot_utterance_type = current_bot_utterance_type

    action = await input_channel.map_input_message(mark_message, async_ws)

    assert isinstance(action, ContinueConversationAction)
    assert call_state.current_bot_utterance_type == current_bot_utterance_type
    assert call_state.internal_queue.qsize() == previous_event_count
    assert call_state.get_marker(marker_id) is None


@pytest.mark.parametrize(
    "current_bot_utterance_type", [StepType.REGULAR_UTTER, StepType.COLLECT, None]
)
@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_mark_with_intermediate_marker(
    input_channel: SignalWireInputChannel,
    async_ws: AsyncMock,
    current_bot_utterance_type: Optional[StepType],
) -> None:
    from rasa.core.channels.voice_stream.call_state import call_state

    marker_id = "marker_intermediate"
    marker = Marker(
        marker_id=marker_id,
        marker_type=MarkerType.INTERMEDIATE,
        step_type=StepType.COLLECT,
    )
    call_state.set_marker(marker)
    mark_message = json.dumps({"event": "mark", "mark": {"name": marker_id}})

    previous_event_count = call_state.internal_queue.qsize()
    call_state.current_bot_utterance_type = current_bot_utterance_type

    action = await input_channel.map_input_message(mark_message, async_ws)

    assert isinstance(action, ContinueConversationAction)
    assert call_state.internal_queue.qsize() == previous_event_count
    assert call_state.current_bot_utterance_type == current_bot_utterance_type
    assert call_state.get_marker(marker_id) is None


async def test_output_channel_audio_sending(
    signalwire_output_channel: SignalWireOutputChannel,
) -> None:
    audio_bytes = RasaAudioBytes(b"test_audio", format=L16_24KHZ)

    await signalwire_output_channel.send_audio_bytes("test_recipient", audio_bytes)
    mock_ws = signalwire_output_channel.voice_websocket
    assert mock_ws.send.called

    sent = mock_ws.send.call_args[0][0]
    parsed = json.loads(sent)
    assert parsed["event"] == "media"
    assert parsed["streamSid"] == "test_recipient"
    assert "payload" in parsed["media"]


@pytest.mark.parametrize(
    "marker_type", [MarkerType.START, MarkerType.INTERMEDIATE, MarkerType.END]
)
@pytest.mark.parametrize("step_type", [StepType.COLLECT, StepType.REGULAR_UTTER, None])
@pytest.mark.usefixtures("setup_call_state")
def test_create_marker_message(
    marker_type: MarkerType,
    step_type: Optional[StepType],
    signalwire_output_channel: SignalWireOutputChannel,
) -> None:
    marker_message = signalwire_output_channel.create_marker_message(
        marker_input=MarkerInput(
            recipient_id="recipient_stream_sid",
            marker_type=marker_type,
            step_type=step_type,
        )
    )

    assert isinstance(marker_message.message, str)
    parsed = json.loads(marker_message.message)
    assert parsed["event"] == "mark"
    assert parsed["streamSid"] == "recipient_stream_sid"
    assert "name" in parsed["mark"]
    assert len(marker_message.message_id) > 0


async def test_blueprint_health_endpoint(input_channel: SignalWireInputChannel) -> None:
    from sanic import Sanic

    app = Sanic("test_app")
    blueprint = input_channel.blueprint(AsyncMock())
    app.blueprint(blueprint)
    routes = [route.uri for route in blueprint.routes]
    prefix = "/rasa.core.channels.voice_stream.signalwire"
    assert prefix in routes
    assert f"{prefix}/webhook" in routes
    assert f"{prefix}/call_status" in routes
    assert f"{prefix}/websocket" in routes


@pytest.mark.parametrize(
    "credentials",
    [
        {"server_url": "example.com"},
        {"server_url": "example.com", "asr": {"name": "azure"}},
        {
            "server_url": "example.com",
            "asr": {"name": "azure"},
            "tts": {"name": "azure"},
            "username": "test_user",
        },
        {
            "server_url": "example.com",
            "asr": {"name": "azure"},
            "tts": {"name": "azure"},
            "password": "test_pass",
        },
    ],
)
def test_from_credentials_validation(credentials: Dict[str, Any]) -> None:
    with pytest.raises(InvalidConfigException):
        SignalWireInputChannel.from_credentials(credentials)


def test_from_credentials_success() -> None:
    credentials = {
        "server_url": "example.com",
        "asr": {"name": "azure"},
        "tts": {"name": "azure"},
        "username": "test_user",
        "password": "test_pass",
    }

    channel = SignalWireInputChannel.from_credentials(credentials)

    assert isinstance(channel, SignalWireInputChannel)
    assert channel.server_url == "example.com"
    assert channel.asr_config == {"name": "azure"}
    assert channel.tts_config == {"name": "azure"}
    assert channel.username == "test_user"
    assert channel.password == "test_pass"


@pytest.mark.parametrize(
    "server_url,expected_ws_url",
    [
        ("example.com", f"wss://example.com/{SIGNALWIRE_WEBSOCKET_PATH}"),
        ("http://example.com", f"wss://example.com/{SIGNALWIRE_WEBSOCKET_PATH}"),
        ("https://example.com", f"wss://example.com/{SIGNALWIRE_WEBSOCKET_PATH}"),
        (
            "http://example.com:8080",
            f"wss://example.com:8080/{SIGNALWIRE_WEBSOCKET_PATH}",
        ),
        (
            "example.com:8080",
            f"wss://example.com:8080/{SIGNALWIRE_WEBSOCKET_PATH}",
        ),
    ],
)
def test_websocket_stream_url(server_url: str, expected_ws_url: str) -> None:
    assert _websocket_stream_url(server_url) == expected_ws_url


def test_websocket_stream_url_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must be configured"):
        _websocket_stream_url("  ")


async def test_map_input_message_dtmf(
    input_channel: SignalWireInputChannel, async_ws: AsyncMock
) -> None:
    dtmf_message = json.dumps({"event": "dtmf", "dtmf": {"digit": "5"}})
    action = await input_channel.map_input_message(dtmf_message, async_ws)

    assert isinstance(action, DTMFInputAction)
    assert action.digit == "5"


@pytest.mark.parametrize(
    "dtmf_digit",
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "#", "*"],
)
async def test_map_input_message_dtmf_all_digits(
    input_channel: SignalWireInputChannel,
    async_ws: AsyncMock,
    dtmf_digit: str,
) -> None:
    dtmf_message = json.dumps({"event": "dtmf", "dtmf": {"digit": dtmf_digit}})
    action = await input_channel.map_input_message(dtmf_message, async_ws)

    assert isinstance(action, DTMFInputAction)
    assert action.digit == dtmf_digit


async def test_map_input_message_unknown_event(
    input_channel: SignalWireInputChannel, async_ws: AsyncMock
) -> None:
    unknown_message = json.dumps({"event": "unknown_event", "data": "something"})
    action = await input_channel.map_input_message(unknown_message, async_ws)

    assert isinstance(action, ContinueConversationAction)


async def test_interrupt_playback(
    input_channel: SignalWireInputChannel,
) -> None:
    ws = AsyncMock()
    params = CallParameters(
        call_id="c",
        user_phone="+1",
        bot_phone="+2",
        stream_id="MZstream123",
    )
    await input_channel.interrupt_playback(ws, params)

    ws.send.assert_called_once()
    payload = json.loads(ws.send.call_args[0][0])
    assert payload == {"event": "clear", "streamSid": "MZstream123"}
