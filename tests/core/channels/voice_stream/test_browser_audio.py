import base64
import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    AudioFormat,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.browser_audio import (
    BrowserAudioInputChannel,
    BrowserAudioOutputChannel,
)
from rasa.core.channels.voice_stream.call_state import (
    BotIsSpeaking,
    BotStoppedSpeaking,
    Marker,
    MarkerType,
    StepType,
    call_state,
)
from rasa.core.channels.voice_stream.tts import TTSCache
from rasa.core.channels.voice_stream.voice_channel import (
    ContinueConversationAction,
    EndConversationAction,
    MarkerInput,
    MarkerMessageOutput,
    NewAudioAction,
)


@pytest.fixture
def input_channel() -> BrowserAudioInputChannel:
    return BrowserAudioInputChannel(
        server_url="localhost",
        asr_config={"name": "dummy_asr"},
        tts_config={"name": "dummy_tts"},
        recording=False,
    )


@pytest.fixture
def output_channel(mulaw_format) -> BrowserAudioOutputChannel:
    return BrowserAudioOutputChannel(
        MagicMock(), MagicMock(), TTSCache(max_size=2000), audio_format=mulaw_format
    )


@pytest.fixture
def mock_websocket() -> AsyncMock:
    ws = AsyncMock()
    return ws


@pytest.fixture
def sample_audio_bytes() -> bytes:
    # 1 second of silence, 16-bit (2 bytes/sample)
    return bytes([0x00] * 16000)


def _marker_message(marker_id: str, marker_type: str, step_type: str) -> str:
    """Build a JSON marker acknowledgement message."""
    return json.dumps(
        {"marker": marker_id, "marker_type": marker_type, "step_type": step_type}
    )


def test_channel_bytes_to_rasa_audio_bytes_mulaw_8khz(
    input_channel: BrowserAudioInputChannel, sample_audio_bytes: bytes
):
    input_channel.audio_format = MULAW_8KHZ
    rasa_audio = input_channel.channel_bytes_to_rasa_audio_bytes(sample_audio_bytes)
    assert isinstance(rasa_audio, RasaAudioBytes)
    # Mulaw 8kHz should result in half the number of samples when converted to L16
    assert len(rasa_audio) == len(sample_audio_bytes) // 2


def test_channel_bytes_to_rasa_audio_bytes_l16_24khz(
    input_channel: BrowserAudioInputChannel, sample_audio_bytes: bytes
):
    input_channel.audio_format = L16_24KHZ
    rasa_audio = input_channel.channel_bytes_to_rasa_audio_bytes(sample_audio_bytes)
    assert isinstance(rasa_audio, RasaAudioBytes)
    # L16 24kHz should have the same number of bytes as the input
    assert len(rasa_audio) == len(sample_audio_bytes)


def test_channel_bytes_to_rasa_audio_bytes_l16_48khz(
    input_channel: BrowserAudioInputChannel, sample_audio_bytes: bytes
):
    input_channel.audio_format = L16_48KHZ
    rasa_audio = input_channel.channel_bytes_to_rasa_audio_bytes(sample_audio_bytes)
    assert isinstance(rasa_audio, RasaAudioBytes)
    # L16 48kHz should have the same number of bytes as the input
    assert len(rasa_audio) == len(sample_audio_bytes)


def test_rasa_audio_bytes_to_channel_bytes(
    output_channel: BrowserAudioOutputChannel,
    sample_audio_bytes: bytes,
    mulaw_format: AudioFormat,
):
    rasa_audio = RasaAudioBytes(sample_audio_bytes[:8000], format=mulaw_format)
    channel_bytes = output_channel.rasa_audio_bytes_to_channel_bytes(rasa_audio)
    assert isinstance(channel_bytes, bytes)


def test_channel_bytes_to_message(
    output_channel: BrowserAudioOutputChannel, sample_audio_bytes: bytes
):
    msg = output_channel.channel_bytes_to_message(
        "recipient", sample_audio_bytes[:8000]
    )
    data = json.loads(msg)
    assert "audio" in data
    decoded = base64.b64decode(data["audio"])
    assert decoded == sample_audio_bytes[:8000]


async def test_map_input_message_audio(
    input_channel: BrowserAudioInputChannel,
    sample_audio_bytes: bytes,
    mock_websocket: AsyncMock,
):
    audio_b64 = base64.b64encode(sample_audio_bytes[:8000]).decode("utf-8")
    msg = json.dumps({"audio": audio_b64})
    action = await input_channel.map_input_message(msg, mock_websocket)
    assert isinstance(action, NewAudioAction)


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_marker_unknown_id_returns_continue(
    input_channel: BrowserAudioInputChannel,
) -> None:
    """A marker message whose ID is not in call_state should
    continue the conversation."""
    ws = AsyncMock()
    message = _marker_message("nonexistent-id", "end", "collect")
    action = await input_channel.map_input_message(message, ws)

    assert isinstance(action, ContinueConversationAction)
    assert call_state.internal_queue.qsize() == 0


@pytest.mark.parametrize(
    "current_bot_utterance_type", [StepType.REGULAR_UTTER, StepType.COLLECT, None]
)
@pytest.mark.parametrize("marker_type", [MarkerType.START, MarkerType.END])
@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_marker_no_step_type(
    input_channel: BrowserAudioInputChannel,
    marker_type: MarkerType,
    current_bot_utterance_type: Optional[StepType],
) -> None:
    """A registered marker without a step_type should be removed and the
    conversation should continue without changing bot-speaking state."""
    marker_id = "no-step-marker"
    call_state.set_marker(
        Marker(marker_id=marker_id, marker_type=marker_type, step_type=None)
    )
    call_state.current_bot_utterance_type = current_bot_utterance_type

    ws = AsyncMock()
    previous_event_count = call_state.internal_queue.qsize()
    message = json.dumps({"marker": marker_id, "marker_type": "end", "step_type": None})
    action = await input_channel.map_input_message(message, ws)

    assert isinstance(action, ContinueConversationAction)
    assert call_state.get_marker(marker_id) is None

    assert call_state.current_bot_utterance_type == current_bot_utterance_type
    assert call_state.internal_queue.qsize() == previous_event_count


@pytest.mark.parametrize("step_type", [StepType.COLLECT, StepType.REGULAR_UTTER])
@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_marker_start(
    input_channel: BrowserAudioInputChannel,
    step_type: StepType,
) -> None:
    """A START marker with a step_type should set is_bot_speaking=True and
    record current_bot_utterance_type."""
    marker_id = "start-collect"
    call_state.set_marker(
        Marker(
            marker_id=marker_id,
            marker_type=MarkerType.START,
            step_type=step_type,
        )
    )

    ws = AsyncMock()
    action = await input_channel.map_input_message(
        _marker_message(marker_id, "start", "collect"), ws
    )

    assert isinstance(action, ContinueConversationAction)

    assert call_state.internal_queue.qsize() == 1
    event = call_state.internal_queue.get_nowait()
    assert isinstance(event, BotIsSpeaking)

    assert call_state.current_bot_utterance_type == step_type
    assert call_state.get_marker(marker_id) is None


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_marker_end_no_hangup(
    input_channel: BrowserAudioInputChannel,
) -> None:
    """An END marker without a pending hangup should queue BotStoppedSpeaking
    and clear current_bot_utterance_type."""
    marker_id = "end-no-hangup"
    call_state.set_marker(
        Marker(
            marker_id=marker_id,
            marker_type=MarkerType.END,
            step_type=StepType.REGULAR_UTTER,
        )
    )
    call_state.current_bot_utterance_type = StepType.REGULAR_UTTER
    call_state.should_hangup = False

    ws = AsyncMock()
    action = await input_channel.map_input_message(
        _marker_message(marker_id, "end", "regular_utter"), ws
    )

    assert isinstance(action, ContinueConversationAction)

    assert call_state.internal_queue.qsize() == 1
    event = call_state.internal_queue.get_nowait()
    assert isinstance(event, BotStoppedSpeaking)

    assert call_state.current_bot_utterance_type is None
    assert call_state.get_marker(marker_id) is None


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_marker_end_with_hangup_returns_end_conversation(
    input_channel: BrowserAudioInputChannel,
) -> None:
    """An END marker when should_hangup=True should return EndConversationAction
    and queue BotStoppedSpeaking."""
    marker_id = "end-hangup"
    call_state.set_marker(
        Marker(
            marker_id=marker_id,
            marker_type=MarkerType.END,
            step_type=StepType.COLLECT,
        )
    )
    call_state.should_hangup = True

    ws = AsyncMock()
    action = await input_channel.map_input_message(
        _marker_message(marker_id, "end", "collect"), ws
    )

    assert isinstance(action, EndConversationAction)

    assert call_state.internal_queue.qsize() == 1
    event = call_state.internal_queue.get_nowait()
    assert isinstance(event, BotStoppedSpeaking)

    assert call_state.current_bot_utterance_type is None
    assert call_state.get_marker(marker_id) is None


@pytest.mark.parametrize("step_type", [StepType.COLLECT, StepType.REGULAR_UTTER])
@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_marker_intermediate_does_not_change_state(
    input_channel: BrowserAudioInputChannel,
    step_type: StepType,
) -> None:
    """An INTERMEDIATE marker should be removed but not change bot-speaking state."""
    marker_id = "intermediate-marker"
    call_state.set_marker(
        Marker(
            marker_id=marker_id,
            marker_type=MarkerType.INTERMEDIATE,
            step_type=step_type,
        )
    )
    call_state.current_bot_utterance_type = step_type

    ws = AsyncMock()
    previous_event_count = call_state.internal_queue.qsize()

    action = await input_channel.map_input_message(
        _marker_message(marker_id, "intermediate", "regular_utter"), ws
    )

    assert isinstance(action, ContinueConversationAction)
    # INTERMEDIATE does not trigger START/END branches so is_bot_speaking is unchanged
    assert call_state.current_bot_utterance_type == step_type

    assert call_state.internal_queue.qsize() == previous_event_count

    assert call_state.get_marker(marker_id) is None


# ---------------------------------------------------------------------------
# Unknown / unrecognised messages
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_unknown_key_returns_continue(
    input_channel: BrowserAudioInputChannel,
) -> None:
    """A JSON message with neither 'audio' nor 'marker' key should continue
    the conversation unchanged."""
    ws = AsyncMock()
    message = json.dumps({"event": "connected", "protocol": "Call"})
    action = await input_channel.map_input_message(message, ws)

    assert isinstance(action, ContinueConversationAction)


# ---------------------------------------------------------------------------
# BrowserAudioOutputChannel.create_marker_message
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("setup_call_state")
def test_create_marker_message_returns_marker_message_output(
    output_channel: BrowserAudioOutputChannel,
) -> None:
    """create_marker_message should return a MarkerMessageOutput instance."""
    marker_input = MarkerInput(
        recipient_id="user-1",
        marker_type=MarkerType.START,
        step_type=StepType.COLLECT,
    )
    result = output_channel.create_marker_message(marker_input)
    assert isinstance(result, MarkerMessageOutput)


@pytest.mark.usefixtures("setup_call_state")
def test_create_marker_message_id_is_hex_uuid(
    output_channel: BrowserAudioOutputChannel,
) -> None:
    """message_id should be a 32-character hexadecimal UUID."""
    marker_input = MarkerInput(
        recipient_id="user-1",
        marker_type=MarkerType.START,
        step_type=StepType.COLLECT,
    )
    result = output_channel.create_marker_message(marker_input)
    assert len(result.message_id) == 32
    # Raises ValueError if not a valid hex string
    int(result.message_id, 16)


@pytest.mark.usefixtures("setup_call_state")
def test_create_marker_message_id_matches_marker_in_json(
    output_channel: BrowserAudioOutputChannel,
) -> None:
    """The message_id on the return value should equal the 'marker' field in the
    serialised JSON message."""
    marker_input = MarkerInput(
        recipient_id="user-1",
        marker_type=MarkerType.END,
        step_type=StepType.REGULAR_UTTER,
    )
    result = output_channel.create_marker_message(marker_input)
    message_data = json.loads(result.message)
    assert message_data["marker"] == result.message_id


@pytest.mark.usefixtures("setup_call_state")
@pytest.mark.parametrize(
    "marker_type, step_type, expected_marker_type, expected_step_type",
    [
        (MarkerType.START, StepType.COLLECT, "start", "collect"),
        (MarkerType.END, StepType.REGULAR_UTTER, "end", "regular_utter"),
        (MarkerType.INTERMEDIATE, StepType.COLLECT, "intermediate", "collect"),
    ],
)
def test_create_marker_message_json_contains_correct_fields(
    output_channel: BrowserAudioOutputChannel,
    marker_type: MarkerType,
    step_type: StepType,
    expected_marker_type: str,
    expected_step_type: str,
) -> None:
    """The serialised JSON should contain the correct marker_type and step_type."""
    marker_input = MarkerInput(
        recipient_id="user-1",
        marker_type=marker_type,
        step_type=step_type,
    )
    result = output_channel.create_marker_message(marker_input)
    message_data = json.loads(result.message)
    assert message_data["marker_type"] == expected_marker_type
    assert message_data["step_type"] == expected_step_type


@pytest.mark.usefixtures("setup_call_state")
def test_create_marker_message_start_stored_in_call_state(
    output_channel: BrowserAudioOutputChannel,
) -> None:
    """A START marker should be stored in call_state with the correct attributes."""
    marker_input = MarkerInput(
        recipient_id="user-1",
        marker_type=MarkerType.START,
        step_type=StepType.COLLECT,
    )
    result = output_channel.create_marker_message(marker_input)
    stored = call_state.get_marker(result.message_id)

    assert stored is not None
    assert stored.marker_id == result.message_id
    assert stored.marker_type == MarkerType.START
    assert stored.step_type == StepType.COLLECT


@pytest.mark.usefixtures("setup_call_state")
def test_create_marker_message_end_stored_in_call_state(
    output_channel: BrowserAudioOutputChannel,
) -> None:
    """An END marker should be stored in call_state with the correct attributes."""
    marker_input = MarkerInput(
        recipient_id="user-1",
        marker_type=MarkerType.END,
        step_type=StepType.REGULAR_UTTER,
    )
    result = output_channel.create_marker_message(marker_input)
    stored = call_state.get_marker(result.message_id)

    assert stored is not None
    assert stored.marker_id == result.message_id
    assert stored.marker_type == MarkerType.END
    assert stored.step_type == StepType.REGULAR_UTTER


@pytest.mark.usefixtures("setup_call_state")
def test_create_marker_message_intermediate_not_stored_in_call_state(
    output_channel: BrowserAudioOutputChannel,
) -> None:
    """An INTERMEDIATE marker should NOT be stored in call_state."""
    marker_input = MarkerInput(
        recipient_id="user-1",
        marker_type=MarkerType.INTERMEDIATE,
        step_type=StepType.COLLECT,
    )
    result = output_channel.create_marker_message(marker_input)
    assert call_state.get_marker(result.message_id) is None


@pytest.mark.usefixtures("setup_call_state")
def test_create_marker_message_start_without_step_type_stored_in_call_state(
    output_channel: BrowserAudioOutputChannel,
) -> None:
    """A START marker with no step_type should still be stored in call_state."""
    marker_input = MarkerInput(
        recipient_id="user-1",
        marker_type=MarkerType.START,
    )
    result = output_channel.create_marker_message(marker_input)
    stored = call_state.get_marker(result.message_id)

    assert stored is not None
    assert stored.step_type is None


@pytest.mark.usefixtures("setup_call_state")
def test_create_marker_message_latency_included_when_all_values_present(
    output_channel: BrowserAudioOutputChannel,
) -> None:
    """When all latency fields in call_state are set, the serialised message
    should contain a 'latency' key."""
    call_state.asr_latency_ms = 100.0
    call_state.rasa_processing_latency_ms = 200.0
    call_state.tts_first_byte_latency_ms = 50.0
    call_state.tts_complete_latency_ms = 300.0

    marker_input = MarkerInput(
        recipient_id="user-1",
        marker_type=MarkerType.END,
        step_type=StepType.COLLECT,
    )
    result = output_channel.create_marker_message(marker_input)
    message_data = json.loads(result.message)

    assert "latency" in message_data


@pytest.mark.usefixtures("setup_call_state")
def test_create_marker_message_latency_excluded_when_values_are_none(
    output_channel: BrowserAudioOutputChannel,
) -> None:
    """When latency fields in call_state are None (default), the serialised
    message should not contain a 'latency' key with null values."""
    # Default call_state has all latency fields as None
    marker_input = MarkerInput(
        recipient_id="user-1",
        marker_type=MarkerType.START,
        step_type=StepType.REGULAR_UTTER,
    )
    result = output_channel.create_marker_message(marker_input)
    message_data = json.loads(result.message)

    assert "latency" not in message_data


@pytest.mark.usefixtures("setup_call_state")
def test_create_marker_message_each_call_produces_unique_message_id(
    output_channel: BrowserAudioOutputChannel,
) -> None:
    """Successive calls to create_marker_message should produce distinct message IDs."""
    marker_input = MarkerInput(
        recipient_id="user-1",
        marker_type=MarkerType.INTERMEDIATE,
        step_type=StepType.COLLECT,
    )
    result_a = output_channel.create_marker_message(marker_input)
    result_b = output_channel.create_marker_message(marker_input)

    assert result_a.message_id != result_b.message_id


@pytest.mark.asyncio
async def test_collect_call_parameters(
    input_channel: BrowserAudioInputChannel, mock_websocket: AsyncMock
):
    params = await input_channel.collect_call_parameters(mock_websocket)
    assert isinstance(params, CallParameters)
    assert params.call_id.startswith("inspect-")


@pytest.mark.asyncio
async def test_interrupt_playback(
    input_channel: BrowserAudioInputChannel, mock_websocket: AsyncMock
):
    await input_channel.interrupt_playback(
        mock_websocket, CallParameters("cid", "u", "b")
    )
    mock_websocket.send.assert_called()


@pytest.mark.asyncio
async def test_blueprint_health_endpoint(input_channel: BrowserAudioInputChannel):
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
