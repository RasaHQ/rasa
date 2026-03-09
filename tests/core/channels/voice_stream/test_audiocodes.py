import base64
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.audio_bytes import L16_24KHZ
from rasa.core.channels.voice_stream.audiocodes import (
    PREFERRED_AUDIO_FORMAT,
    AudiocodesVoiceInputChannel,
    AudiocodesVoiceOutputChannel,
)
from rasa.core.channels.voice_stream.call_state import (
    BotIsSpeaking,
    BotStoppedSpeaking,
    _call_state,
)
from rasa.core.channels.voice_stream.voice_channel import (
    ContinueConversationAction,
    DTMFInputAction,
    EndConversationAction,
    NewAudioAction,
)
from rasa.shared.exceptions import RasaException


@pytest.fixture
def input_channel(
    mock_validate_voice_license_scope: None,
) -> AudiocodesVoiceInputChannel:
    """Returns a default initialized AudiocodesVoiceInputChannel."""
    server_url = "https://example.com"
    asr_config = {"name": "deepgram"}
    tts_config = {"name": "azure"}
    token = "test_token"
    input_channel = AudiocodesVoiceInputChannel(
        token=token,
        server_url=server_url,
        asr_config=asr_config,
        tts_config=tts_config,
    )
    return input_channel


@pytest.fixture
def input_channel_no_token(
    mock_validate_voice_license_scope: None,
) -> AudiocodesVoiceInputChannel:
    """Returns a default initialized AudiocodesVoiceInputChannel without a token."""
    server_url = "https://example.com"
    asr_config = {"name": "deepgram"}
    tts_config = {"name": "azure"}
    input_channel = AudiocodesVoiceInputChannel(
        token=None,
        server_url=server_url,
        asr_config=asr_config,
        tts_config=tts_config,
    )
    return input_channel


@pytest.fixture
def call_parameters() -> CallParameters:
    """Returns a sample CallParameters object."""
    return CallParameters(
        "test_conversation_id",
        "test_caller",
        "test_bot_name",
        stream_id="test_stream_id",
    )


@pytest.fixture
def start_activity_message() -> Dict[str, Any]:
    return {
        "conversation": "f010e998-4499-4ddb-80d4-fea137fd7b4d",
        "type": "activities",
        "activities": [
            {
                "id": "e54d4dfe-e1ff-4272-8c3d-4ec4f4294681",
                "timestamp": "2024-12-04T15:07:55.145Z",
                "language": "en-US",
                "type": "event",
                "name": "start",
                "parameters": {
                    "callee": "+493040739365",
                    "calleeHost": "20.113.51.15",
                    "caller": "+491604697810",
                    "callerHost": "sip.telnyx.eu",
                    "callerDisplayName": "+491604697810",
                    "vaigConversationId": "f010e998-4499-4ddb-80d4-fea137fd7b4d",
                },
            }
        ],
    }


def test_channel_name(input_channel: AudiocodesVoiceInputChannel):
    """Tests that the channel name is properly set."""
    assert input_channel.name() == "audiocodes_stream"


async def test_collect_call_parameters(
    input_channel: AudiocodesVoiceInputChannel, start_activity_message: dict
):
    """Tests the collection of call parameters from the initiate message."""
    websocket = AsyncMock()
    websocket.__aiter__.return_value = [json.dumps(start_activity_message)]

    call_parameters = await input_channel.collect_call_parameters(websocket)
    activity_params = start_activity_message["activities"][0]["parameters"]
    assert call_parameters is not None
    assert call_parameters.call_id == activity_params["vaigConversationId"]
    assert call_parameters.user_phone == activity_params["caller"]


def test_is_token_valid(
    input_channel: AudiocodesVoiceInputChannel,
    input_channel_no_token: AudiocodesVoiceInputChannel,
):
    """Tests the token validation logic."""
    assert input_channel._is_token_valid(None) is False
    assert input_channel._is_token_valid("invalid_token") is False
    assert input_channel._is_token_valid("test_token") is True

    # for a channel without a token, all tokens are considered valid
    assert input_channel_no_token._is_token_valid(None) is True
    assert input_channel_no_token._is_token_valid("invalid_token") is True
    assert input_channel_no_token._is_token_valid("test_token") is True


@pytest.mark.parametrize(
    "input_data",
    [
        {
            "token": "test_token",
            "server_url": "https://example.com",
            "asr": {"name": "deepgram"},
            "tts": {"name": "azure"},
        },
        {
            "token": None,
            "server_url": "https://example.com",
            "asr": {"name": "deepgram"},
            "tts": {"name": "azure"},
        },
    ],
)
@pytest.mark.usefixtures("mock_validate_voice_license_scope")
def test_from_credentials(input_data: dict):
    """Tests the from_credentials method."""
    channel = AudiocodesVoiceInputChannel.from_credentials(
        input_data,
    )

    assert isinstance(channel, AudiocodesVoiceInputChannel)
    assert channel.token == input_data["token"]
    assert channel.server_url == input_data["server_url"]
    assert channel.asr_config == input_data["asr"]
    assert channel.tts_config == input_data["tts"]


@pytest.mark.parametrize(
    "config",
    [
        None,  # No credentials
        {},  # Empty credentials
        {
            "asr": {"name": "deepgram"},
            "tts": {"name": "azure"},
        },
        {
            "server_url": "https://example.com",
            "asr": {"name": "deepgram"},
        },
        {
            "server_url": "https://example.com",
            "tts": {"name": "azure"},
        },
    ],
)
def test_invalid_credentials(
    config: Dict[str, str],
):
    """Test creation of TwilioMediaStreamsInputChannel with invalid credentials."""
    with pytest.raises(RasaException):
        AudiocodesVoiceInputChannel.from_credentials(config)


async def test_map_input_message_dtmf(input_channel: AudiocodesVoiceInputChannel):
    """Test handling of DTMF input messages."""
    dtmf_message = {
        "type": "activities",
        "activities": [
            {
                "name": "DTMF",
                "value": "5",
            }
        ],
    }
    websocket = AsyncMock()
    action = await input_channel.map_input_message(json.dumps(dtmf_message), websocket)

    assert isinstance(action, DTMFInputAction)
    assert action.digit == "5"


@pytest.mark.parametrize(
    "dtmf_digit",
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "#", "*"],
)
async def test_map_input_message_dtmf_all_digits(
    input_channel: AudiocodesVoiceInputChannel, dtmf_digit: str
):
    """Test handling of all valid DTMF digits."""
    dtmf_message = {
        "type": "activities",
        "activities": [
            {
                "name": "DTMF",
                "value": dtmf_digit,
            }
        ],
    }
    websocket = AsyncMock()
    action = await input_channel.map_input_message(json.dumps(dtmf_message), websocket)

    assert isinstance(action, DTMFInputAction)
    assert action.digit == dtmf_digit


async def test_map_input_message_unknown_activity(
    input_channel: AudiocodesVoiceInputChannel,
):
    """Test handling of unknown activity types."""
    unknown_message = {
        "type": "activities",
        "activities": [
            {
                "name": "unknown_activity",
                "value": "something",
            }
        ],
    }
    websocket = AsyncMock()
    action = await input_channel.map_input_message(
        json.dumps(unknown_message), websocket
    )

    assert isinstance(action, ContinueConversationAction)


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_start_activity_returns_continue(
    input_channel: AudiocodesVoiceInputChannel,
):
    """start activity is handled in collect_call_parameters; map_input_message
    must return ContinueConversationAction without raising."""
    start_message = {
        "type": "activities",
        "activities": [{"name": "start", "parameters": {}}],
    }
    websocket = AsyncMock()
    action = await input_channel.map_input_message(json.dumps(start_message), websocket)

    assert isinstance(action, ContinueConversationAction)


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_play_finished_enqueues_bot_stopped_speaking(
    input_channel: AudiocodesVoiceInputChannel,
):
    """playFinished activity must enqueue a BotStoppedSpeaking message."""
    play_finished_message = {
        "type": "activities",
        "activities": [{"name": "playFinished", "streamId": "1"}],
    }
    websocket = AsyncMock()
    await input_channel.map_input_message(json.dumps(play_finished_message), websocket)

    internal_queue = _call_state.get().internal_queue
    assert not internal_queue.empty()
    message = internal_queue.get_nowait()
    assert isinstance(message, BotStoppedSpeaking)


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_play_finished_returns_continue(
    input_channel: AudiocodesVoiceInputChannel,
):
    """playFinished activity (no hangup) must return ContinueConversationAction."""
    play_finished_message = {
        "type": "activities",
        "activities": [{"name": "playFinished", "streamId": "1"}],
    }
    websocket = AsyncMock()
    action = await input_channel.map_input_message(
        json.dumps(play_finished_message), websocket
    )

    assert isinstance(action, ContinueConversationAction)


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_play_finished_with_hangup_sends_hangup(
    input_channel: AudiocodesVoiceInputChannel,
):
    """playFinished activity when should_hangup is True must call _send_hangup."""
    _call_state.get().should_hangup = True

    play_finished_message = {
        "conversationId": "conv-123",
        "type": "activities",
        "activities": [{"name": "playFinished", "streamId": "1"}],
    }
    websocket = AsyncMock()

    with patch.object(input_channel, "_send_hangup") as mock_send_hangup:
        action = await input_channel.map_input_message(
            json.dumps(play_finished_message), websocket
        )

    mock_send_hangup.assert_called_once()
    # Conversation must continue; session.end from Audiocodes ends it later
    assert isinstance(action, ContinueConversationAction)


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_user_stream_start_returns_continue(
    input_channel: AudiocodesVoiceInputChannel,
):
    """userStream.start must send recognition.started and return Continue."""
    user_stream_start = {"type": "userStream.start", "streamId": "1"}
    websocket = MagicMock()
    websocket.send = AsyncMock()

    action = await input_channel.map_input_message(
        json.dumps(user_stream_start), websocket
    )

    websocket.send.assert_called_once_with(json.dumps({"type": "userStream.started"}))
    assert isinstance(action, ContinueConversationAction)


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_user_stream_chunk_returns_new_audio_action(
    input_channel: AudiocodesVoiceInputChannel,
):
    """userStream.chunk must decode the audio and return a NewAudioAction."""
    # Create a small raw PCM-like payload encoded as base64
    raw_audio = b"\x00\x01\x02\x03"
    encoded_audio = base64.b64encode(raw_audio).decode("utf-8")
    user_stream_chunk = {"type": "userStream.chunk", "audioChunk": encoded_audio}
    websocket = AsyncMock()

    action = await input_channel.map_input_message(
        json.dumps(user_stream_chunk), websocket
    )

    assert isinstance(action, NewAudioAction)
    assert action.audio_bytes.data == raw_audio


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_user_stream_stop_returns_continue(
    input_channel: AudiocodesVoiceInputChannel,
):
    """userStream.stop must send recognition.stopped and return Continue."""
    user_stream_stop = {"type": "userStream.stop", "streamId": "1"}
    websocket = MagicMock()
    websocket.send = AsyncMock()

    action = await input_channel.map_input_message(
        json.dumps(user_stream_stop), websocket
    )

    websocket.send.assert_called_once_with(json.dumps({"type": "userStream.stopped"}))
    assert isinstance(action, ContinueConversationAction)


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_session_resume_returns_continue(
    input_channel: AudiocodesVoiceInputChannel,
):
    """session.resume must call _send_accepted and return ContinueConversationAction."""
    session_resume = {
        "type": "session.resume",
        "supportedMediaFormats": [PREFERRED_AUDIO_FORMAT],
    }
    websocket = MagicMock()
    websocket.send = AsyncMock()

    action = await input_channel.map_input_message(
        json.dumps(session_resume), websocket
    )

    websocket.send.assert_called_once_with(
        json.dumps(
            {
                "type": "session.accepted",
                "mediaFormat": PREFERRED_AUDIO_FORMAT,
            }
        )
    )
    assert isinstance(action, ContinueConversationAction)


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_session_end_returns_end_conversation(
    input_channel: AudiocodesVoiceInputChannel,
):
    """session.end must return EndConversationAction."""
    session_end = {"type": "session.end"}
    websocket = AsyncMock()

    action = await input_channel.map_input_message(json.dumps(session_end), websocket)

    assert isinstance(action, EndConversationAction)


@pytest.mark.usefixtures("setup_call_state")
async def test_map_input_message_unknown_type_returns_continue(
    input_channel: AudiocodesVoiceInputChannel,
):
    """An entirely unknown message type must return ContinueConversationAction."""
    unknown_message = {"type": "some.unknown.type", "data": "irrelevant"}
    websocket = AsyncMock()

    action = await input_channel.map_input_message(
        json.dumps(unknown_message), websocket
    )

    assert isinstance(action, ContinueConversationAction)


@pytest.fixture
def mock_websocket() -> AsyncMock:
    """Mock websocket for output channel tests."""
    return AsyncMock()


@pytest.fixture
def output_channel(mock_websocket: AsyncMock) -> AudiocodesVoiceOutputChannel:
    """Returns an AudiocodesVoiceOutputChannel with a mock websocket."""
    return AudiocodesVoiceOutputChannel(
        voice_websocket=mock_websocket,
        tts_engine=MagicMock(),
        tts_cache=MagicMock(),
        audio_format=L16_24KHZ,
    )


@pytest.mark.usefixtures("setup_call_state")
async def test_send_start_marker_sends_play_stream_start(
    output_channel: AudiocodesVoiceOutputChannel,
    mock_websocket: AsyncMock,
):
    """send_start_marker must send a playStream.start message over the websocket."""
    await output_channel.send_start_marker("recipient")

    mock_websocket.send.assert_awaited_once()
    sent_payload = json.loads(mock_websocket.send.call_args[0][0])
    assert sent_payload["type"] == "playStream.start"
    assert sent_payload["mediaFormat"] == PREFERRED_AUDIO_FORMAT
    assert "streamId" in sent_payload


@pytest.mark.usefixtures("setup_call_state")
async def test_send_start_marker_increments_stream_id(
    output_channel: AudiocodesVoiceOutputChannel,
    mock_websocket: AsyncMock,
):
    """Each call to send_start_marker must use a higher streamId than the previous."""
    await output_channel.send_start_marker("recipient")
    first_payload = json.loads(mock_websocket.send.call_args_list[0][0][0])
    first_stream_id = int(first_payload["streamId"])

    # Reset the websocket send mock before the second call
    mock_websocket.send.reset_mock()
    await output_channel.send_start_marker("recipient")
    second_payload = json.loads(mock_websocket.send.call_args_list[0][0][0])
    second_stream_id = int(second_payload["streamId"])

    assert second_stream_id > first_stream_id


@pytest.mark.usefixtures("setup_call_state")
async def test_send_start_marker_puts_bot_is_speaking_on_queue(
    output_channel: AudiocodesVoiceOutputChannel,
):
    """send_start_marker must enqueue a BotIsSpeaking message."""
    await output_channel.send_start_marker("recipient")

    internal_queue = _call_state.get().internal_queue
    assert not internal_queue.empty()
    message = internal_queue.get_nowait()
    assert isinstance(message, BotIsSpeaking)


@pytest.mark.usefixtures("setup_call_state")
async def test_send_start_marker_stream_id_starts_at_one(
    output_channel: AudiocodesVoiceOutputChannel,
    mock_websocket: AsyncMock,
):
    """The first send_start_marker call should use streamId '1'."""
    await output_channel.send_start_marker("recipient")

    sent_payload = json.loads(mock_websocket.send.call_args[0][0])
    assert sent_payload["streamId"] == "1"
