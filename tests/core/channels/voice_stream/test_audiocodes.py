import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.audiocodes import AudiocodesVoiceInputChannel


@pytest.fixture
def mock_validate_voice_license_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock the validate_voice_license_scope function."""
    monkeypatch.setattr(
        "rasa.core.channels.voice_stream.voice_channel.validate_voice_license_scope",
        MagicMock(),
    )


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
def start_activity_message():
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
def test_from_credentials(input_data: dict, mock_validate_voice_license_scope):
    """Tests the from_credentials method."""
    channel = AudiocodesVoiceInputChannel.from_credentials(
        input_data,
    )

    assert isinstance(channel, AudiocodesVoiceInputChannel)
    assert channel.token == input_data["token"]
    assert channel.server_url == input_data["server_url"]
    assert channel.asr_config == input_data["asr"]
    assert channel.tts_config == input_data["tts"]
