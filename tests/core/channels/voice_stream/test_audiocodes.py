import json
from unittest.mock import AsyncMock, patch

import pytest

from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.audiocodes import AudiocodesVoiceInputChannel


@pytest.fixture
def input_channel() -> AudiocodesVoiceInputChannel:
    """Returns a default initialized AudiocodesVoiceInputChannel."""
    with patch(
        "rasa.core.channels.voice_stream.voice_channel.validate_voice_license_scope"
    ):
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
        yield input_channel


@pytest.fixture
def input_channel_no_token() -> AudiocodesVoiceInputChannel:
    """Returns a default initialized AudiocodesVoiceInputChannel without a token."""
    with patch(
        "rasa.core.channels.voice_stream.voice_channel.validate_voice_license_scope"
    ):
        server_url = "https://example.com"
        asr_config = {"name": "deepgram"}
        tts_config = {"name": "azure"}
        input_channel = AudiocodesVoiceInputChannel(
            token=None,
            server_url=server_url,
            asr_config=asr_config,
            tts_config=tts_config,
        )
        yield input_channel


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
def valid_initiate_message():
    return {
        "conversationId": "4a5b4b9d-dab7-42d0-a977-6740c9349588",
        "type": "initiate",
        "botName": "my_bot_name",
        "caller": "+1234567890",
        "expectAudioMessages": True,
        "supportedMediaFormats": ["raw/lpcm16", "raw/mulaw"],
    }


def test_channel_name(input_channel: AudiocodesVoiceInputChannel):
    """Tests that the channel name is properly set."""
    assert input_channel.name() == "ac_voice"


async def test_collect_call_parameters(
    input_channel: AudiocodesVoiceInputChannel, valid_initiate_message: dict
):
    """Tests the collection of call parameters from the initiate message."""
    websocket = AsyncMock()
    websocket.__aiter__.return_value = [json.dumps(valid_initiate_message)]

    call_parameters = await input_channel.collect_call_parameters(websocket)

    assert call_parameters is not None
    assert call_parameters.call_id == valid_initiate_message["conversationId"]
    assert call_parameters.user_phone == valid_initiate_message["caller"]


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
