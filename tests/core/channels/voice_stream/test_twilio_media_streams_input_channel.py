import asyncio
import base64
import json
from dataclasses import asdict
from typing import List
from unittest.mock import AsyncMock

import pytest

from rasa.core.channels import TwilioMediaStreamsInputChannel, UserMessage
from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.twilio_media_streams import (
    TwilioMediaStreamsOutputChannel,
)
from rasa.core.channels.voice_stream.util import read_wav_to_rasa_audio_bytes
from rasa.core.channels.voice_stream.voice_channel import (
    NewAudioAction,
    tts_engine_from_config,
)


@pytest.fixture
def input_channel() -> TwilioMediaStreamsInputChannel:
    server_url = "https://example.com"
    asr_config = {"name": "deepgram"}
    tts_config = {"name": "azure"}
    input_channel = TwilioMediaStreamsInputChannel(server_url, asr_config, tts_config)
    yield input_channel


@pytest.fixture
def call_parameters() -> CallParameters:
    yield CallParameters(
        "test_id", "test_phone", "test_bot_phone", stream_id="test_stream_id"
    )


def create_twilio_media_streams_start_message(
    bot_phone: str, user_phone: str, call_id: str, stream_sid: str, direction: str
) -> str:
    return json.dumps(
        {
            "event": "start",
            "sequenceNumber": "1",
            "start": {
                "accountSid": "ACbc2d4fd426ce33de19d54bdcd6e41186",
                "streamSid": "MZcdce5426d49ccf48c7b0d0ab86a63d52",
                "callSid": "CAa874cb4d1ac15290b51b28c91d467812",
                "tracks": ["inbound"],
                "mediaFormat": {
                    "encoding": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "channels": 1,
                },
                "customParameters": {
                    "direction": direction,
                    "call_id": call_id,
                    "user_phone": user_phone,
                    "bot_phone": bot_phone,
                },
            },
            "streamSid": stream_sid,
        }
    )


def create_twilio_media_streams_media_messages(
    audio_file_path: str, stream_sid: str
) -> List[str]:
    audio_bytes = read_wav_to_rasa_audio_bytes(audio_file_path)
    messages = []
    chunk_size = 1024
    i = 0
    offset = 0
    while offset < len(audio_bytes):
        payload = base64.b64encode(audio_bytes[offset : offset + chunk_size])
        payload = payload.decode("utf-8")
        messages.append(
            json.dumps(
                {
                    "event": "media",
                    "sequenceNumber": str(i + 1),
                    "media": {
                        "track": "inbound",
                        "chunk": str(i),
                        "timestamp": str(offset // 8),
                        "payload": payload,
                    },
                    "streamSid": stream_sid,
                }
            )
        )
        i += 1
        offset += chunk_size
    return messages


def create_twilio_media_streams_input_stream(audio_data_path: str) -> AsyncMock:
    bot_phone = "+49123456789"
    user_phone = "+49176124567"
    stream_id = "MZcdce5426d49ccf48c7b0d0ab86a63d52"
    call_id = "CAa874cb4d1ac15290b51b28c91d467812"
    direction = "inbound"
    channel_start_message = create_twilio_media_streams_start_message(
        bot_phone, user_phone, call_id, stream_id, direction
    )
    media_messages = create_twilio_media_streams_media_messages(
        audio_data_path + "/01.wav", stream_id
    )
    stop_message = json.dumps({"event": "stop"})
    websocket = AsyncMock()

    def spaced_return(messages: List[str], timeout: float = 1024 / 8000):
        async def wrapped(self):
            for message in messages:
                yield message
                await asyncio.sleep(timeout)

        return wrapped

    websocket.__aiter__ = spaced_return(
        [channel_start_message] + media_messages + [stop_message]
    )
    return websocket


def test_channel_creation(input_channel: TwilioMediaStreamsInputChannel):
    assert input_channel.tts_cache is not None


def test_fitting_output_channel_creation(input_channel: TwilioMediaStreamsInputChannel):
    tts_config = {"name": "azure"}
    websocket = AsyncMock()
    tts_engine = tts_engine_from_config(tts_config)
    output_channel = input_channel.create_output_channel(websocket, tts_engine)
    assert isinstance(output_channel, TwilioMediaStreamsOutputChannel)
    assert output_channel.tts_cache == input_channel.tts_cache


def test_channel_name():
    assert TwilioMediaStreamsInputChannel.name() == "twilio_media_streams"


async def test_start_session(
    input_channel: TwilioMediaStreamsInputChannel, call_parameters: CallParameters
):
    websocket = AsyncMock()
    on_new_message = AsyncMock()
    tts_engine = AsyncMock()
    await input_channel.start_session(
        websocket, on_new_message, tts_engine, call_parameters
    )

    on_new_message.assert_called_once()
    call_args = on_new_message.call_args
    user_message = call_args[0][0]
    assert isinstance(user_message, UserMessage)
    assert user_message.text == "/session_start"
    assert user_message.sender_id == call_parameters.stream_id
    assert isinstance(user_message.output_channel, TwilioMediaStreamsOutputChannel)
    assert user_message.input_channel == input_channel.name()
    assert user_message.metadata == asdict(call_parameters)


async def test_collect_call_parameters(input_channel: TwilioMediaStreamsInputChannel):
    bot_phone = "+49123456789"
    user_phone = "+49176124567"
    stream_id = "MZcdce5426d49ccf48c7b0d0ab86a63d52"
    call_id = "CAa874cb4d1ac15290b51b28c91d467812"
    direction = "inbound"
    channel_start_data = create_twilio_media_streams_start_message(
        bot_phone, user_phone, call_id, stream_id, direction
    )
    websocket = AsyncMock()
    websocket.__aiter__.return_value = [channel_start_data]
    call_parameters = await input_channel.collect_call_parameters(websocket)
    assert call_parameters is not None
    assert call_parameters.bot_phone == bot_phone
    assert call_parameters.user_phone == user_phone
    assert call_parameters.stream_id == stream_id
    assert call_parameters.call_id == call_id
    assert call_parameters.direction == direction


async def test_map_media_input_message(
    input_channel: TwilioMediaStreamsInputChannel, audio_data_path: str
):
    media_messages = create_twilio_media_streams_media_messages(
        audio_data_path + "/01.wav", "test_id"
    )
    action = input_channel.map_input_message(media_messages[0])
    assert isinstance(action, NewAudioAction)


async def test_run_audio_streaming(
    input_channel: TwilioMediaStreamsInputChannel, audio_data_path: str
):
    websocket = create_twilio_media_streams_input_stream(audio_data_path)
    on_new_message = AsyncMock()
    await input_channel.run_audio_streaming(on_new_message, websocket)
    assert on_new_message.call_count == 2
