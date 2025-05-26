import asyncio
import base64
import json
import logging
from dataclasses import asdict
from http import HTTPStatus
from typing import Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from sanic import Request
from twilio.twiml.voice_response import VoiceResponse

from rasa import server
from rasa.core.agent import Agent
from rasa.core.channels import TwilioMediaStreamsInputChannel, UserMessage, channel
from rasa.core.channels.channel import BASIC_AUTH_SCHEME
from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.twilio_media_streams import (
    CALL_SID_REQUEST_KEY,
    DIRECTION_REQUEST_KEY,
    FROM_NUMBER_REQUEST_KEY,
    TO_NUMBER_REQUEST_KEY,
    TWILIO_MEDIA_STREAMS_WEBHOOK_PATH,
    TWILIO_MEDIA_STREAMS_WEBSOCKET_PATH,
    TwilioMediaStreamsOutputChannel,
)
from rasa.core.channels.voice_stream.util import read_wav_to_rasa_audio_bytes
from rasa.core.channels.voice_stream.voice_channel import (
    NewAudioAction,
    tts_engine_from_config,
)
from rasa.shared.exceptions import RasaException


@pytest.fixture
def server_url() -> str:
    return "example.com"


@pytest.fixture
def input_channel(server_url: str) -> TwilioMediaStreamsInputChannel:
    asr_config = {"name": "deepgram"}
    tts_config = {"name": "azure"}
    input_channel = TwilioMediaStreamsInputChannel(
        f"https://{server_url}", asr_config, tts_config
    )
    return input_channel


@pytest.fixture
def call_parameters() -> CallParameters:
    return CallParameters(
        "test_id", "test_phone", "test_bot_phone", stream_id="test_stream_id"
    )


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "server_url": "https://example.com",
                "asr": {"name": "deepgram"},
                "tts": {"name": "azure"},
            },
            {
                "server_url": "https://example.com",
                "asr": {"name": "deepgram"},
                "tts": {"name": "azure"},
                "monitor_silence": False,
                "username": None,
                "password": None,
            },
        ),
        (
            {
                "server_url": "https://example.com",
                "asr": {"name": "deepgram"},
                "tts": {"name": "azure"},
                "monitor_silence": True,
            },
            {
                "server_url": "https://example.com",
                "asr": {"name": "deepgram"},
                "tts": {"name": "azure"},
                "monitor_silence": True,
                "username": None,
                "password": None,
            },
        ),
        (
            {
                "server_url": "https://example.com",
                "asr": {"name": "deepgram"},
                "tts": {"name": "azure"},
                "monitor_silence": True,
                "username": "test_user",
                "password": "test_password",
            },
            {
                "server_url": "https://example.com",
                "asr": {"name": "deepgram"},
                "tts": {"name": "azure"},
                "monitor_silence": True,
                "username": "test_user",
                "password": "test_password",
            },
        ),
    ],
)
def test_twilio_voice_valid_credentials(
    config: Dict[str, str], expected: Dict[str, str]
):
    """Test creation of TwilioMediaStreamsInputChannel with valid credentials."""
    input_channel = TwilioMediaStreamsInputChannel.from_credentials(config)
    assert isinstance(input_channel, TwilioMediaStreamsInputChannel)

    assert input_channel.server_url == expected["server_url"]
    assert input_channel.asr_config == expected["asr"]
    assert input_channel.tts_config == expected["tts"]
    assert input_channel.monitor_silence == expected["monitor_silence"]
    assert input_channel.username == expected["username"]
    assert input_channel.password == expected["password"]


@pytest.mark.parametrize(
    "config",
    [
        {
            "server_url": f"https://{server_url}",
            "asr": {"name": "deepgram"},
            "tts": {"name": "azure"},
            "username": "test_user",
        },
        {
            "server_url": f"https://{server_url}",
            "asr": {"name": "deepgram"},
            "tts": {"name": "azure"},
            "password": "test_password",
        },
    ],
)
def test_twilio_voice_input_invalid_credentials(
    config: Dict[str, str],
):
    """Test creation of TwilioMediaStreamsInputChannel with invalid credentials."""
    with pytest.raises(RasaException):
        TwilioMediaStreamsInputChannel.from_credentials(config)


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


async def test_fitting_output_channel_creation(
    input_channel: TwilioMediaStreamsInputChannel,
):
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
    websocket = AsyncMock()
    media_messages = create_twilio_media_streams_media_messages(
        audio_data_path + "/01.wav", "test_id"
    )
    action = input_channel.map_input_message(media_messages[0], websocket)
    assert isinstance(action, NewAudioAction)


async def test_run_audio_streaming(
    input_channel: TwilioMediaStreamsInputChannel, audio_data_path: str
):
    websocket = create_twilio_media_streams_input_stream(audio_data_path)
    on_new_message = AsyncMock()
    await input_channel.run_audio_streaming(on_new_message, websocket)
    # Should be called thrice with,
    # - /session_start
    # - transcribed audio
    # - /session_end
    assert on_new_message.call_count == 3


USERNAME = 0
PASSWORD = 1


@pytest.fixture
def twilio_username_password() -> Tuple[str, str]:
    """Fixture to create a Twilio username and password."""
    return "test_user", "test_password"


@pytest.fixture
def twilio_media_streams_input_with_auth(
    server_url: str,
    twilio_username_password: Tuple[str, str],
) -> TwilioMediaStreamsInputChannel:
    """Fixture to create a TwilioMediaStreamsInputChannel with authentication."""
    inputs = {
        "server_url": f"https://{server_url}",
        "asr_config": {"name": "deepgram"},
        "tts_config": {"name": "azure"},
        "username": twilio_username_password[USERNAME],
        "password": twilio_username_password[PASSWORD],
    }

    return TwilioMediaStreamsInputChannel(**inputs)


@pytest.fixture
def twilio_body_request() -> Dict[str, str]:
    """Fixture to create a Twilio body request."""
    return {
        CALL_SID_REQUEST_KEY: "CAa874cb4d1ac15290b51b28c91d467812",
        FROM_NUMBER_REQUEST_KEY: "+49123456789",
        TO_NUMBER_REQUEST_KEY: "+49176124567",
        DIRECTION_REQUEST_KEY: "outbound",
    }


async def test_twilio_media_streams_authentication_without_authorization_header(
    twilio_media_streams_input_with_auth: TwilioMediaStreamsInputChannel,
    twilio_body_request: Dict[str, str],
    stack_agent: Agent,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that authentication is required for the webhook endpoint."""
    app = server.create_app(agent=stack_agent)

    channel.register([twilio_media_streams_input_with_auth], app, "/webhooks/")
    client = app.asgi_client

    with caplog.at_level(logging.DEBUG):
        _, response = await client.post(
            f"/{TWILIO_MEDIA_STREAMS_WEBHOOK_PATH}",
            headers={"Content-type": "application/x-www-form-urlencoded"},
            data=twilio_body_request,
        )

    exception_headers = {
        "WWW-Authenticate": f"{BASIC_AUTH_SCHEME} "
        f'realm="{TWILIO_MEDIA_STREAMS_WEBHOOK_PATH}"'
    }
    assert (
        f"Responding with {HTTPStatus.UNAUTHORIZED} and "
        f"necessary auth headers {exception_headers}"
    ) in caplog.text

    assert response.status == HTTPStatus.UNAUTHORIZED
    assert (
        response.body == b"\xe2\x9a\xa0\xef\xb8\x8f 401 \xe2\x80\x94 "
        b"Unauthorized\n=====================\nAuthentication requested.\n\n"
    )
    assert (
        response.headers["WWW-Authenticate"]
        == f'{BASIC_AUTH_SCHEME} realm="{TWILIO_MEDIA_STREAMS_WEBHOOK_PATH}"'
    )


async def test_twilio_media_streams_authentication_with_invalid_authorization_header(
    server_url: str,
    twilio_username_password: Tuple[str, str],
    twilio_media_streams_input_with_auth: TwilioMediaStreamsInputChannel,
    twilio_body_request: Dict[str, str],
    stack_agent: Agent,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test Twilio authentication with invalid authorization header."""
    app = server.create_app(agent=stack_agent)

    channel.register([twilio_media_streams_input_with_auth], app, "/webhooks/")
    client = app.asgi_client

    username = twilio_username_password[USERNAME]
    password = twilio_username_password[PASSWORD]

    encoded_credentials = base64.b64encode(f"{username}:{password}".encode()).decode()

    with caplog.at_level(logging.DEBUG):
        _, response = await client.post(
            f"{TWILIO_MEDIA_STREAMS_WEBHOOK_PATH}",
            headers={
                "Content-type": "application/x-www-form-urlencoded",
                "Authorization": f"Bearer {encoded_credentials}",
            },
            data=twilio_body_request,
        )

    assert "Missing or invalid authorization header." in caplog.text

    assert response.status == HTTPStatus.UNAUTHORIZED
    assert (
        response.body == b"\xe2\x9a\xa0\xef\xb8\x8f 401 \xe2\x80\x94 "
        b"Unauthorized\n=====================\n"
        b"Missing or invalid authorization header.\n\n"
    )


async def test_twilio_media_streams_authentication_with_invalid_credentials(
    server_url: str,
    twilio_username_password: Tuple[str, str],
    twilio_media_streams_input_with_auth: TwilioMediaStreamsInputChannel,
    twilio_body_request: Dict[str, str],
    stack_agent: Agent,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test Twilio authentication with invalid credentials."""
    app = server.create_app(agent=stack_agent)

    channel.register([twilio_media_streams_input_with_auth], app, "/webhooks/")
    client = app.asgi_client

    username = twilio_username_password[USERNAME]
    password = twilio_username_password[PASSWORD]

    encoded_credentials = base64.b64encode(
        f"{username}'a':{password}'b'".encode()
    ).decode()

    with caplog.at_level(logging.DEBUG):
        _, response = await client.post(
            f"{TWILIO_MEDIA_STREAMS_WEBHOOK_PATH}",
            headers={
                "Content-type": "application/x-www-form-urlencoded",
                "Authorization": f"{BASIC_AUTH_SCHEME} {encoded_credentials}",
            },
            data=twilio_body_request,
        )

    assert "Invalid username or password." in caplog.text

    assert response.status == HTTPStatus.UNAUTHORIZED
    assert (
        response.body == b"\xe2\x9a\xa0\xef\xb8\x8f 401 \xe2\x80\x94 "
        b"Unauthorized\n=====================\nInvalid username or password.\n\n"
    )


async def test_twilio_media_streams_authentication_with_authorization_header(
    server_url: str,
    twilio_username_password: Tuple[str, str],
    twilio_media_streams_input_with_auth: TwilioMediaStreamsInputChannel,
    twilio_body_request: Dict[str, str],
    stack_agent: Agent,
) -> None:
    """Test Twilio authentication for the webhook endpoint."""
    app = server.create_app(agent=stack_agent)

    channel.register([twilio_media_streams_input_with_auth], app, "/webhooks/")
    client = app.asgi_client

    username = twilio_username_password[USERNAME]
    password = twilio_username_password[PASSWORD]

    encoded_credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
    _, response = await client.post(
        f"{TWILIO_MEDIA_STREAMS_WEBHOOK_PATH}",
        headers={
            "Content-type": "application/x-www-form-urlencoded",
            "Authorization": f"{BASIC_AUTH_SCHEME} {encoded_credentials}",
        },
        data=twilio_body_request,
    )

    assert response.status == HTTPStatus.OK
    assert response.headers["Content-Type"] == "text/xml"

    call_id = twilio_body_request[CALL_SID_REQUEST_KEY]
    from_number = twilio_body_request[FROM_NUMBER_REQUEST_KEY]
    to_number = twilio_body_request[TO_NUMBER_REQUEST_KEY]
    direction = twilio_body_request[DIRECTION_REQUEST_KEY]

    expected_response_body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="wss://{server_url}/{TWILIO_MEDIA_STREAMS_WEBSOCKET_PATH}">'
        f'<Parameter name="call_id" value="{call_id}" />'
        f'<Parameter name="user_phone" value="{from_number}" />'
        f'<Parameter name="bot_phone" value="{to_number}" />'
        f'<Parameter name="direction" value="{direction}" />'
        "</Stream></Connect></Response>"
    )
    assert response.body == expected_response_body.encode()


@pytest.mark.parametrize(
    "request_form, expected",
    [
        (
            {
                CALL_SID_REQUEST_KEY: "CAa874cb4d1ac15290b51b28c91d467812",
                FROM_NUMBER_REQUEST_KEY: "+49123456789",
                TO_NUMBER_REQUEST_KEY: "+49176124567",
                DIRECTION_REQUEST_KEY: "outbound",
            },
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Connect>"
            '<Stream url="wss://example.com/webhooks/twilio_media_streams/websocket">'
            '<Parameter name="call_id" value="CAa874cb4d1ac15290b51b28c91d467812" />'
            '<Parameter name="user_phone" value="+49123456789" />'
            '<Parameter name="bot_phone" value="+49176124567" />'
            '<Parameter name="direction" value="outbound" />'
            "</Stream></Connect></Response>",
        ),
        (
            {
                CALL_SID_REQUEST_KEY: "CAa874cb4d1ac15290b51b28c91d467812",
                FROM_NUMBER_REQUEST_KEY: "+49123456789",
                TO_NUMBER_REQUEST_KEY: "+49176124567",
            },
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Connect>"
            '<Stream url="wss://example.com/webhooks/twilio_media_streams/websocket">'
            '<Parameter name="call_id" value="CAa874cb4d1ac15290b51b28c91d467812" />'
            '<Parameter name="user_phone" value="+49123456789" />'
            '<Parameter name="bot_phone" value="+49176124567" />'
            '<Parameter name="direction" />'
            "</Stream></Connect></Response>",
        ),
        (
            {
                CALL_SID_REQUEST_KEY: "CAa874cb4d1ac15290b51b28c91d467812",
                FROM_NUMBER_REQUEST_KEY: "+49123456789",
            },
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Connect>"
            '<Stream url="wss://example.com/webhooks/twilio_media_streams/websocket">'
            '<Parameter name="call_id" value="CAa874cb4d1ac15290b51b28c91d467812" />'
            '<Parameter name="user_phone" value="+49123456789" />'
            '<Parameter name="bot_phone" />'
            '<Parameter name="direction" />'
            "</Stream></Connect></Response>",
        ),
        (
            {
                CALL_SID_REQUEST_KEY: "CAa874cb4d1ac15290b51b28c91d467812",
            },
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Connect>"
            '<Stream url="wss://example.com/webhooks/twilio_media_streams/websocket">'
            '<Parameter name="call_id" value="CAa874cb4d1ac15290b51b28c91d467812" />'
            '<Parameter name="user_phone" />'
            '<Parameter name="bot_phone" />'
            '<Parameter name="direction" />'
            "</Stream></Connect></Response>",
        ),
        (
            {},
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Connect>"
            '<Stream url="wss://example.com/webhooks/twilio_media_streams/websocket">'
            '<Parameter name="call_id" />'
            '<Parameter name="user_phone" />'
            '<Parameter name="bot_phone" />'
            '<Parameter name="direction" />'
            "</Stream></Connect></Response>",
        ),
    ],
)
def test_twilio_media_streams_build_twilio_response(
    request_form: Dict[str, str],
    expected: str,
    input_channel: TwilioMediaStreamsInputChannel,
) -> None:
    request_mock = MagicMock(spec=Request)
    request_mock.form = request_form

    result = input_channel._build_twilio_response(request_mock)

    assert isinstance(result, VoiceResponse)
    assert str(result) == expected
