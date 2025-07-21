from typing import Dict
from unittest.mock import AsyncMock

import pytest

from rasa.core.channels.voice_stream.genesys import (
    HEADER_API_KEY,
    GenesysInputChannel,
    map_call_params,
)
from rasa.shared.exceptions import RasaException


@pytest.fixture
def input_channel() -> GenesysInputChannel:
    server_url = "pro-grouse-possibly.ngrok-free.app"
    asr_config = {"name": "azure"}
    tts_config = {"name": "azure"}
    api_key = "SGVsbG8sIEkgYW0gdGhlIEFQSSBrZXkh"
    client_secret = "TXlTdXBlclNlY3JldEtleVRlbGxOby0xITJAMyM0JDU="
    return GenesysInputChannel(
        api_key=api_key,
        client_secret=client_secret,
        server_url=server_url,
        asr_config=asr_config,
        tts_config=tts_config,
    )


@pytest.fixture
def valid_headers():
    return {
        "Audiohook-Organization-Id": "22352111-6076-492a-8163-514a00723975",
        "Audiohook-Correlation-Id": "386dc26f-6d1d-4cf0-b153-93c43f540874",
        "Audiohook-Session-Id": "386dc26f-6d1d-4cf0-b153-93c43f540874",
        "X-Api-Key": "SGVsbG8sIEkgYW0gdGhlIEFQSSBrZXkh",
        "Signature": "sig1=:Y6jeGihkrFCi3XwNcwQzw1TeDurpB4bZsMAT6KUiGj4=:",
        "Signature-Input": 'sig1=("@request-target" "audiohook-session-id" "audiohook-organization-id" "audiohook-correlation-id" "x-api-key" "@authority");created=1744210396;expires=1744210406;keyid="SGVsbG8sIEkgYW0gdGhlIEFQSSBrZXkh";nonce="9qAwwlMvkHqzxf0DoOOXVvNe";alg="hmac-sha256"',  # noqa: E501
    }


@pytest.fixture
def mocked_request(valid_headers):
    return AsyncMock(
        headers=valid_headers,
    )


@pytest.fixture
def open_message():
    return {
        "version": "2",
        "id": "3ccd9712-cdbe-44f7-bcc1-486e2a1a8ff6",
        "type": "open",
        "seq": 1,
        "position": "PT0.0S",
        "parameters": {
            "organizationId": "22352111-6076-492a-8163-514a00723975",
            "conversationId": "28faf323-fd6e-4bc8-b859-fb25b133d16d",
            "participant": {
                "id": "28faf323-fd6e-4bc8-b859-fb25b133d16d",
                "ani": "tel:+491604697810",
                "aniName": "",
                "dnis": "+493070016507",
            },
            "media": [
                {
                    "type": "audio",
                    "format": "PCMU",
                    "channels": ["external"],
                    "rate": 8000,
                }
            ],
            "language": "en-us",
            "inputVariables": {},
        },
        "serverseq": 0,
    }


async def test_call_params(open_message):
    call_parameters = map_call_params(open_message)
    assert call_parameters is not None
    assert call_parameters.bot_phone == "+493070016507"
    assert call_parameters.user_phone == "+491604697810"
    assert call_parameters.stream_id is None
    assert call_parameters.call_id == "28faf323-fd6e-4bc8-b859-fb25b133d16d"
    assert call_parameters.direction is None


def test_ensure_api_key(input_channel, mocked_request):
    assert input_channel._ensure_api_key(mocked_request) is True

    mocked_request.headers[HEADER_API_KEY] = "invalid-key"
    assert input_channel._ensure_api_key(mocked_request) is False


def test_ensure_required_headers(input_channel, mocked_request):
    assert input_channel._ensure_required_headers(mocked_request) is True

    del mocked_request.headers["Audiohook-Organization-Id"]
    assert input_channel._ensure_required_headers(mocked_request) is False

    del mocked_request.headers["Audiohook-Correlation-Id"]
    assert input_channel._ensure_required_headers(mocked_request) is False

    del mocked_request.headers["Audiohook-Session-Id"]
    assert input_channel._ensure_required_headers(mocked_request) is False


async def test_verify_signature(input_channel, mocked_request):
    assert await input_channel._verify_signature(mocked_request) is True

    # modify the header and verification should fail
    mocked_request.headers["Audiohook-Organization-Id"] = "random-value"
    assert await input_channel._verify_signature(mocked_request) is False


@pytest.mark.parametrize(
    "input_data",
    [
        {
            "api_key": "test_token",
            "client_secret": "some_secret",
            "server_url": "https://example.com",
            "asr": {"name": "deepgram"},
            "tts": {"name": "azure"},
        },
        {
            "api_key": "test_token",
            "client_secret": None,
            "server_url": "https://example.com",
            "asr": {"name": "deepgram"},
            "tts": {"name": "azure"},
        },
    ],
)
@pytest.mark.usefixtures("mock_validate_voice_license_scope")
def test_from_credentials(input_data: dict, mock_validate_voice_license_scope):
    """Tests the from_credentials method."""
    channel = GenesysInputChannel.from_credentials(
        input_data,
    )

    assert isinstance(channel, GenesysInputChannel)
    assert channel.api_key == input_data["api_key"]
    assert channel.client_secret == input_data["client_secret"]
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
            "asr": {"name": "azure"},
            # Missing API key
        },
        {
            "server_url": "https://example.com",
            "tts": {"name": "azure"},
            "asr": {"name": "azure"},
            "client_secret": "some_secret",
            # Missing API key
        },
    ],
)
def test_invalid_credentials(
    config: Dict[str, str],
):
    """Test creation of GenesysInputChannel with invalid credentials."""
    with pytest.raises(RasaException):
        GenesysInputChannel.from_credentials(config)
