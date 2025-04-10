import json
from unittest.mock import AsyncMock, patch

import pytest
from sanic.request import Request

from rasa.core.channels.voice_stream.genesys import HEADER_API_KEY, GenesysInputChannel


@pytest.fixture
def input_channel() -> GenesysInputChannel:
    with patch(
        "rasa.core.channels.voice_stream.voice_channel.validate_voice_license_scope"
    ):
        server_url = "pro-grouse-possibly.ngrok-free.app"
        asr_config = {"name": "azure"}
        tts_config = {"name": "azure"}
        api_key = "SGVsbG8sIEkgYW0gdGhlIEFQSSBrZXkh"
        client_secret = "TXlTdXBlclNlY3JldEtleVRlbGxOby0xITJAMyM0JDU="
        return GenesysInputChannel(
            api_key, client_secret, server_url, asr_config, tts_config
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


async def test_websocket_connection(input_channel, valid_headers):
    message_handler = AsyncMock()
    blueprint = input_channel.blueprint(message_handler)

    request = Request.fake("/webhook", headers=valid_headers)
    ws = AsyncMock()

    handler = next(
        route.handler for route in blueprint.routes if route.uri == "/webhook"
    )

    await handler(request, ws)
    ws.close.assert_not_called()


async def test_missing_headers(input_channel):
    message_handler = AsyncMock()
    blueprint = input_channel.blueprint(message_handler)

    request = Request.fake("/webhook", headers={})
    ws = AsyncMock()

    handler = next(
        route.handler for route in blueprint.routes if route.uri == "/webhook"
    )

    await handler(request, ws)
    ws.close.assert_called_once()
    assert "Missing required header" in ws.close.call_args[0][1]


async def test_open_event(input_channel, valid_headers, open_message):
    message_handler = AsyncMock()
    blueprint = input_channel.blueprint(message_handler)

    request = Request.fake("/webhook", headers=valid_headers)
    ws = AsyncMock()
    ws.__aiter__.return_value = [json.dumps(open_message)]

    handler = next(
        route.handler for route in blueprint.routes if route.uri == "/webhook"
    )

    await handler(request, ws)

    ws.send.assert_called_once()
    sent_message = json.loads(ws.send.call_args[0][0])
    assert sent_message["type"] == "opened"
    assert sent_message["parameters"]["media"] == open_message["parameters"]["media"]


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
