import json
from unittest.mock import AsyncMock

import pytest
from sanic.request import Request

from rasa.core.channels.voice_stream.genesys import GenesysInputChannel


@pytest.fixture
def input_channel():
    return GenesysInputChannel()


@pytest.fixture
def valid_headers():
    return {
        "audiohook-organization-id": "test-org",
        "audiohook-correlation-id": "test-corr",
        "audiohook-session-id": "test-session",
        "x-api-key": "test-key",
    }


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
