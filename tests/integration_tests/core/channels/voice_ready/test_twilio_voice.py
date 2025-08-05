import base64
import logging
from http import HTTPStatus
from typing import Dict, Tuple

import pytest
import structlog.testing

from rasa import server
from rasa.core.agent import Agent
from rasa.core.channels import channel
from rasa.core.channels.channel import BASIC_AUTH_SCHEME
from rasa.core.channels.voice_ready.twilio_voice import (
    TWILIO_VOICE_PATH,
    TwilioVoiceInput,
)
from tests.utilities import filter_logs

logger = logging.getLogger(__name__)

USERNAME = 0
PASSWORD = 1


@pytest.fixture
def twilio_voice_input() -> TwilioVoiceInput:
    inputs = {
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    return TwilioVoiceInput(**inputs)


async def test_twilio_receive_answer(
    twilio_voice_input: TwilioVoiceInput, stack_agent: Agent
):
    app = server.create_app(agent=stack_agent)

    channel.register([twilio_voice_input], app, "/webhooks/")

    client = app.asgi_client

    body = {"From": "Tobias", "CallStatus": "ringing"}
    _, response = await client.post(
        "/webhooks/twilio_voice/webhook",
        headers={"Content-type": "application/x-www-form-urlencoded"},
        data=body,
    )
    assert response.status == HTTPStatus.OK
    # Actual test xml content, response depends on pattern_session_start
    assert (
        response.body == b'<?xml version="1.0" encoding="UTF-8"?><Response>'
        b'<Gather action="/webhooks/twilio_voice/webhook" '
        b'actionOnEmptyResult="true" '
        b'enhanced="false" '
        b'input="speech" '
        b'speechModel="default" '
        b'speechTimeout="5" />'
        b"</Response>"
    )


async def test_twilio_receive_no_response(stack_agent: Agent):
    app = server.create_app(agent=stack_agent)

    inputs = {
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    twilio_voice_input = TwilioVoiceInput(**inputs)
    channel.register([twilio_voice_input], app, "/webhooks/")

    client = app.asgi_client

    body = {"From": "Matthew", "CallStatus": "ringing"}
    _, response = await client.post(
        f"/{TWILIO_VOICE_PATH}",
        headers={"Content-type": "application/x-www-form-urlencoded"},
        data=body,
    )
    assert response.status == HTTPStatus.OK
    assert response.body

    body = {"From": "Matthew", "CallStatus": "answered"}

    with structlog.testing.capture_logs() as log:
        _, response = await client.post(
            "/webhooks/twilio_voice/webhook",
            headers={"Content-type": "application/x-www-form-urlencoded"},
            data=body,
        )

        assert response.status == HTTPStatus.OK
        assert (
            response.body == b'<?xml version="1.0" encoding="UTF-8"?><Response>'
            b'<Gather action="/webhooks/twilio_voice/webhook" '
            b'actionOnEmptyResult="true" '
            b'enhanced="false" '
            b'input="speech" '
            b'speechModel="default" '
            b'speechTimeout="5">'
            b'<Say voice="woman">i didn\'t get that</Say>'
            b"</Gather></Response>"
        )

        logs = filter_logs(log, "twilio_voice.webhook.twilio_response")
        assert len(logs) == 1


async def test_twilio_receive_no_previous_response(stack_agent: Agent):
    app = server.create_app(agent=stack_agent)

    inputs = {
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    twilio_voice_input = TwilioVoiceInput(**inputs)
    channel.register([twilio_voice_input], app, "/webhooks/")

    client = app.asgi_client

    body = {"From": "Ray", "CallStatus": "answered"}

    with structlog.testing.capture_logs() as log:
        _, response = await client.post(
            "/webhooks/twilio_voice/webhook",
            headers={"Content-type": "application/x-www-form-urlencoded"},
            data=body,
        )

        assert response.status == HTTPStatus.OK
        expected_response = (
            '<?xml version="1.0" encoding="UTF-8"?><Response>'
            '<Gather action="/webhooks/twilio_voice/webhook" '
            'actionOnEmptyResult="true" '
            'enhanced="false" '
            'input="speech" '
            'speechModel="default" '
            'speechTimeout="5">'
            '<Say voice="woman">i didn\'t get that</Say></Gather></Response>'
        )
        assert response.body == expected_response.encode()

        logs = filter_logs(log, "twilio_voice.webhook.twilio_response")
        assert len(logs) == 1


@pytest.fixture
def twilio_username_password() -> Tuple[str, str]:
    """Fixture to create a Twilio username and password."""
    return "test_user", "test_password"


@pytest.fixture
def twilio_voice_input_with_auth(
    twilio_username_password: Tuple[str, str],
) -> TwilioVoiceInput:
    """Fixture to create a TwilioVoiceInput with authentication."""
    inputs = {
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
        "username": twilio_username_password[USERNAME],
        "password": twilio_username_password[PASSWORD],
    }
    return TwilioVoiceInput(**inputs)


@pytest.fixture
def twilio_body_request() -> Dict[str, str]:
    """Fixture to create a Twilio body request."""
    return {"From": "Ray", "CallStatus": "ringing"}


async def test_twilio_authentication_without_authorization_header(
    stack_agent: Agent,
    twilio_voice_input_with_auth: TwilioVoiceInput,
    twilio_body_request: Dict[str, str],
) -> None:
    """Test that Twilio Voice authentication is required for the webhook endpoint."""
    app = server.create_app(agent=stack_agent)

    channel.register([twilio_voice_input_with_auth], app, "/webhooks/")
    client = app.asgi_client

    _, response = await client.post(
        f"/{TWILIO_VOICE_PATH}",
        headers={"Content-type": "application/x-www-form-urlencoded"},
        data=twilio_body_request,
    )

    assert response.status == HTTPStatus.UNAUTHORIZED
    assert (
        response.body == b"\xe2\x9a\xa0\xef\xb8\x8f 401 \xe2\x80\x94 "
        b"Unauthorized\n=====================\nAuthentication requested.\n\n"
    )
    assert (
        response.headers["WWW-Authenticate"]
        == f'{BASIC_AUTH_SCHEME} realm="{TWILIO_VOICE_PATH}"'
    )


async def test_twilio_media_streams_authentication_with_invalid_authorization_header(
    twilio_username_password: Tuple[str, str],
    twilio_voice_input_with_auth: TwilioVoiceInput,
    twilio_body_request: Dict[str, str],
    stack_agent: Agent,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test Twilio Voice authentication with invalid authorization header."""
    app = server.create_app(agent=stack_agent)

    channel.register([twilio_voice_input_with_auth], app, "/webhooks/")
    client = app.asgi_client

    username = twilio_username_password[USERNAME]
    password = twilio_username_password[PASSWORD]

    encoded_credentials = base64.b64encode(f"{username}:{password}".encode()).decode()

    with caplog.at_level(logging.DEBUG):
        _, response = await client.post(
            f"{TWILIO_VOICE_PATH}",
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
    twilio_username_password: Tuple[str, str],
    twilio_voice_input_with_auth: TwilioVoiceInput,
    twilio_body_request: Dict[str, str],
    stack_agent: Agent,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test Twilio Voice authentication with invalid credentials."""
    app = server.create_app(agent=stack_agent)

    channel.register([twilio_voice_input_with_auth], app, "/webhooks/")
    client = app.asgi_client

    username = twilio_username_password[USERNAME]
    password = twilio_username_password[PASSWORD]

    encoded_credentials = base64.b64encode(
        f"{username}'a':{password}'b'".encode()
    ).decode()

    with caplog.at_level(logging.DEBUG):
        _, response = await client.post(
            f"{TWILIO_VOICE_PATH}",
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


async def test_twilio_voice_authentication_with_authorization_header(
    twilio_username_password: Tuple[str, str],
    twilio_voice_input_with_auth: TwilioVoiceInput,
    twilio_body_request: Dict[str, str],
    stack_agent: Agent,
) -> None:
    """Test Twilio Voice authentication for the webhook endpoint."""
    app = server.create_app(agent=stack_agent)

    channel.register([twilio_voice_input_with_auth], app, "/webhooks/")
    client = app.asgi_client

    username = twilio_username_password[USERNAME]
    password = twilio_username_password[PASSWORD]

    encoded_credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
    with structlog.testing.capture_logs() as log:
        _, response = await client.post(
            "/webhooks/twilio_voice/webhook",
            headers={
                "Content-type": "application/x-www-form-urlencoded",
                "Authorization": f"{BASIC_AUTH_SCHEME} {encoded_credentials}",
            },
            data=twilio_body_request,
        )

        assert response.status == HTTPStatus.OK
        expected_response = (
            '<?xml version="1.0" encoding="UTF-8"?><Response>'
            '<Gather action="/webhooks/twilio_voice/webhook" '
            'actionOnEmptyResult="true" '
            'enhanced="false" '
            'input="speech" '
            'speechModel="default" '
            'speechTimeout="5" />'
            "</Response>"
        )
        assert response.body == expected_response.encode()
        logs = filter_logs(
            log,
            "twilio_voice.webhook.twilio_response",
        )
        assert len(logs) == 1
