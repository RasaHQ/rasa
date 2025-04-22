import base64
import logging
from http import HTTPStatus
from typing import Any, Dict, Text, Tuple, Type

import pytest
import structlog.testing

from rasa import server
from rasa.core.agent import Agent
from rasa.core.channels import channel
from rasa.core.channels.channel import BASIC_AUTH_SCHEME
from rasa.core.channels.voice_ready.twilio_voice import (
    TWILIO_VOICE_PATH,
    TwilioVoiceCollectingOutputChannel,
    TwilioVoiceInput,
)
from rasa.shared.exceptions import InvalidConfigException, RasaException
from tests.utilities import filter_logs

logger = logging.getLogger(__name__)


async def test_twilio_voice_twiml_response_text():
    inputs = {
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    twilio_voice_input = TwilioVoiceInput(**inputs)

    output_channel = TwilioVoiceCollectingOutputChannel()

    await output_channel.send_text_message(recipient_id="Chuck Norris", text="Test:")
    assert len(output_channel.messages) == 1
    assert output_channel.messages[0]["text"] == "Test:"

    twiml = twilio_voice_input._build_twilio_voice_response(output_channel.messages)
    assert (
        str(twiml) == '<?xml version="1.0" encoding="UTF-8"?><Response>'
        '<Gather action="/webhooks/twilio_voice/webhook" '
        'actionOnEmptyResult="true" enhanced="false" input="speech" '
        'speechModel="default" speechTimeout="5"><Say voice="woman">'
        "Test:</Say></Gather></Response>"
    )


async def test_twilio_voice_twiml_response_buttons():
    inputs = {
        "reprompt_fallback_phrase": "i didn't get that",
        "speech_model": "default",
        "speech_timeout": "5",
        "assistant_voice": "woman",
        "enhanced": "false",
    }

    twilio_voice_input = TwilioVoiceInput(**inputs)

    output_channel = TwilioVoiceCollectingOutputChannel()
    await output_channel.send_text_with_buttons(
        recipient_id="Chuck Norris",
        text="Buttons:",
        buttons=[
            {"title": "Yes", "payload": "/affirm"},
            {"title": "No", "payload": "/deny"},
        ],
    )
    assert len(output_channel.messages) == 1
    message_str = " ".join([m["text"] for m in output_channel.messages])
    assert message_str == "Buttons: Yes, No"

    twiml = twilio_voice_input._build_twilio_voice_response(output_channel.messages)
    assert (
        str(twiml) == '<?xml version="1.0" encoding="UTF-8"?><Response>'
        '<Gather action="/webhooks/twilio_voice/webhook" '
        'actionOnEmptyResult="true" enhanced="false" input="speech" '
        'speechModel="default" speechTimeout="5">'
        '<Say voice="woman">Buttons: Yes, No</Say>'
        "</Gather></Response>"
    )


@pytest.mark.parametrize(
    "configs, expected",
    [
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "speech_model": "default",
                "speech_timeout": "5",
                "assistant_voice": "alien",
                "enhanced": "false",
            },
            InvalidConfigException,
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "speech_model": "default",
                "speech_timeout": "not a number",
                "assistant_voice": "woman",
                "enhanced": "false",
            },
            InvalidConfigException,
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "speech_model": "default",
                "speech_timeout": "auto",
                "assistant_voice": "woman",
                "enhanced": "wrong",
            },
            InvalidConfigException,
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "speech_model": "default",
                "speech_timeout": "5",
                "assistant_voice": "woman",
                "enhanced": "true",
            },
            InvalidConfigException,
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "enhanced": "true",
                "speech_model": "default",
                "speech_timeout": "auto",
            },
            InvalidConfigException,
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "enhanced": "true",
                "speech_model": "phone_call",
                "speech_timeout": "auto",
            },
            InvalidConfigException,
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "5",
                "speech_model": "default",
                "enhanced": "false",
                "password": "test_password",
            },
            InvalidConfigException,
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "5",
                "speech_model": "default",
                "enhanced": "false",
                "username": "test_user",
            },
            InvalidConfigException,
        ),
    ],
)
def test_invalid_configs(configs: Dict[Text, Any], expected: Type[RasaException]):
    with pytest.raises(expected):
        TwilioVoiceInput(**configs)


@pytest.mark.parametrize(
    "configs, expected",
    [
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
            },
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "5",
                "speech_model": "default",
                "enhanced": "false",
                "username": None,
                "password": None,
            },
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "3",
                "speech_model": "phone_call",
                "enhanced": "true",
            },
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "3",
                "speech_model": "phone_call",
                "enhanced": "true",
                "username": None,
                "password": None,
            },
        ),
        (
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "3",
                "speech_model": "phone_call",
                "enhanced": "true",
                "username": "test_user",
                "password": "test_password",
            },
            {
                "reprompt_fallback_phrase": "i didn't get that",
                "assistant_voice": "woman",
                "speech_timeout": "3",
                "speech_model": "phone_call",
                "enhanced": "true",
                "username": "test_user",
                "password": "test_password",
            },
        ),
    ],
)
def test_twilio_voice_input(configs: Dict[Text, Any], expected: Dict[Text, Any]):
    """Test TwilioVoiceInput initialization with various configurations."""
    twilio_voice_input = TwilioVoiceInput.from_credentials(configs)

    assert isinstance(twilio_voice_input, TwilioVoiceInput)

    assert (
        twilio_voice_input.reprompt_fallback_phrase
        == expected["reprompt_fallback_phrase"]
    )
    assert twilio_voice_input.speech_model == expected["speech_model"]
    assert twilio_voice_input.speech_timeout == expected["speech_timeout"]
    assert twilio_voice_input.assistant_voice == expected["assistant_voice"]
    assert twilio_voice_input.enhanced == expected["enhanced"]
    assert twilio_voice_input.username == expected["username"]
    assert twilio_voice_input.password == expected["password"]


@pytest.mark.parametrize(
    "config",
    [
        {
            "reprompt_fallback_phrase": "i didn't get that",
            "assistant_voice": "woman",
            "speech_timeout": "3",
            "speech_model": "phone_call",
            "enhanced": "true",
            "username": "test_user",
        },
        {
            "reprompt_fallback_phrase": "i didn't get that",
            "assistant_voice": "woman",
            "speech_timeout": "3",
            "speech_model": "phone_call",
            "enhanced": "true",
            "password": "test_password",
        },
    ],
)
def test_twilio_voice_input_invalid_credentials(
    config: Dict[str, str],
):
    with pytest.raises(RasaException):
        TwilioVoiceInput.from_credentials(config)


async def test_twilio_voice_remove_image():
    with pytest.warns(UserWarning):
        output_channel = TwilioVoiceCollectingOutputChannel()
        await output_channel.send_response(
            recipient_id="Chuck Norris",
            message={"image": "https://i.imgur.com/nGF1K8f.jpg", "text": "Some text."},
        )


async def test_twilio_voice_keep_image_text():
    output_channel = TwilioVoiceCollectingOutputChannel()
    await output_channel.send_response(
        recipient_id="Chuck Norris",
        message={"image": "https://i.imgur.com/nGF1K8f.jpg", "text": "Some text."},
    )
    assert len(output_channel.messages) == 1
    assert output_channel.messages[0]["text"] == "Some text."


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


async def test_twilio_voice_multiple_responses(
    twilio_voice_input: TwilioVoiceInput,
):
    output_channel = TwilioVoiceCollectingOutputChannel()

    await output_channel.send_text_message(
        recipient_id="Chuck Norris", text="message 1"
    )
    await output_channel.send_text_message(
        recipient_id="Chuck Norris", text="message 2"
    )
    assert len(output_channel.messages) == 2
    assert output_channel.messages[0]["text"] == "message 1"
    assert output_channel.messages[1]["text"] == "message 2"

    twiml = twilio_voice_input._build_twilio_voice_response(output_channel.messages)

    assert (
        str(twiml) == '<?xml version="1.0" encoding="UTF-8"?><Response>'
        '<Say voice="woman">message 1</Say>'
        '<Pause length="1" />'
        '<Gather action="/webhooks/twilio_voice/webhook" '
        'actionOnEmptyResult="true" '
        'enhanced="false" '
        'input="speech" '
        'speechModel="default" '
        'speechTimeout="5">'
        '<Say voice="woman">message 2</Say>'
        "</Gather></Response>"
    )


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


USERNAME = 0
PASSWORD = 1


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
