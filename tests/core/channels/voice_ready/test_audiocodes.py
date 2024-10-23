from typing import Any

import pytest
from rasa.core import run, utils
from rasa.shared.exceptions import RasaException
from _pytest.capture import CaptureFixture

from rasa.core.channels.voice_ready.audiocodes import (
    AudiocodesInput,
    AudiocodesOutput,
    Conversation,
)


@pytest.mark.parametrize(
    "credentials",
    [
        ({"token": "abc", "keep_alive": "123"}),
        ({"token": 123}),
        ({"token": "123", "use_websocket": "true"}),
        ({"use_websocket": True}),
        ({"token": "123", "keep_alive_expiration_factor": 0.5}),
    ],
)
def test_from_credentials_invalid_format(credentials: Any) -> None:
    with pytest.raises(RasaException):
        AudiocodesInput.from_credentials(credentials)


@pytest.mark.parametrize(
    "credentials",
    [
        ({}),
    ],
)
def test_from_credentials_empty(credentials: Any) -> None:
    with pytest.raises(RasaException):
        AudiocodesInput.from_credentials(credentials)


@pytest.mark.parametrize(
    "credentials",
    [
        ({"token": "abc", "use_websocket": False}),
        ({"token": "abc", "keep_alive": 123}),
    ],
)
def test_from_credentials(credentials: Any) -> None:
    input_channel = AudiocodesInput.from_credentials(credentials)
    assert input_channel is not None
    assert isinstance(input_channel, AudiocodesInput)


async def test_attachment_messages_raise_exceptions() -> None:
    with pytest.raises(RasaException):
        output_channel = AudiocodesOutput()
        await output_channel.send_attachment(recipient_id="123", attachment="xxx")


async def test_image_messages_raise_exceptions() -> None:
    with pytest.raises(RasaException):
        output_channel = AudiocodesOutput()
        await output_channel.send_image_url(recipient_id="123", image="xxx")


def test_audiocodes_input_channel() -> None:
    input_channel = AudiocodesInput(
        token="TOKEN",
        use_websocket=True,
        keep_alive=120,
        keep_alive_expiration_factor=1.5,
    )

    s = run.configure_app([input_channel], port=5004)
    routes_list = utils.list_routes(s)
    print(routes_list)
    assert routes_list["ac_webhook.health"].startswith("/webhooks/audiocodes")
    assert routes_list["ac_webhook.receive"].startswith("/webhooks/audiocodes/webhook")
    assert routes_list["ac_webhook.on_activities"].startswith(
        "/webhooks/audiocodes/conversation/<conversation_id:str>/activities"
    )
    assert routes_list["ac_webhook.disconnect"].startswith(
        "/webhooks/audiocodes/conversation/<conversation_id:str>/disconnect"
    )
    assert routes_list["ac_webhook.keepalive"].startswith(
        "/webhooks/audiocodes/conversation/<conversation_id:str>/keepalive"
    )


async def test_send_text_message() -> None:
    output_channel = AudiocodesOutput()

    await output_channel.send_text_message(recipient_id="123", text="hey")
    assert len(output_channel.messages) == 1
    message = output_channel.messages[0]
    assert "id" in message
    assert "timestamp" in message
    assert "type" in message and message.get("type") == "message"
    assert "text" in message and message.get("text") == "hey"


async def test_send_custom_json_message() -> None:
    from rasa.core.channels.voice_ready.audiocodes import AudiocodesOutput

    output_channel = AudiocodesOutput()

    await output_channel.send_custom_json(
        recipient_id="123", json_message={"key": "val"}
    )
    assert len(output_channel.messages) == 1
    message = output_channel.messages[0]
    assert "id" in message
    assert "timestamp" in message
    assert "key" in message and message.get("key") == "val"


async def test_conversation_handle_event_start() -> None:
    conversation = Conversation(conversation_id="123")
    start_event = {
        "id": "0ec183eb-adac-4f42-9867-4f3e49327c5f",
        "timestamp": "2024-08-06T14:36:33.468Z",
        "language": "en-US",
        "type": "event",
        "name": "start",
        "parameters": {
            "callee": "+493040739365",
            "calleeHost": "20.113.51.15",
            "caller": "+491604697810",
            "callerHost": "sip.telnyx.eu",
            "callerDisplayName": "+491604697810",
            "vaigConversationId": "f1bc9d81-7d2d-40c3-a538-0672c7b80339",
        },
    }
    text = conversation._handle_event(start_event)
    assert (
        text
        == '/session_start{"call_id": "f1bc9d81-7d2d-40c3-a538-0672c7b80339", "user_phone": "+493040739365", "bot_phone": "+491604697810", "user_name": "+491604697810", "user_host": "sip.telnyx.eu", "bot_host": "20.113.51.15", "direction": null, "stream_id": null}'  # noqa: E501
    )


async def test_conversation_handle_event_invalid_payload(
    capsys: CaptureFixture,
) -> None:
    conversation = Conversation(conversation_id="123")
    event_payload = {}

    text = conversation._handle_event(event_payload)

    # assert that warning was raised and text is ""
    assert text == ""
    captured = capsys.readouterr()
    assert "audiocodes.handle.event.no_name_key" in captured.out


async def test_conversation_handle_event_invalid_name(capsys: CaptureFixture) -> None:
    conversation = Conversation(conversation_id="123")
    event_payload = {"name": "invalid"}

    text = conversation._handle_event(event_payload)

    # assert that warning was raised and text is ""
    assert text == ""
    captured = capsys.readouterr()
    assert "audiocodes.handle.event.unknown_event" in captured.out
