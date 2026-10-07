import logging

from rasa.core.channels.rocketchat import RocketChatInput

logger = logging.getLogger(__name__)


def test_rocketchat_verify_token():
    channel = RocketChatInput(
        user="rasa_bot",
        password="pw",
        server_url="https://rocketchat.example.com",
        webhook_token="outgoing_integration_token",
    )
    output = {
        "token": "outgoing_integration_token",
        "user_name": "real_user",
        "text": "hi",
        "channel_id": "c1",
    }
    assert channel.is_request_from_rocketchat_authentic(output) is True


def test_rocketchat_rejects_forged_token():
    channel = RocketChatInput(
        user="rasa_bot",
        password="pw",
        server_url="https://rocketchat.example.com",
        webhook_token="outgoing_integration_token",
    )
    output = {
        "token": "wrong_token",
        "user_name": "attacker",
        "text": "forged",
        "channel_id": "c1",
    }
    assert channel.is_request_from_rocketchat_authentic(output) is False


def test_rocketchat_rejects_missing_token():
    channel = RocketChatInput(
        user="rasa_bot",
        password="pw",
        server_url="https://rocketchat.example.com",
        webhook_token="outgoing_integration_token",
    )
    output = {"user_name": "attacker", "text": "forged", "channel_id": "c1"}
    assert channel.is_request_from_rocketchat_authentic(output) is False


def test_rocketchat_rejects_when_webhook_token_not_configured():
    channel = RocketChatInput(
        user="rasa_bot", password="pw", server_url="https://rocketchat.example.com"
    )
    output = {"token": "anything", "user_name": "x", "text": "x"}
    assert channel.is_request_from_rocketchat_authentic(output) is False


def test_rocketchat_rejects_non_string_token():
    channel = RocketChatInput(
        user="rasa_bot",
        password="pw",
        server_url="https://rocketchat.example.com",
        webhook_token="outgoing_integration_token",
    )
    output = {"token": 12345, "user_name": "x", "text": "x"}
    assert channel.is_request_from_rocketchat_authentic(output) is False
