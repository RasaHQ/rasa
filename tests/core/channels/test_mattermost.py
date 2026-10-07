import logging

from rasa.core.channels.mattermost import MattermostInput

logger = logging.getLogger(__name__)


def test_mattermost_verify_token():
    channel = MattermostInput(
        url="https://mattermost.example.com/api/v4",
        token="bot_api_token",
        webhook_url="https://bot.example.com/webhooks/mattermost/webhook",
        webhook_token="outgoing_webhook_token",
    )
    output = {
        "token": "outgoing_webhook_token",
        "trigger_word": "hi",
        "text": "hi bot",
        "user_id": "u1",
        "channel_id": "c1",
    }
    assert channel.is_request_from_mattermost_authentic(output) is True


def test_mattermost_rejects_forged_token():
    channel = MattermostInput(
        url="https://mattermost.example.com/api/v4",
        token="bot_api_token",
        webhook_url="https://bot.example.com/webhooks/mattermost/webhook",
        webhook_token="outgoing_webhook_token",
    )
    output = {
        "token": "wrong_token",
        "trigger_word": "hi",
        "text": "forged",
        "user_id": "attacker",
        "channel_id": "c1",
    }
    assert channel.is_request_from_mattermost_authentic(output) is False


def test_mattermost_rejects_missing_token():
    channel = MattermostInput(
        url="https://mattermost.example.com/api/v4",
        token="bot_api_token",
        webhook_url="https://bot.example.com/webhooks/mattermost/webhook",
        webhook_token="outgoing_webhook_token",
    )
    output = {"trigger_word": "hi", "text": "forged", "user_id": "attacker"}
    assert channel.is_request_from_mattermost_authentic(output) is False


def test_mattermost_rejects_when_webhook_token_not_configured():
    channel = MattermostInput(
        url="https://mattermost.example.com/api/v4",
        token="bot_api_token",
        webhook_url="https://bot.example.com/webhooks/mattermost/webhook",
    )
    output = {"token": "anything", "trigger_word": "hi", "text": "x"}
    assert channel.is_request_from_mattermost_authentic(output) is False


def test_mattermost_rejects_non_string_token():
    channel = MattermostInput(
        url="https://mattermost.example.com/api/v4",
        token="bot_api_token",
        webhook_url="https://bot.example.com/webhooks/mattermost/webhook",
        webhook_token="outgoing_webhook_token",
    )
    output = {"token": 12345, "trigger_word": "hi", "text": "x"}
    assert channel.is_request_from_mattermost_authentic(output) is False
