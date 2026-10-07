import logging
from typing import Any, Dict, Text

from twilio.request_validator import RequestValidator

from rasa.core import utils, run
from rasa.core.channels.twilio import TwilioInput

logger = logging.getLogger(__name__)


def test_twilio_channel():
    """Twilio channel test."""
    input_channel = TwilioInput(
        account_sid="ACCOUNT_SID",
        # Find your Account SID and Auth Token at twilio.com/console
        auth_token="AUTH_TOKEN",
        # Phone Number you want to use
        twilio_number="TWILIO_NUMBER",
    )
    s = run.configure_app([input_channel], port=5011)
    routes_list = utils.list_routes(s)

    assert routes_list["twilio_webhook.health"].startswith("/webhooks/twilio")
    assert routes_list["twilio_webhook.message"].startswith("/webhooks/twilio/webhook")


class _FakeRequest:
    def __init__(
        self, headers: Dict[Text, Text], form: Dict[Text, Any], url: Text
    ) -> None:
        self.headers = headers
        self.form = form
        self.url = url


def test_twilio_verify_signature():
    channel = TwilioInput(
        account_sid="ACCOUNT_SID", auth_token="AUTH_TOKEN", twilio_number="NUMBER"
    )
    url = "https://example.com/webhooks/twilio/webhook"
    body = {"From": "+19995551234", "Body": "hi"}
    signature = RequestValidator("AUTH_TOKEN").compute_signature(url, body)

    request = _FakeRequest(
        headers={"X-Original-Url": url, "X-Twilio-Signature": signature},
        form={k: [v] for k, v in body.items()},
        url=url,
    )
    assert channel.is_request_from_twilio_authentic(request) is True


def test_twilio_rejects_forged_signature():
    channel = TwilioInput(
        account_sid="ACCOUNT_SID", auth_token="AUTH_TOKEN", twilio_number="NUMBER"
    )
    url = "https://example.com/webhooks/twilio/webhook"
    body = {"From": "attacker", "Body": "forged"}

    request = _FakeRequest(
        headers={"X-Original-Url": url, "X-Twilio-Signature": "forged=="},
        form={k: [v] for k, v in body.items()},
        url=url,
    )
    assert channel.is_request_from_twilio_authentic(request) is False


def test_twilio_rejects_missing_signature():
    channel = TwilioInput(
        account_sid="ACCOUNT_SID", auth_token="AUTH_TOKEN", twilio_number="NUMBER"
    )
    url = "https://example.com/webhooks/twilio/webhook"
    request = _FakeRequest(
        headers={}, form={"From": ["attacker"]}, url=url
    )
    assert channel.is_request_from_twilio_authentic(request) is False


def test_twilio_rejects_when_auth_token_not_configured():
    channel = TwilioInput(
        account_sid="ACCOUNT_SID", auth_token=None, twilio_number="NUMBER"
    )
    url = "https://example.com/webhooks/twilio/webhook"
    request = _FakeRequest(
        headers={"X-Twilio-Signature": "anything"}, form={}, url=url
    )
    assert channel.is_request_from_twilio_authentic(request) is False
