import logging
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
