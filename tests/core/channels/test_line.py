import logging
import pytest
from rasa.core import utils, run
from rasa.core.channels.line import LineConnectorOutput
from fbmessenger import MessengerClient
from linebot import LineBotApi

logger = logging.getLogger(__name__)


def test_line_channel():
    from rasa.core.channels.line import LineConnectorInput

    input_channel = LineConnectorInput(
        app_secret="YOUR_APP_SECRET",  # your app secret
        access_token="YOUR_ACCESS_TOKEN"
        # token for the page you subscribed to
    )

    s = run.configure_app([input_channel], port=5004)
    routes_list = utils.list_routes(s)

    assert routes_list["line_webhook.health"].startswith("/webhooks/line")
    assert routes_list["line_webhook.message"].startswith("/webhooks/line/callback")


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {
                "blocks": [
                    {"type": "title", "text": {"text": "Conversation progress"}},
                    {
                        "type": "progression_bar",
                        "text": {"text": "progression 1", "level": "1"},
                    },
                ]
            },
            "test_id",
        ),
        (
            {
                "blocks": [
                    {"type": "title", "text": {"text": "Conversation progress"}},
                    {
                        "type": "progression_bar",
                        "text": {"text": "progression 1", "level": "1"},
                    },
                ],
                "sender": {"id": "test_json_id"},
            },
            "test_json_id",
        ),
        (
            {
                "blocks": {
                    "type": "progression_bar",
                    "text": {"text": "progression 1", "level": "1"},
                },
                "sender": {"id": "test_json_id_2"},
            },
            "test_json_id_2",
        ),
        (
            [
                {
                    "blocks": {
                        "type": "progression_bar",
                        "text": {"text": "progression 1", "level": "1"},
                    }
                },
                {"sender": {"id": "test_json_id_3"}},
            ],
            "test_json_id_3",
        ),
    ],
)
async def test_send_line_custom_json(test_input, expected):
    class TestableLineClient(LineBotApi):
        def __init__(self, page_access_token, **kwargs):
            self.recipient_id = ""
            super(TestableLineClient, self).__init__(page_access_token, **kwargs)

        def send(
            self,
            payload,
            recipient_id,
            messaging_type="RESPONSE",
            notification_type="REGULAR",
            timeout=None,
            tag=None,
        ):
            self.recipient_id = recipient_id

    line_client = TestableLineClient(page_access_token="test_token")
    line_bot = LineConnectorOutput(line_client)
    await line_bot.send_custom_json(recipient_id="test_id", json_message=test_input)
    assert line_bot.line_client.recipient_id == expected
