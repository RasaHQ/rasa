import logging
from sanic import Blueprint, response

from rasa_core.channels import (
    CollectingOutputChannel,
    UserMessage, RestInput)
from rasa_core.utils import EndpointConfig

logger = logging.getLogger(__name__)


class CallbackOutput(CollectingOutputChannel):

    @classmethod
    def name(cls):
        return "callback"

    def __init__(self, endpoint: EndpointConfig) -> None:

        self.callback_endpoint = endpoint
        super(CallbackOutput, self).__init__()

    async def _persist_message(self, message):
        super(CallbackOutput, self)._persist_message(message)

        r = await self.callback_endpoint.request(
            "post",
            content_type="application/json",
            json=message)

        if not 200 <= r.status_code < 300:
            logger.error("Failed to send output message to callback. "
                         "Status: {} Response: {}"
                         "".format(r.status_code, r.text))


class CallbackInput(RestInput):
    """A custom REST http input channel that responds using a callback server.

    Incoming messages are received through a REST interface. Responses
    are sent asynchronously by calling a configured external REST endpoint."""

    @classmethod
    def name(cls):
        return "callback"

    @classmethod
    def from_credentials(cls, credentials):
        return cls(EndpointConfig.from_dict(credentials))

    def __init__(self, endpoint):
        self.callback_endpoint = endpoint

    def blueprint(self, on_new_message):
        callback_webhook = Blueprint('callback_webhook', __name__)

        @callback_webhook.route("/", methods=['GET'])
        async def health(request):
            return response.json({"status": "ok"})

        @callback_webhook.route("/webhook", methods=['POST'])
        async def webhook(request):
            sender_id = self._extract_sender(request)
            text = self._extract_message(request)

            collector = CallbackOutput(self.callback_endpoint)
            await on_new_message(UserMessage(text, collector, sender_id,
                                             input_channel=self.name()))
            return response.text("success")

        return callback_webhook
