import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Text

from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse

from rasa.core.channels.channel import (
    CollectingOutputChannel,
    InputChannel,
    UserMessage,
)
from rasa.core.channels.rest import RestInput
from rasa.utils.endpoints import ClientResponseError, EndpointConfig

logger = logging.getLogger(__name__)


class CallbackOutput(CollectingOutputChannel):
    @classmethod
    def name(cls) -> Text:
        return "callback"

    def __init__(self, endpoint: EndpointConfig) -> None:
        self.callback_endpoint = endpoint
        super().__init__()

    async def _persist_message(self, message: Dict[Text, Any]) -> None:
        await super()._persist_message(message)

        try:
            await self.callback_endpoint.request(
                "post", content_type="application/json", json=message
            )
        except ClientResponseError as e:
            logger.error(
                "Failed to send output message to callback. "
                "Status: {} Response: {}"
                "".format(e.status, e.text)
            )


class CallbackInput(RestInput):
    """A custom REST http input channel that responds using a callback server.

    Incoming messages are received through a REST interface. Responses
    are sent asynchronously by calling a configured external REST endpoint.
    """

    @classmethod
    def name(cls) -> Text:
        return "callback"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        return cls(EndpointConfig.from_dict(credentials))

    def __init__(self, endpoint: EndpointConfig) -> None:
        self.callback_endpoint = endpoint

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        callback_webhook = Blueprint("callback_webhook", __name__)

        @callback_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @callback_webhook.route("/webhook", methods=["POST"])
        async def webhook(request: Request) -> HTTPResponse:
            sender_id = await self._extract_sender(request)
            text = self._extract_message(request)

            collector = self.get_output_channel()
            await on_new_message(
                UserMessage(text, collector, sender_id, input_channel=self.name())
            )
            return response.text("success")

        return callback_webhook

    def get_output_channel(self) -> CollectingOutputChannel:
        return CallbackOutput(self.callback_endpoint)
