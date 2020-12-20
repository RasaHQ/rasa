import inspect
import logging
import uuid
from asyncio import CancelledError
from typing import (Any, Awaitable, Callable, Dict, Iterable, List, Optional,
                    Text)

from rasa.core.channels.channel import (CollectingOutputChannel, InputChannel,
                                        OutputChannel, UserMessage)
from rasa.utils.common import raise_warning
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse
from stream_chat import StreamChat

logger = logging.getLogger(__name__)


class StreamIOOutput(OutputChannel):
    @classmethod
    def name(cls) -> Text:
        return "getstream"

    def __init__(self, api_key: Text, api_secret: Text, channel_id: Text) -> None:
        self.bot = {"id": "chatbot", "name": "chatbot", "role": "admin"}
        chat = StreamChat(api_key=api_key, api_secret=api_secret)
        chat.update_user(self.bot)
        self.channel = chat.channel("messaging", channel_id)

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Send message to output channel"""
        for message_part in text.strip().split("\n\n"):
            self.channel.send_message({"text": message_part}, self.bot.get("id"))

    @staticmethod
    def _convert_to_getstream_buttons(buttons: List[Dict]) -> List[Dict]:
        return [
            {
                "text": b["title"],
                "value": b["payload"],
                "style": "primary",
                "type": "button",
            }
            for b in buttons
        ]

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        attachments = [
            {
                "type": "form",
                "title": text,
                "actions": self._convert_to_getstream_buttons(buttons),
            }
        ]

        self.channel.send_message(
            {"text": text, "attachments": attachments}, self.bot.get("id")
        )


class StreamIOInput(InputChannel):
    """A custom http input channel.

    This implementation is the basis for a custom implementation of a chat
    frontend. You can customize this to send messages to Rasa Core and
    retrieve responses from the agent."""

    @classmethod
    def name(cls):
        return "getstream"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        # pytype: disable=attribute-error
        return cls(
            credentials.get("api_key"),
            credentials.get("api_secret"),
        )
        # pytype: enable=attribute-error

    def __init__(
        self,
        api_key: Optional[Text],
        api_secret: Optional[Text],
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret

    async def _extract_sender(self, req: Request) -> Optional[Text]:
        return req.json.get("user", {}).get("id", None)

    def _extract_sender_name(self, req: Request) -> Optional[Text]:
        return req.json.get("user", {}).get("name", None)

    def _extract_channel_id(self, req: Request) -> Optional[Text]:
        return req.json.get("channel_id", None)

    # noinspection PyMethodMayBeStatic
    def _extract_message(self, req: Request) -> Optional[Text]:
        customField = req.json.get("message", {}).get("customField", None)
        return (
            customField if customField else req.json.get("message", {}).get("text", "")
        )

    def get_metadata(self, req: Request) -> Optional[Dict[Text, Any]]:
        return req.json.get("message", {}).get("customData", {}).get("token", None)

    def _extract_role(self, req: Request) -> Optional[Text]:
        return req.json.get("user", {}).get("role", "")

    async def get_output_channel(self, channel_id: Text) -> OutputChannel:
        return StreamIOOutput(self.api_key, self.api_secret, channel_id)

    def blueprint(self, on_new_message: Callable[[UserMessage], Awaitable[None]]):
        custom_webhook = Blueprint(
            "custom_webhook_{}".format(type(self).__name__),
            inspect.getmodule(self).__name__,
        )

        @custom_webhook.route("/", methods=["GET"])
        async def health(request: Request):
            return response.json({"status": "ok"})

        @custom_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request) -> HTTPResponse:
            sender_id = await self._extract_sender(request)
            user_message = self._extract_message(request)
            sender_name = self._extract_sender_name(request)
            channel_id = self._extract_channel_id(request)
            metadata = self.get_metadata(request)
            role = self._extract_role(request)

            if not request.json.get("type", "") == "message.new":
                return response.text("")

            if role != "admin":
                collector = await self.get_output_channel(channel_id)
                try:
                    message = await on_new_message(
                        UserMessage(
                            text=user_message,
                            output_channel=collector,
                            sender_id=sender_id,
                            input_channel=self.name(),
                            metadata=metadata,
                        )
                    )
                    logger.info(f"{request.url}  |  {sender_id}: {user_message}")
                except CancelledError:
                    logger.error(
                        "Message handling timed out for "
                        "user message '{}'.".format(user_message)
                    )
                except Exception:
                    logger.exception(
                        "An exception occured while handling "
                        "user message '{}'.".format(user_message)
                    )

            return response.text("")

        return custom_webhook
