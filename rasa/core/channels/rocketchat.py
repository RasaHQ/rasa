import logging
from sanic import Blueprint, response
from sanic.request import Request
from typing import Text, Dict, Any, List, Iterable, Optional, Callable, Awaitable

from rasa.core.channels.channel import UserMessage, OutputChannel, InputChannel
from sanic.response import HTTPResponse

logger = logging.getLogger(__name__)


class RocketChatBot(OutputChannel):
    @classmethod
    def name(cls) -> Text:
        return "rocketchat"

    def __init__(self, user: Text, password: Text, server_url: Text) -> None:
        from rocketchat_API.rocketchat import RocketChat

        self.rocket = RocketChat(user, password, server_url=server_url)

    @staticmethod
    def _convert_to_rocket_buttons(buttons: List[Dict]) -> List[Dict]:
        return [
            {
                "text": b["title"],
                "msg": b["payload"],
                "type": "button",
                "msg_in_chat_window": True,
            }
            for b in buttons
        ]

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Send message to output channel."""
        for message_part in text.strip().split("\n\n"):
            self.rocket.chat_post_message(message_part, room_id=recipient_id)

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        image_attachment = [{"image_url": image, "collapsed": False}]

        return self.rocket.chat_post_message(
            None, room_id=recipient_id, attachments=image_attachment
        )

    async def send_attachment(
        self, recipient_id: Text, attachment: Text, **kwargs: Any
    ) -> None:
        return self.rocket.chat_post_message(
            None, room_id=recipient_id, attachments=[attachment]
        )

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        # implementation is based on
        # https://github.com/RocketChat/Rocket.Chat/pull/11473
        # should work in rocket chat >= 0.69.0
        button_attachment = [{"actions": self._convert_to_rocket_buttons(buttons)}]

        return self.rocket.chat_post_message(
            text, room_id=recipient_id, attachments=button_attachment
        )

    async def send_elements(
        self, recipient_id: Text, elements: Iterable[Dict[Text, Any]], **kwargs: Any
    ) -> None:
        return self.rocket.chat_post_message(
            None, room_id=recipient_id, attachments=elements
        )

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        text = json_message.pop("text")

        if json_message.get("channel"):
            if json_message.get("room_id"):
                logger.warning(
                    "Only one of `channel` or `room_id` can be passed to a RocketChat "
                    "message post. Defaulting to `channel`."
                )
                del json_message["room_id"]
            return self.rocket.chat_post_message(text, **json_message)
        else:
            json_message.setdefault("room_id", recipient_id)
            return self.rocket.chat_post_message(text, **json_message)


class RocketChatInput(InputChannel):
    """RocketChat input channel implementation."""

    @classmethod
    def name(cls) -> Text:
        return "rocketchat"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(
            credentials.get("user"),
            credentials.get("password"),
            credentials.get("server_url"),
        )

    def __init__(self, user: Text, password: Text, server_url: Text) -> None:
        self.user = user
        self.password = password
        self.server_url = server_url

    async def send_message(
        self,
        text: Optional[Text],
        sender_name: Optional[Text],
        recipient_id: Optional[Text],
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        metadata: Optional[Dict],
    ) -> None:
        if sender_name != self.user:
            output_channel = self.get_output_channel()

            user_msg = UserMessage(
                text,
                output_channel,
                recipient_id,
                input_channel=self.name(),
                metadata=metadata,
            )
            await on_new_message(user_msg)

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        rocketchat_webhook = Blueprint("rocketchat_webhook", __name__)

        @rocketchat_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @rocketchat_webhook.route("/webhook", methods=["GET", "POST"])
        async def webhook(request: Request) -> HTTPResponse:
            output = request.json
            metadata = self.get_metadata(request)
            if output:
                if "visitor" not in output:
                    sender_name = output.get("user_name", None)
                    text = output.get("text", None)
                    recipient_id = output.get("channel_id", None)
                else:
                    messages_list = output.get("messages", None)
                    text = messages_list[0].get("msg", None)
                    sender_name = messages_list[0].get("username", None)
                    recipient_id = output.get("_id")

                await self.send_message(
                    text, sender_name, recipient_id, on_new_message, metadata
                )

            return response.text("")

        return rocketchat_webhook

    def get_output_channel(self) -> OutputChannel:
        return RocketChatBot(self.user, self.password, self.server_url)
