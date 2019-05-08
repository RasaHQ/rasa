import logging
from sanic import Blueprint, response
from typing import Text, Dict, Any

from rasa.core.channels.channel import UserMessage, OutputChannel, InputChannel

logger = logging.getLogger(__name__)


class RocketChatBot(OutputChannel):
    @classmethod
    def name(cls):
        return "rocketchat"

    def __init__(self, user, password, server_url):
        from rocketchat_API.rocketchat import RocketChat

        self.rocket = RocketChat(user, password, server_url=server_url)

    async def send_text_message(self, recipient_id, message):
        """Messages handler."""

        for message_part in message.split("\n\n"):
            self.rocket.chat_post_message(message_part, room_id=recipient_id)

    async def send_image_url(self, recipient_id, image_url):
        image_attachment = [{"image_url": image_url, "collapsed": False}]

        return self.rocket.chat_post_message(
            None, room_id=recipient_id, attachments=image_attachment
        )

    async def send_attachment(self, recipient_id, attachment, message=""):
        return self.rocket.chat_post_message(
            None, room_id=recipient_id, attachments=[attachment]
        )

    @staticmethod
    def _convert_to_rocket_buttons(buttons):
        return [
            {
                "text": b["title"],
                "msg": b["payload"],
                "type": "button",
                "msg_in_chat_window": True,
            }
            for b in buttons
        ]

    async def send_text_with_buttons(self, recipient_id, message, buttons, **kwargs):
        # implementation is based on
        # https://github.com/RocketChat/Rocket.Chat/pull/11473
        # should work in rocket chat >= 0.69.0
        button_attachment = [{"actions": self._convert_to_rocket_buttons(buttons)}]

        return self.rocket.chat_post_message(
            message, room_id=recipient_id, attachments=button_attachment
        )

    async def send_elements(self, recipient_id, elements):
        return self.rocket.chat_post_message(
            None, room_id=recipient_id, attachments=elements
        )

    async def send_custom_json(self, recipient_id, kwargs: Dict[Text, Any]):
        text = kwargs.pop("text")

        if kwargs.get("channel"):
            if kwargs.get("room_id"):
                logger.warning(
                    "Only one of `channel` or `room_id` can be passed to a RocketChat message post. Defaulting to `channel`."
                )
                del kwargs["room_id"]
            return self.rocket.chat_post_message(text, **kwargs)
        else:
            kwargs.setdefault("room_id", recipient_id)
            return self.rocket.chat_post_message(text, **kwargs)


class RocketChatInput(InputChannel):
    """RocketChat input channel implementation."""

    @classmethod
    def name(cls):
        return "rocketchat"

    @classmethod
    def from_credentials(cls, credentials):
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

    async def send_message(self, text, sender_name, recipient_id, on_new_message):
        if sender_name != self.user:
            output_channel = RocketChatBot(self.user, self.password, self.server_url)

            user_msg = UserMessage(
                text, output_channel, recipient_id, input_channel=self.name()
            )
            await on_new_message(user_msg)

    def blueprint(self, on_new_message):
        rocketchat_webhook = Blueprint("rocketchat_webhook", __name__)

        @rocketchat_webhook.route("/", methods=["GET"])
        async def health(request):
            return response.json({"status": "ok"})

        @rocketchat_webhook.route("/webhook", methods=["GET", "POST"])
        async def webhook(request):
            output = request.json
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

                await self.send_message(text, sender_name, recipient_id, on_new_message)

            return response.text("")

        return rocketchat_webhook
