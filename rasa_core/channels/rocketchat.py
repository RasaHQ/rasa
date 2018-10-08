from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from typing import Text

from flask import Blueprint, request, jsonify, make_response

from rasa_core.channels.channel import UserMessage, OutputChannel, InputChannel

logger = logging.getLogger(__name__)


class RocketChatBot(OutputChannel):
    @classmethod
    def name(cls):
        return "rocketchat"

    def __init__(self, user, password, server_url):
        from rocketchat_API.rocketchat import RocketChat

        self.rocket = RocketChat(user, password, server_url=server_url)

    def send_text_message(self, recipient_id, message):
        """Messages handler."""

        for message_part in message.split("\n\n"):
            self.rocket.chat_post_message(message_part,
                                          room_id=recipient_id)

    def send_image_url(self, recipient_id, image_url):
        image_attachment = [{
            "image_url": image_url,
            "collapsed": False,
        }]

        return self.rocket.chat_post_message(None,
                                             room_id=recipient_id,
                                             attachments=image_attachment)

    def send_attachment(self, recipient_id, attachment, message=""):
        return self.rocket.chat_post_message(None,
                                             room_id=recipient_id,
                                             attachments=[attachment])

    @staticmethod
    def _convert_to_rocket_buttons(buttons):
        return [{"text": b['title'],
                 "msg": b['payload'],
                 "type": "button",
                 "msg_in_chat_window": True}
                for b in buttons]

    def send_text_with_buttons(self, recipient_id, message, buttons, **kwargs):
        # implementation is based on
        # https://github.com/RocketChat/Rocket.Chat/pull/11473
        # should work in rocket chat >= 0.69.0
        button_attachment = [
            {"actions": self._convert_to_rocket_buttons(buttons)}]

        return self.rocket.chat_post_message(message,
                                             room_id=recipient_id,
                                             attachments=button_attachment)

    def send_custom_message(self, recipient_id, elements):
        return self.rocket.chat_post_message(None, room_id=recipient_id, attachments=elements)


class RocketChatInput(InputChannel):
    """RocketChat input channel implementation."""

    @classmethod
    def name(cls):
        return "rocketchat"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(credentials.get("user"),
                   credentials.get("password"),
                   credentials.get("server_url"))

    def __init__(self, user, password, server_url):
        # type: (Text, Text, Text) -> None

        self.user = user
        self.password = password
        self.server_url = server_url

    def send_message(self, text, sender_name, recipient_id, on_new_message):
        if sender_name != self.user:
            output_channel = RocketChatBot(
                    self.user, self.password, self.server_url)

            user_msg = UserMessage(text, output_channel, recipient_id,
                                   input_channel=self.name())
            on_new_message(user_msg)

    def blueprint(self, on_new_message):
        rocketchat_webhook = Blueprint('rocketchat_webhook', __name__)

        @rocketchat_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @rocketchat_webhook.route("/webhook", methods=['GET', 'POST'])
        def webhook():
            request.get_data()
            if request.json:
                output = request.json

                if "visitor" not in output:
                    sender_id = output.get("user_id", None)
                    sender_name = output.get("user_name", None)
                    text = output.get("text", None)
                    recipient_id = output.get("channel_id", None)
                else:
                    sender_id = output.get("visitor", None).get("_id", None)
                    messages_list = output.get("messages", None)
                    text = messages_list[0].get("msg", None)
                    sender_name = messages_list[0].get("username", None)
                    recipient_id = output.get("_id")

                self.send_message(text, sender_name, recipient_id,
                                  on_new_message)

            return make_response()

        return rocketchat_webhook
