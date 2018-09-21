from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from typing import Optional, Text

import socketio
from flask import Blueprint, jsonify

from rasa_core.channels import InputChannel
from rasa_core.channels.channel import (
    UserMessage,
    OutputChannel)

logger = logging.getLogger(__name__)


class SocketBlueprint(Blueprint):
    def __init__(self, sio, *args, **kwargs):
        self.sio = sio
        super(SocketBlueprint, self).__init__(*args, **kwargs)

    def register(self, app, options, first_registration=False):
        app.wsgi_app = socketio.Middleware(self.sio, app.wsgi_app)
        super(SocketBlueprint, self).register(app, options, first_registration)


class SocketIOOutput(OutputChannel):

    @classmethod
    def name(cls):
        return "socketio"

    def __init__(self, sio, bot_message_evt):
        self.sio = sio
        self.bot_message_evt = bot_message_evt

    def send(self, recipient_id, message):
        # type: (Text, Any) -> None
        """Sends a message to the recipient."""
        self.sio.emit(message, room=recipient_id)

    def _send_message(self, recipient_id, response):
        # type: (Text, Any) -> None
        """Sends a message to the recipient using the bot event."""
        self.sio.emit(self.bot_message_evt, response, room=recipient_id)

    def send_text_message(self, recipient_id, message):
        # type: (Text, Text) -> None
        """Send a message through this channel."""

        self._send_message(recipient_id, {"text": message})

    def send_image_url(self, recipient_id, image_url):
        # type: (Text, Text) -> None
        """Sends an image. Default will just post the url as a string."""
        message = {
            "attachment": {
                "type": "image",
                "payload": {"src": image_url}
            }
        }
        self._send_message(recipient_id, message)

    def send_text_with_buttons(self, recipient_id, text, buttons, **kwargs):
        # type: (Text, Text, List[Dict[Text, Any]], **Any) -> None
        """Sends buttons to the output."""

        message = {
            "text": text,
            "quick_replies": []
        }

        for button in buttons:
            message["quick_replies"].append({
                "content_type": "text",
                "title": button['title'],
                "payload": button['payload']
            })

        self._send_message(recipient_id, message)

    def send_custom_message(self, recipient_id, elements):
        # type: (Text, List[Dict[Text, Any]]) -> None
        """Sends elements to the output."""

        message = {"attachment": {
            "type": "template",
            "payload": {
                "template_type": "generic",
                "elements": elements[0]
            }}}

        self._send_message(recipient_id, message)


class SocketIOInput(InputChannel):
    """A socket.io input channel."""

    @classmethod
    def name(cls):
        return "socketio"

    @classmethod
    def from_credentials(cls, credentials):
        credentials = credentials or {}
        return cls(credentials.get("user_message_evt", "user_uttered"),
                   credentials.get("bot_message_evt", "bot_uttered"),
                   credentials.get("namespace"))

    def __init__(self,
                 user_message_evt="user_uttered",  # type: Text
                 bot_message_evt="bot_uttered",  # type: Text
                 namespace=None  # type: Optional[Text]
                 ):
        self.bot_message_evt = bot_message_evt
        self.user_message_evt = user_message_evt
        self.namespace = namespace

    def blueprint(self, on_new_message):
        sio = socketio.Server()
        socketio_webhook = SocketBlueprint(sio, 'socketio_webhook', __name__)

        @socketio_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @sio.on('connect', namespace=self.namespace)
        def connect(sid, environ):
            logger.debug("User {} connected to socketio endpoint.".format(sid))

        @sio.on('disconnect', namespace=self.namespace)
        def disconnect(sid):
            logger.debug("User {} disconnected from socketio endpoint."
                         "".format(sid))

        @sio.on(self.user_message_evt, namespace=self.namespace)
        def handle_message(sid, data):
            output_channel = SocketIOOutput(sio, self.bot_message_evt)
            message = UserMessage(data['message'], output_channel, sid,
                                  input_channel=self.name())
            on_new_message(message)

        return socketio_webhook
