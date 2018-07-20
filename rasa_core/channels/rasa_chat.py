from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import requests
from flask import Blueprint, request, jsonify, abort
from flask_cors import CORS, cross_origin

from rasa_core.channels.channel import UserMessage, OutputChannel
from rasa_core.channels.rest import HttpInputComponent

logger = logging.getLogger(__name__)


class NoOutputChannel(OutputChannel):
    """Output channel that does nothing. Really, just nothing."""

    def send_text_message(self, recipient_id, message):
        # type: (Text, Text) -> None
        """Just ignore everything."""
        pass


class RasaChatInput(HttpInputComponent):
    def __init__(self, host):
        self.host = host

    def blueprint(self, on_new_message):
        rasa_chat = Blueprint('rasa_chat', __name__)
        CORS(rasa_chat)

        @rasa_chat.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @rasa_chat.route("/send", methods=['GET', 'POST'])
        @cross_origin()
        def receive():
            user = fetch_user(self.host, request)
            msg = request.json["message"]
            if user.get("role") == "admin":
                conversation_id = request.args.get("conversation",
                                                   user["username"])
            else:
                conversation_id = user["username"]
            on_new_message(UserMessage(msg, NoOutputChannel(),
                                       sender_id=conversation_id))

            return jsonify({"status": "ok"})

        return rasa_chat


def check_token(host, token):
    url = "{}/users/me".format(host)
    headers = {"Authorization": token}
    result = requests.get(url, headers=headers)

    if result.status_code == 200:
        return result.json()
    else:
        logger.info("Failed to check token: {}. "
                    "Content: {}".format(token, request.data))
        return None


def fetch_user(host, req):
    user = None
    if req.headers.get("Authorization"):
        user = check_token(host, req.headers.get("Authorization"))
    if user:
        return user

    user = check_token(host, req.args.get('token', default=None))
    if user:
        return user
    abort(401)
