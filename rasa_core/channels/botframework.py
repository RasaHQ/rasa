# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import json
import logging
import requests
from flask import Blueprint, request, jsonify
from typing import Text, Dict, Any

from rasa_core.channels.channel import UserMessage, OutputChannel, InputChannel

logger = logging.getLogger(__name__)

MICROSOFT_OAUTH2_URL = 'https://login.microsoftonline.com'

MICROSOFT_OAUTH2_PATH = 'botframework.com/oauth2/v2.0/token'


class BotFramework(OutputChannel):
    """A Microsoft Bot Framework communication channel."""

    token_expiration_date = datetime.datetime.now()

    headers = None

    @classmethod
    def name(cls):
        return "botframework"

    def __init__(self, app_id, app_password, conversation, bot_id, service_url):
        # type: (Text, Text, Dict[Text], Text, Text) -> None

        self.app_id = app_id
        self.app_password = app_password
        self.conversation = conversation
        self.global_uri = "{}v3/".format(service_url)
        self.bot_id = bot_id

    def _get_headers(self):
        if BotFramework.token_expiration_date < datetime.datetime.now():
            uri = "{}/{}".format(MICROSOFT_OAUTH2_URL, MICROSOFT_OAUTH2_PATH)
            grant_type = 'client_credentials'
            scope = 'https://api.botframework.com/.default'
            payload = {'client_id': self.app_id,
                       'client_secret': self.app_password,
                       'grant_type': grant_type,
                       'scope': scope}

            token_response = requests.post(uri, data=payload)

            if token_response.ok:
                token_data = token_response.json()
                access_token = token_data['access_token']
                token_expiration = token_data['expires_in']

                BotFramework.token_expiration_date = \
                    datetime.datetime.now() + \
                    datetime.timedelta(seconds=int(token_expiration))

                BotFramework.headers = {"content-type": "application/json",
                                        "Authorization": "Bearer %s" %
                                                         access_token}
                return BotFramework.headers
            else:
                logger.error('Could not get BotFramework token')
        else:
            return BotFramework.headers

    def send(self, recipient_id, message_data):
        # type: (Text, Dict[Text, Any]) -> None

        post_message_uri = self.global_uri + \
                           'conversations/{}/activities'.format(
                                   self.conversation['id'])
        data = {"type": "message",
                "recipient": {
                    "id": recipient_id
                },
                "from": self.bot_id,
                "channelData": {
                    "notification": {
                        "alert": "true"
                    }
                },
                "text": ""}

        data.update(message_data)
        headers = self._get_headers()
        send_response = requests.post(post_message_uri,
                                      headers=headers,
                                      data=json.dumps(data))

        if not send_response.ok:
            logger.error("Error trying to send botframework messge. "
                         "Response: %s", send_response.text)

    def send_text_message(self, recipient_id, message):
        for message_part in message.split("\n\n"):
            text_message = {"text": message_part}
            self.send(recipient_id, text_message)

    def send_image_url(self, recipient_id, image_url):
        hero_content = {
            'contentType': 'application/vnd.microsoft.card.hero',
            'content': {
                'images': [{'url': image_url}]
            }
        }

        image_message = {"attachments": [hero_content]}
        self.send(recipient_id, image_message)

    def send_text_with_buttons(self, recipient_id, message, buttons, **kwargs):
        hero_content = {
            'contentType': 'application/vnd.microsoft.card.hero',
            'content': {
                'subtitle': message,
                'buttons': buttons
            }
        }

        buttons_message = {"attachments": [hero_content]}
        self.send(recipient_id, buttons_message)

    def send_custom_message(self, recipient_id, elements):
        self.send(recipient_id, elements[0])


class BotFrameworkInput(InputChannel):
    """Bot Framework input channel implementation."""

    @classmethod
    def name(cls):
        return "botframework"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(credentials.get("app_id"), credentials.get("app_password"))

    def __init__(self, app_id, app_password):
        # type: (Text, Text) -> None
        """Create a Bot Framework input channel.

        :param app_id: Bot Framework's API id
        :param app_password: Bot Framework application secret
        """

        self.app_id = app_id
        self.app_password = app_password

    def blueprint(self, on_new_message):

        botframework_webhook = Blueprint('botframework_webhook', __name__)

        @botframework_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @botframework_webhook.route("/webhook", methods=['POST'])
        def webhook():
            postdata = request.get_json(force=True)

            try:
                if postdata["type"] == "message":
                    out_channel = BotFramework(self.app_id, self.app_password,
                                               postdata["conversation"],
                                               postdata["recipient"],
                                               postdata["serviceUrl"])

                    user_msg = UserMessage(postdata["text"], out_channel,
                                           postdata["from"]["id"],
                                           input_channel=self.name())
                    on_new_message(user_msg)
                else:
                    logger.info("Not received message type")
            except Exception as e:
                logger.error("Exception when trying to handle "
                             "message.{0}".format(e))
                logger.error(e, exc_info=True)
                pass

            return "success"

        return botframework_webhook
