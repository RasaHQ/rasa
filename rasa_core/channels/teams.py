from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
from requests import post

from flask import Blueprint, request, jsonify

from rasa_core.channels.channel import UserMessage, OutputChannel
from rasa_core.channels.rest import HttpInputComponent

logger = logging.getLogger(__name__)


class Teams(OutputChannel):
    """A Microsoft Teams communication channel. Not using any library yet."""

    def __init__(self, teams_id, teams_secret, conversation, bot_id,
                 service_url):
        # type: (Text, Text, Dict[Text]) -> None

        self.teams_id = teams_id
        self.teams_secret = teams_secret
        self.conversation = conversation
        self.global_uri = "%s/v3/" % service_url
        self.bot_id = bot_id

    def get_headers(self):
        # TODO : Not ask a new token if token still valid. Expiration check

        microsoft_oauth2_url = 'https://login.microsoftonline.com'
        microsoft_oauth2_path = 'botframework.com/oauth2/v2.0/token'
        uri = "%s/%s" % (microsoft_oauth2_url, microsoft_oauth2_path)
        grant_type = 'client_credentials'
        scope = 'https://api.botframework.com/.default'
        payload = {'client_id': self.teams_id,
                   'client_secret': self.teams_secret,
                   'grant_type': grant_type,
                   'scope': scope}

        token_response = post(uri, data=payload)
        if token_response.status_code == 200:
            token_data = token_response.json()
            access_token = token_data['access_token']
            token_expiration = token_data['expires_in']
            headers = {'content-type': 'application/json',
                       'Authorization': 'Bearer ' + access_token}
            return headers
        else:
            logger.error('Could not get Teams token')

    def send(self, recipient_id, message_data):
        # type: (Text, Dict[Text, Any]) -> None

        post_message_uri = self.global_uri + 'conversations/%s/activities' \
                                             % self.conversation['id']
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

        headers = self.get_headers()
        send_response = post(post_message_uri,
                             headers=headers,
                             data=json.dumps(data))
        
        status_code = send_response.status_code
        if status_code not in [200, 201, 202, 204]:
            logger.error('Error in send: %s', send_response.text)

    def send_text_message(self, recipient_id, message):
        logger.info("Sending message: " + message)

        text_message = {"text": message}
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
        custom_message = {"attachments": elements}
        self.send(recipient_id, custom_message)


class TeamsInput(HttpInputComponent):
    """Teams input channel implementation. Based on the HTTPInputChannel."""

    def __init__(self, teams_id, teams_secret):
        # type: (Text, Text) -> None
        """Create a facebook input channel.

        Needs a couple of settings to properly authenticate and validate
        messages.

        :param teams_id: Teams' API id
        :param teams_secret: Teams application secret
        """

        self.teams_id = teams_id
        self.teams_secret = teams_secret

    def blueprint(self, on_new_message):

        teams_webhook = Blueprint('teams_webhook', __name__)

        @teams_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @teams_webhook.route("/api/messages", methods=['POST'])
        def webhook():
            postdata = request.get_json(force=True)
            logger.info(postdata)
            try:
                out_channel = Teams(self.teams_id, self.teams_secret,
                                    postdata["conversation"],
                                    postdata["recipient"],
                                    postdata["serviceUrl"])
                user_msg = UserMessage(postdata["text"], out_channel,
                                       postdata["from"]["id"])
                on_new_message(user_msg)
            except Exception as e:
                logger.error("Exception when trying to handle "
                             "message.{0}".format(e))
                logger.error(e, exc_info=True)
                pass

            return "success"

        return teams_webhook
