from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from flask import Blueprint, request, jsonify, make_response
from mattermostwrapper import MattermostAPI
from typing import Text

from rasa_core.channels.channel import UserMessage, OutputChannel, InputChannel

logger = logging.getLogger(__name__)


class MattermostBot(MattermostAPI, OutputChannel):
    """A Mattermost communication channel"""

    @classmethod
    def name(cls):
        return "mattermost"

    def __init__(self, url, team, user, pw, bot_channel):
        self.url = url
        self.team = team
        self.user = user
        self.pw = pw
        self.bot_channel = bot_channel

        super(MattermostBot, self).__init__(url, team)
        super(MattermostBot, self).login(user, pw)

    def send_text_message(self, recipient_id, message):
        for message_part in message.split("\n\n"):
            super(MattermostBot, self).post_channel(self.bot_channel,
                                                    message_part)


class MattermostInput(InputChannel):
    """Mattermost input channel implemenation."""

    @classmethod
    def name(cls):
        return "mattermost"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(credentials.get("url"),
                   credentials.get("team"),
                   credentials.get("user"),
                   credentials.get("pw"))

    def __init__(self, url, team, user, pw):
        # type: (Text, Text, Text, Text) -> None
        """Create a Mattermost input channel.
        Needs a couple of settings to properly authenticate and validate
        messages.
        :param url: Your Mattermost team url including /v4 example
                    https://mysite.example.com/api/v4

        :param team: Your mattermost team name

        :param user: Your mattermost userid that will post messages

        :param pw: Your mattermost password for your user
        """
        self.url = url
        self.team = team
        self.user = user
        self.pw = pw

    def blueprint(self, on_new_message):
        mattermost_webhook = Blueprint('mattermost_webhook', __name__)

        @mattermost_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @mattermost_webhook.route("/webhook", methods=['POST'])
        def webhook():
            request.get_data()
            if request.json:
                output = request.json
                # splitting to get rid of the @botmention
                # trigger we are using for this
                text = output['text'].split(" ", 1)
                text = text[1]
                sender_id = output['user_id']
                self.bot_channel = output['channel_id']
                try:
                    out_channel = MattermostBot(self.url,
                                                self.team,
                                                self.user,
                                                self.pw,
                                                self.bot_channel)
                    user_msg = UserMessage(text, out_channel, sender_id,
                                           input_channel=self.name())
                    on_new_message(user_msg)
                except Exception as e:
                    logger.error("Exception when trying to handle "
                                 "message.{0}".format(e))
                    logger.error(e, exc_info=True)
                    pass
            return make_response()

        return mattermost_webhook
