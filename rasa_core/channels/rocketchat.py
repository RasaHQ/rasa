from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import logging
import threading

from flask import Blueprint, request, jsonify, make_response
from rocketchat_py_sdk import driver

from rasa_core.channels.channel import UserMessage, OutputChannel
from rasa_core.channels.rest import HttpInputComponent

logger = logging.getLogger(__name__)


class RocketChatInput(HttpInputComponent):

    """RocketChat input channel implementation."""

    def __init__(self, user=None, password=None,
                 server_url='open.rocket.chat', ssl=True):

        self.user = user
        self.password = password
        self.server_url = server_url
        self.ssl = ssl

        self.rocketchat_bot = RocketChatBot(self.user, self.password,
                                            server=self.server_url,
                                            ssl=self.ssl)
        self.rocketchat_bot.start()

    def send_message(self, text, sender_name, recipient_id, on_new_message):
        try:
            if sender_name != self.user:
                out_channel = self.rocketchat_bot
                user_msg = UserMessage(text, out_channel, recipient_id)
                on_new_message(user_msg)

        except Exception as e:
            logger.error("Exception when trying to handle "
                         "message.{0}".format(e))
            logger.error(e, exc_info=True)
            pass

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
                logger.debug(output)

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


class RocketChatBot(OutputChannel):
    def __init__(self, user, password, server, ssl):
        self.username = user
        self.password = password
        self.connector = driver.Driver(url=server , ssl=ssl)
        self.users = {}

    """
    Internal callback handlers
    """
    def method_callback(self, error, data):
        if error:
            logger.error('[-] callback error:')
            logger.error(error)
        else:
            logger.info("[+] callback success")
            logger.debug(data)

    """
    Public initializers
    """
    def start(self):
        self.connector.connect()
        self.connector.login(user=self.username, password=self.password,
                             callback=self.method_callback)
        self.connector.subscribe_to_messages()

    """
    Messages handlers
    """
    def send_text_message(self, recipient_id, message):
        if recipient_id not in self.users:
            self.users[recipient_id] = RocketchatHandleMessages(recipient_id, self)
        self.users[recipient_id].add_message(message)


class RocketchatHandleMessages:
    def __init__(self, rid, bot):
        self.rid = rid
        self.messages = []
        self.message_index = 0
        self.bot = bot

    def send_message(self):
        msg = self.messages[self.message_index]
        self.message_index += 1

        logger.info('[+] send message {}: {}'.format(self.rid, msg['message']))

        self.bot.connector.send_message(self.rid, msg['message'])

        if self.message_index == len(self.messages):
            logger.info('deactivate typing for {}'.format(self.rid))

            self.bot.connector.call(
                'stream-notify-room',
                [self.rid + '/typing', self.bot.username, False]
            )

            self.messages = []
            self.message_index = 0

    def add_message(self, message):
        logger.info('activate typing for {}'.format(self.rid))
        self.bot.connector.call(
            'stream-notify-room',
            [self.rid + '/typing', self.bot.username, True]
        )

        wait_time = 1

        if len(self.messages) != 0:
            last_msg = self.messages[-1]
            n_words = len(last_msg['message'].split(' '))

            words_per_sec = 5
            wait_time = max(1, n_words // words_per_sec) + last_msg['time']

        threading.Timer(wait_time, self.send_message).start()

        logger.info('[ ] schedule message {}: {}'.format(self.rid, message))
        self.messages.append({'message': message, 'time': wait_time})
