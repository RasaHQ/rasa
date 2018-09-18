# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from flask import Blueprint, request, jsonify
from twilio.rest import Client

from rasa_core.channels import InputChannel
from rasa_core.channels import UserMessage, OutputChannel

logger = logging.getLogger(__name__)


class TwilioOutput(Client, OutputChannel):
    """Output channel for Twilio"""

    @classmethod
    def name(cls):
        return "twilio"

    def __init__(self, account_sid, auth_token, twilio_number):
        super(TwilioOutput, self).__init__(account_sid, auth_token)
        self.twilio_number = twilio_number
        self.send_retry = 0
        self.max_retry = 5

    def send_text_message(self, recipient_number, text):
        """Sends text message"""

        for message_part in text.split("\n\n"):
            self._send_text(recipient_number, message_part)

    def _send_text(self, recipient_number, text):
        from twilio.base.exceptions import TwilioRestException

        message = None
        try:
            while not message and self.send_retry < self.max_retry:
                message = self.messages.create(body=text,
                                               to=recipient_number,
                                               from_=self.twilio_number)
                self.send_retry += 1
        except TwilioRestException as e:
            logger.error("Something went wrong " + repr(e.msg))
        finally:
            self.send_retry = 0

        if not message and self.send_retry == self.max_retry:
            logger.error("Failed to send message. Max number of "
                         "retires exceeded.")

        return message

    def send_image_url(self, recipient_number, image_url):
        pass


class TwilioInput(InputChannel):
    """Twilio input channel"""

    @classmethod
    def name(cls):
        return "twilio"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(credentials.get("account_sid"),
                   credentials.get("auth_token"),
                   credentials.get("twilio_number"))

    def __init__(self, account_sid, auth_token, twilio_number,
                 debug_mode=True):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.twilio_number = twilio_number
        self.debug_mode = debug_mode

    def blueprint(self, on_new_message):
        twilio_webhook = Blueprint('twilio_webhook', __name__)

        @twilio_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @twilio_webhook.route("/webhook", methods=['POST'])
        def message():
            sender = request.values.get('From', None)
            text = request.values.get('Body', None)

            out_channel = TwilioOutput(self.account_sid, self.auth_token,
                                       self.twilio_number)

            if sender is not None and message is not None:
                try:
                    # @ signs get corrupted in SMSes by some carriers
                    text = text.replace('¡', '@')
                    on_new_message(UserMessage(text, out_channel, sender,
                                               input_channel=self.name()))
                except Exception as e:
                    logger.error("Exception when trying to handle "
                                 "message.{0}".format(e))
                    logger.error(e, exc_info=True)
                    if self.debug_mode:
                        raise
                    pass
            else:
                logger.debug("Invalid message")

            return "success"

        return twilio_webhook
