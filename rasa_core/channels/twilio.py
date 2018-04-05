from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from flask import Blueprint, request, jsonify

from twilio.rest import Client

from rasa_core.channels import UserMessage, OutputChannel
from rasa_core.channels.rest import HttpInputComponent

logger = logging.getLogger(__name__)


class TwilioOutputChannel(Client, OutputChannel):
    """Output channel for Twilio"""
    max_retry = 5

    def __init__(self, account_sid, auth_token, twilio_number):
        super(TwilioOutputChannel, self).__init__(account_sid, auth_token)
        self.twilio_number = twilio_number
        self.send_retry = 0

    def send_text_message(self, recipient_number, text):
        from twilio.base.exceptions import TwilioRestException
        message = None
        try:
            while not message and self.send_retry < self.max_retry:
                message = self.client.messages.create(body=text,
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


class TwilioInput(HttpInputComponent):
    """Twilio input channel"""

    def __init__()
    @app.route("/helvetia-demo/webhook", methods=['GET', 'POST'])
    def hello():
        sender = request.values.get('From', None)
        message = request.values.get('Body', None)

        if sender != None and message != None:
            # do stuff

        return "success"
