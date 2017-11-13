from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from builtins import str
from flask import Blueprint, request, jsonify
from pymessenger.bot import Bot

from rasa_core.channels.channel import UserMessage, OutputChannel
from rasa_core.channels.rest import HttpInputComponent

logger = logging.getLogger(__name__)


class MessengerBot(Bot, OutputChannel):
    """A bot that uses fb-messenger to communicate."""

    def __init__(self, access_token):
        super(MessengerBot, self).__init__(access_token)

    def send_text_with_buttons(self, recipient_id, text, buttons, **kwargs):
        # buttons is a list of tuples: [(option_name,payload)]
        if len(buttons) > 3:
            logger.warn("Facebook API currently allows only up to 3 buttons. "
                        "If you add more, all will be ignored.")
            return self.send_text_message(recipient_id, text)
        else:
            self._add_postback_info(buttons)
            return self.send_button_message(recipient_id, text, buttons)

    def _add_postback_info(self, buttons):
        for button in buttons:
            button['type'] = "postback"

    def send_custom_message(self, recipient_id, elements):
        for element in elements:
            self._add_postback_info(element['buttons'])
        return self.send_generic_message(recipient_id, elements)


class FacebookInput(HttpInputComponent):
    def __init__(self, fb_verify, fb_secret, fb_tokens, debug_mode):
        self.fb_verify = fb_verify
        self.fb_secret = fb_secret
        self.debug_mode = debug_mode
        self.fb_tokens = {str(k): v for k, v in fb_tokens.items()}

    @staticmethod
    def _is_user_message(fb_event):
        return (fb_event.get('message') and
                fb_event['message'].get('text') and
                not fb_event['message'].get("is_echo"))

    @staticmethod
    def _is_audio_message(fb_event):
        return (fb_event.get('message') and
                fb_event['message'].get('attachments') and
                fb_event['message']['attachments'][0]['type'] == 'audio')

    @staticmethod
    def _get_audio_attachment_url(fb_event):
        attachment = fb_event['message']['attachments'][0]
        return attachment['payload']['url']

    @staticmethod
    def _is_quick_reply(fb_event):
        return fb_event.get('postback') and fb_event['postback'].get('payload')

    @staticmethod
    def _get_quick_reply(fb_event):
        return fb_event['postback']['payload']

    def blueprint(self, on_new_message):
        from pymessenger.utils import validate_hub_signature

        fb_webhook = Blueprint('fb_webhook', __name__)

        @fb_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @fb_webhook.route("/webhook", methods=['GET', 'POST'])
        def hello():
            if request.method == 'GET':
                if request.args.get("hub.verify_token") == self.fb_verify:
                    return request.args.get("hub.challenge")
                else:
                    logger.warn("Invalid fb verify token! Make sure this matches "
                                "your webhook settings on the facebook app.")
                    return "failure, invalid token"
            if request.method == 'POST':

                signature = request.headers.get("X-Hub-Signature") or ''
                if not validate_hub_signature(self.fb_secret, request.data,
                                              signature):
                    logger.warn("Wrong fb secret! Make sure this matches the "
                                 "secret in your facebook app settings")
                    return "not validated"

                output = request.json
                page_id = output['entry'][0]['id']
                event = output['entry'][0]['messaging']
                for x in event:
                    if self._is_user_message(x):
                        text = x['message']['text']
                    elif self._is_audio_message(x):
                        text = self._get_audio_attachment_url(x)
                    elif self._is_quick_reply(x):
                        text = self._get_quick_reply(x)
                    else:
                        continue
                    try:
                        sender_id = x['sender']['id']
                        if page_id in self.fb_tokens:
                            out_channel = MessengerBot(self.fb_tokens[page_id])
                            user_msg = UserMessage(text, out_channel, sender_id)
                            on_new_message(user_msg)
                        else:
                            raise Exception("Unknown page id '{}'. Make sure to"
                                            " add a page token to the "
                                            "configuration.".format(page_id))
                    except Exception as e:
                        logger.error("Exception when trying to handle "
                                     "message.{0}".format(e))
                        logger.error(e,exc_info=True)
                        if self.debug_mode:
                            raise
                        pass

                return "success"

        return fb_webhook
