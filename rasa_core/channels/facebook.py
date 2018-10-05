from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import hashlib
import hmac
import logging
from typing import Text, List, Dict, Any, Callable

import six
from fbmessenger import (
    BaseMessenger, MessengerClient, attachments)
from fbmessenger.elements import Text as FBText
from flask import Blueprint, request, jsonify

from rasa_core.channels.channel import UserMessage, OutputChannel, InputChannel

logger = logging.getLogger(__name__)


class Messenger(BaseMessenger):
    """Implement a fbmessenger to parse incoming webhooks and send msgs."""

    @classmethod
    def name(cls):
        return "facebook"

    def __init__(self, page_access_token, on_new_message):
        # type: (Text, Callable[[UserMessage], None]) -> None

        self.page_access_token = page_access_token
        self.on_new_message = on_new_message
        super(Messenger, self).__init__(self.page_access_token)

    @staticmethod
    def _is_audio_message(message):
        # type: (Dict[Text, Any]) -> bool
        """Check if the users message is a recorced voice message."""
        return (message.get('message') and
                message['message'].get('attachments') and
                message['message']['attachments'][0]['type'] == 'audio')

    @staticmethod
    def _is_user_message(message):
        # type: (Dict[Text, Any]) -> bool
        """Check if the message is a message from the user"""
        return (message.get('message') and
                message['message'].get('text') and
                not message['message'].get("is_echo"))

    def message(self, message):
        # type: (Dict[Text, Any]) -> None
        """Handle an incoming event from the fb webhook."""

        if self._is_user_message(message):
            text = message['message']['text']
        elif self._is_audio_message(message):
            attachment = message['message']['attachments'][0]
            text = attachment['payload']['url']
        else:
            logger.warning("Received a message from facebook that we can not "
                           "handle. Message: {}".format(message))
            return

        self._handle_user_message(text, self.get_user_id())

    def postback(self, message):
        # type: (Dict[Text, Any]) -> None
        """Handle a postback (e.g. quick reply button)."""

        text = message['postback']['payload']
        self._handle_user_message(text, self.get_user_id())

    def _handle_user_message(self, text, sender_id):
        # type: (Text, Text) -> None
        """Pass on the text to the dialogue engine for processing."""

        out_channel = MessengerBot(self.client)
        user_msg = UserMessage(text, out_channel, sender_id,
                               input_channel=self.name())

        # noinspection PyBroadException
        try:
            self.on_new_message(user_msg)
        except Exception:
            logger.exception("Exception when trying to handle webhook "
                             "for facebook message.")
            pass

    def delivery(self, message):
        # type: (Dict[Text, Any]) -> None
        """Do nothing. Method to handle `message_deliveries`"""
        pass

    def read(self, message):
        # type: (Dict[Text, Any]) -> None
        """Do nothing. Method to handle `message_reads`"""
        pass

    def account_linking(self, message):
        # type: (Dict[Text, Any]) -> None
        """Do nothing. Method to handle `account_linking`"""
        pass

    def optin(self, message):
        # type: (Dict[Text, Any]) -> None
        """Do nothing. Method to handle `messaging_optins`"""
        pass


class MessengerBot(OutputChannel):
    """A bot that uses fb-messenger to communicate."""

    @classmethod
    def name(cls):
        return "facebook"

    def __init__(self, messenger_client):
        # type: (MessengerClient) -> None

        self.messenger_client = messenger_client
        super(MessengerBot, self).__init__()

    def send(self, recipient_id, element):
        # type: (Text, Any) -> None
        """Sends a message to the recipient using the messenger client."""

        # this is a bit hacky, but the client doesn't have a proper API to
        # send messages but instead expects the incoming sender to be present
        # which we don't have as it is stored in the input channel.
        self.messenger_client.send(element.to_dict(),
                                   {"sender": {"id": recipient_id}},
                                   'RESPONSE')

    def send_text_message(self, recipient_id, message):
        # type: (Text, Text) -> None
        """Send a message through this channel."""

        logger.info("Sending message: " + message)

        for message_part in message.split("\n\n"):
            self.send(recipient_id, FBText(text=message_part))

    def send_image_url(self, recipient_id, image_url):
        # type: (Text, Text) -> None
        """Sends an image. Default will just post the url as a string."""

        self.send(recipient_id, attachments.Image(url=image_url))

    def send_text_with_buttons(self, recipient_id, text, buttons, **kwargs):
        # type: (Text, Text, List[Dict[Text, Any]], Any) -> None
        """Sends buttons to the output."""

        # buttons is a list of tuples: [(option_name,payload)]
        if len(buttons) > 3:
            logger.warning(
                    "Facebook API currently allows only up to 3 buttons. "
                    "If you add more, all will be ignored.")
            self.send_text_message(recipient_id, text)
        else:
            self._add_postback_info(buttons)

            # Currently there is no predefined way to create a message with
            # buttons in the fbmessenger framework - so we need to create the
            # payload on our own
            payload = {
                "attachment": {
                    "type": "template",
                    "payload": {
                        "template_type": "button",
                        "text": text,
                        "buttons": buttons
                    }
                }
            }
            self.messenger_client.send(payload,
                                       {"sender": {"id": recipient_id}},
                                       'RESPONSE')

    def send_custom_message(self, recipient_id, elements):
        # type: (Text, List[Dict[Text, Any]]) -> None
        """Sends elements to the output."""

        for element in elements:
            self._add_postback_info(element['buttons'])

        payload = {
            "attachment": {
                "type": "template",
                "payload": {
                    "template_type": "generic",
                    "elements": elements
                }
            }
        }
        self.messenger_client.send(payload,
                                   self._recipient_json(recipient_id),
                                   'RESPONSE')

    @staticmethod
    def _add_postback_info(buttons):
        # type: (List[Dict[Text, Any]]) -> None
        """Set the button type to postback for all buttons. Happens in place."""
        for button in buttons:
            button['type'] = "postback"

    @staticmethod
    def _recipient_json(recipient_id):
        # type: (Text) -> Dict[Text, Dict[Text, Text]]
        """Generate the response json for the recipient expected by FB."""
        return {"sender": {"id": recipient_id}}


class FacebookInput(InputChannel):
    """Facebook input channel implementation. Based on the HTTPInputChannel."""

    @classmethod
    def name(cls):
        return "facebook"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(credentials.get("verify"),
                   credentials.get("secret"),
                   credentials.get("page-access-token"))

    def __init__(self, fb_verify, fb_secret, fb_access_token):
        # type: (Text, Text, Text) -> None
        """Create a facebook input channel.

        Needs a couple of settings to properly authenticate and validate
        messages. Details to setup:

        https://github.com/rehabstudio/fbmessenger#facebook-app-setup
        :param fb_verify: FB Verification string
                          (can be chosen by yourself on webhook creation)
        :param fb_secret: facebook application secret
        :param fb_access_token: access token to post in the name of the FB page
        """
        self.fb_verify = fb_verify
        self.fb_secret = fb_secret
        self.fb_access_token = fb_access_token

    def blueprint(self, on_new_message):

        fb_webhook = Blueprint('fb_webhook', __name__)

        @fb_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @fb_webhook.route("/webhook", methods=['GET'])
        def token_verification():
            if request.args.get("hub.verify_token") == self.fb_verify:
                return request.args.get("hub.challenge")
            else:
                logger.warning(
                        "Invalid fb verify token! Make sure this matches "
                        "your webhook settings on the facebook app.")
                return "failure, invalid token"

        @fb_webhook.route("/webhook", methods=['POST'])
        def webhook():
            signature = request.headers.get("X-Hub-Signature") or ''
            if not self.validate_hub_signature(self.fb_secret, request.data,
                                               signature):
                logger.warning("Wrong fb secret! Make sure this matches the "
                               "secret in your facebook app settings")
                return "not validated"

            messenger = Messenger(self.fb_access_token, on_new_message)

            messenger.handle(request.get_json(force=True))
            return "success"

        return fb_webhook

    @staticmethod
    def validate_hub_signature(app_secret, request_payload,
                               hub_signature_header):
        """Makes sure the incoming webhook requests are properly signed.

        :param app_secret: Secret Key for application
        :param request_payload: request body
        :param hub_signature_header: X-Hub-Signature header sent with request
        :return: boolean indicated that hub signature is validated
        """

        # noinspection PyBroadException
        try:
            hash_method, hub_signature = hub_signature_header.split('=')
        except Exception:
            pass
        else:
            digest_module = getattr(hashlib, hash_method)
            if six.PY2:
                # noinspection PyCompatibility,PyUnresolvedReferences
                hmac_object = hmac.new(
                        str(app_secret),
                        unicode(request_payload), digest_module)
            else:
                hmac_object = hmac.new(
                        bytearray(app_secret, 'utf8'),
                        request_payload, digest_module)
            generated_hash = hmac_object.hexdigest()
            if hub_signature == generated_hash:
                return True
        return False
