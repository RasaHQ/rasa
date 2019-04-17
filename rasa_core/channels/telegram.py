import logging

from flask import Blueprint, request, jsonify
from telegram import (
    Bot, InlineKeyboardButton, Update, InlineKeyboardMarkup,
    KeyboardButton, ReplyKeyboardMarkup)

from rasa_core import constants
from rasa_core.channels import InputChannel
from rasa_core.channels.channel import UserMessage, OutputChannel
from rasa_core.constants import INTENT_MESSAGE_PREFIX, USER_INTENT_RESTART

logger = logging.getLogger(__name__)


class TelegramOutput(Bot, OutputChannel):
    """Output channel for Telegram"""

    @classmethod
    def name(cls):
        return "telegram"

    def __init__(self, access_token):
        super(TelegramOutput, self).__init__(access_token)

    def send_text_message(self, recipient_id, message):
        for message_part in message.split("\n\n"):
            self.send_message(recipient_id, message_part)

    def send_image_url(self, recipient_id, image_url):
        self.send_photo(recipient_id, image_url)

    def send_text_with_buttons(self, recipient_id, text,
                               buttons, button_type="inline", **kwargs):
        """Sends a message with keyboard.

        For more information: https://core.telegram.org/bots#keyboards

        :button_type inline: horizontal inline keyboard

        :button_type vertical: vertical inline keyboard

        :button_type custom: custom keyboard
        """
        if button_type == "inline":
            button_list = [[InlineKeyboardButton(s["title"],
                                                 callback_data=s["payload"])
                            for s in buttons]]
            reply_markup = InlineKeyboardMarkup(button_list)

        elif button_type == "vertical":
            button_list = [[InlineKeyboardButton(s["title"],
                                                 callback_data=s["payload"])]
                           for s in buttons]
            reply_markup = InlineKeyboardMarkup(button_list)

        elif button_type == "custom":
            button_list = []
            for bttn in buttons:
                if isinstance(bttn, list):
                    button_list.append([KeyboardButton(s['title'])
                                        for s in bttn])
                else:
                    button_list.append([KeyboardButton(bttn["title"])])
            reply_markup = ReplyKeyboardMarkup(button_list,
                                               resize_keyboard=True,
                                               one_time_keyboard=True)
        else:
            logger.error('Trying to send text with buttons for unknown '
                         'button type {}'.format(button_type))
            return

        self.send_message(recipient_id, text, reply_markup=reply_markup)


class TelegramInput(InputChannel):
    """Telegram input channel"""

    @classmethod
    def name(cls):
        return "telegram"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(credentials.get("access_token"),
                   credentials.get("verify"),
                   credentials.get("webhook_url"))

    def __init__(self, access_token, verify, webhook_url, debug_mode=True):
        self.access_token = access_token
        self.verify = verify
        self.webhook_url = webhook_url
        self.debug_mode = debug_mode

    @staticmethod
    def _is_location(message):
        return message.location

    @staticmethod
    def _is_user_message(message):
        return message.text

    @staticmethod
    def _is_button(update):
        return update.callback_query

    def blueprint(self, on_new_message):
        telegram_webhook = Blueprint('telegram_webhook', __name__)
        out_channel = TelegramOutput(self.access_token)

        @telegram_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @telegram_webhook.route("/set_webhook", methods=['GET', 'POST'])
        def set_webhook():
            s = out_channel.setWebhook(self.webhook_url)
            if s:
                logger.info("Webhook Setup Successful")
                return "Webhook setup successful"
            else:
                logger.warning("Webhook Setup Failed")
                return "Invalid webhook"

        @telegram_webhook.route("/webhook", methods=['GET', 'POST'])
        def message():
            if request.method == 'POST':

                if not out_channel.get_me()['username'] == self.verify:
                    logger.debug("Invalid access token, check it "
                                 "matches Telegram")
                    return "failed"

                update = Update.de_json(request.get_json(force=True),
                                        out_channel)
                if self._is_button(update):
                    msg = update.callback_query.message
                    text = update.callback_query.data
                else:
                    msg = update.message
                    if self._is_user_message(msg):
                        text = msg.text.replace('/bot', '')
                    elif self._is_location(msg):
                        text = ('{{"lng":{0}, "lat":{1}}}'
                                ''.format(msg.location.longitude,
                                          msg.location.latitude))
                    else:
                        return "success"
                sender_id = msg.chat.id
                try:
                    if text == (INTENT_MESSAGE_PREFIX + USER_INTENT_RESTART):

                        on_new_message(UserMessage(
                            text, out_channel, sender_id,
                            input_channel=self.name()))
                        on_new_message(UserMessage(
                            '/start', out_channel, sender_id,
                            input_channel=self.name()))
                    else:
                        on_new_message(UserMessage(
                            text, out_channel, sender_id,
                            input_channel=self.name()))
                except Exception as e:
                    logger.error("Exception when trying to handle "
                                 "message.{0}".format(e))
                    logger.debug(e, exc_info=True)
                    if self.debug_mode:
                        raise
                    pass

                return "success"

        set_webhook()
        return telegram_webhook
