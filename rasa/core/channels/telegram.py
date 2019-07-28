import logging
from sanic import Blueprint, response
from sanic.request import Request
from telegram import (
    Bot,
    InlineKeyboardButton,
    Update,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)
from typing import Dict, Text, Any, List, Optional

from rasa.core.channels.channel import InputChannel, UserMessage, OutputChannel
from rasa.core.constants import INTENT_MESSAGE_PREFIX, USER_INTENT_RESTART

logger = logging.getLogger(__name__)


class TelegramOutput(Bot, OutputChannel):
    """Output channel for Telegram"""

    @classmethod
    def name(cls):
        return "telegram"

    def __init__(self, access_token):
        super(TelegramOutput, self).__init__(access_token)

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        for message_part in text.split("\n\n"):
            self.send_message(recipient_id, message_part)

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        self.send_photo(recipient_id, image)

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        button_type: Optional[Text] = "inline",
        **kwargs: Any
    ) -> None:
        """Sends a message with keyboard.

        For more information: https://core.telegram.org/bots#keyboards

        :button_type inline: horizontal inline keyboard

        :button_type vertical: vertical inline keyboard

        :button_type reply: reply keyboard
        """
        if button_type == "inline":
            button_list = [
                [
                    InlineKeyboardButton(s["title"], callback_data=s["payload"])
                    for s in buttons
                ]
            ]
            reply_markup = InlineKeyboardMarkup(button_list)

        elif button_type == "vertical":
            button_list = [
                [InlineKeyboardButton(s["title"], callback_data=s["payload"])]
                for s in buttons
            ]
            reply_markup = InlineKeyboardMarkup(button_list)

        elif button_type == "reply":
            button_list = []
            for bttn in buttons:
                if isinstance(bttn, list):
                    button_list.append([KeyboardButton(s["title"]) for s in bttn])
                else:
                    button_list.append([KeyboardButton(bttn["title"])])
            reply_markup = ReplyKeyboardMarkup(
                button_list, resize_keyboard=True, one_time_keyboard=True
            )
        else:
            logger.error(
                "Trying to send text with buttons for unknown "
                "button type {}".format(button_type)
            )
            return

        self.send_message(recipient_id, text, reply_markup=reply_markup)

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        recipient_id = json_message.pop("chat_id", recipient_id)

        send_functions = {
            ("text",): "send_message",
            ("photo",): "send_photo",
            ("audio",): "send_audio",
            ("document",): "send_document",
            ("sticker",): "send_sticker",
            ("video",): "send_video",
            ("video_note",): "send_video_note",
            ("animation",): "send_animation",
            ("voice",): "send_voice",
            ("media",): "send_media_group",
            ("latitude", "longitude", "title", "address"): "send_venue",
            ("latitude", "longitude"): "send_location",
            ("phone_number", "first_name"): "send_contact",
            ("game_short_name",): "send_game",
            ("action",): "send_chat_action",
            (
                "title",
                "decription",
                "payload",
                "provider_token",
                "start_parameter",
                "currency",
                "prices",
            ): "send_invoice",
        }

        for params in send_functions.keys():
            if all(json_message.get(p) is not None for p in params):
                args = [json_message.pop(p) for p in params]
                api_call = getattr(self, send_functions[params])
                api_call(recipient_id, *args, **json_message)


class TelegramInput(InputChannel):
    """Telegram input channel"""

    @classmethod
    def name(cls):
        return "telegram"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(
            credentials.get("access_token"),
            credentials.get("verify"),
            credentials.get("webhook_url"),
        )

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
        telegram_webhook = Blueprint("telegram_webhook", __name__)
        out_channel = TelegramOutput(self.access_token)

        @telegram_webhook.route("/", methods=["GET"])
        async def health(request: Request):
            return response.json({"status": "ok"})

        @telegram_webhook.route("/set_webhook", methods=["GET", "POST"])
        async def set_webhook(request: Request):
            s = out_channel.setWebhook(self.webhook_url)
            if s:
                logger.info("Webhook Setup Successful")
                return response.text("Webhook setup successful")
            else:
                logger.warning("Webhook Setup Failed")
                return response.text("Invalid webhook")

        @telegram_webhook.route("/webhook", methods=["GET", "POST"])
        async def message(request: Request):
            if request.method == "POST":

                if not out_channel.get_me()["username"] == self.verify:
                    logger.debug("Invalid access token, check it matches Telegram")
                    return response.text("failed")

                update = Update.de_json(request.json, out_channel)
                if self._is_button(update):
                    msg = update.callback_query.message
                    text = update.callback_query.data
                else:
                    msg = update.message
                    if self._is_user_message(msg):
                        text = msg.text.replace("/bot", "")
                    elif self._is_location(msg):
                        text = '{{"lng":{0}, "lat":{1}}}'.format(
                            msg.location.longitude, msg.location.latitude
                        )
                    else:
                        return response.text("success")
                sender_id = msg.chat.id
                try:
                    if text == (INTENT_MESSAGE_PREFIX + USER_INTENT_RESTART):
                        await on_new_message(
                            UserMessage(
                                text, out_channel, sender_id, input_channel=self.name()
                            )
                        )
                        await on_new_message(
                            UserMessage(
                                "/start",
                                out_channel,
                                sender_id,
                                input_channel=self.name(),
                            )
                        )
                    else:
                        await on_new_message(
                            UserMessage(
                                text, out_channel, sender_id, input_channel=self.name()
                            )
                        )
                except Exception as e:
                    logger.error(
                        "Exception when trying to handle message.{0}".format(e)
                    )
                    logger.debug(e, exc_info=True)
                    if self.debug_mode:
                        raise
                    pass

                return response.text("success")

        out_channel.setWebhook(self.webhook_url)
        return telegram_webhook
