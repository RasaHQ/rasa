import json
import logging
import typing
from copy import deepcopy
from typing import Any, Awaitable, Callable, Dict, List, Optional, Text

# Import aiogram at module level to raise error if not installed
from aiogram import Bot
from aiogram.types import Message, Update
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse

from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.shared.core.constants import USER_INTENT_RESTART
from rasa.shared.exceptions import RasaException

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from aiogram.types import Message, Update


class TelegramOutput(OutputChannel):
    """Output channel for Telegram."""

    # skipcq: PYL-W0236
    @classmethod
    def name(cls) -> Text:
        return "telegram"

    def __init__(self, access_token: Optional[Text]) -> None:
        self.bot = Bot(access_token)

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Sends text message."""
        for message_part in text.strip().split("\n\n"):
            await self.bot.send_message(recipient_id, message_part)

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """Sends an image."""
        await self.bot.send_photo(recipient_id, image)

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        button_type: Optional[Text] = "inline",
        **kwargs: Any,
    ) -> None:
        """Sends a message with keyboard.

        For more information: https://core.telegram.org/bots#keyboards

        :button_type inline: horizontal inline keyboard

        :button_type vertical: vertical inline keyboard

        :button_type reply: reply keyboard
        """
        from aiogram.types import InlineKeyboardButton, KeyboardButton
        from aiogram.utils.keyboard import (
            InlineKeyboardBuilder,
            KeyboardBuilder,
            ReplyKeyboardBuilder,
        )

        if button_type == "inline":
            reply_markup_builder: "KeyboardBuilder" = InlineKeyboardBuilder()
            button_list = [
                InlineKeyboardButton(text=s["title"], callback_data=s["payload"])
                for s in buttons
            ]
            reply_markup_builder.row(*button_list)
            reply_markup = reply_markup_builder.as_markup()

        elif button_type == "vertical":
            reply_markup_builder = InlineKeyboardBuilder()
            [
                reply_markup_builder.row(
                    InlineKeyboardButton(text=s["title"], callback_data=s["payload"])
                )
                for s in buttons
            ]
            reply_markup = reply_markup_builder.as_markup()

        elif button_type == "reply":
            reply_markup_builder = ReplyKeyboardBuilder()

            for idx, button in enumerate(buttons):
                if isinstance(button, list):
                    reply_markup_builder.add(
                        *[KeyboardButton(text=s["title"]) for s in button]
                    )
                else:
                    reply_markup_builder.add(KeyboardButton(text=button["title"]))
            reply_markup = reply_markup_builder.as_markup(
                resize_keyboard=False, one_time_keyboard=True
            )
        else:
            logger.error(
                "Trying to send text with buttons for unknown button type {}".format(
                    button_type
                )
            )
            return

        await self.bot.send_message(recipient_id, text, reply_markup=reply_markup)

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """Sends a message with a custom json payload."""
        json_message = deepcopy(json_message)

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
                api_call = getattr(self.bot, send_functions[params])
                await api_call(recipient_id, *args, **json_message)

    async def get_me(self) -> Any:
        """Get information about the bot itself."""
        return await self.bot.get_me()

    async def set_webhook(self, url: Text) -> None:
        """Set the webhook URL for telegram."""
        await self.bot.set_webhook(url=url)


class TelegramInput(InputChannel):
    """Telegram input channel."""

    @classmethod
    def name(cls) -> Text:
        return "telegram"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(
            credentials.get("access_token"),
            credentials.get("verify"),
            credentials.get("webhook_url"),
        )

    def __init__(
        self,
        access_token: Optional[Text],
        verify: Optional[Text],
        webhook_url: Optional[Text],
        debug_mode: bool = True,
    ) -> None:
        self.access_token = access_token
        self.verify = verify
        self.webhook_url = webhook_url
        self.debug_mode = debug_mode

    @staticmethod
    def _is_location(message: "Message") -> bool:
        return message.location is not None

    @staticmethod
    def _is_user_message(message: "Message") -> bool:
        return message.text is not None

    @staticmethod
    def _is_edited_message(message: "Update") -> bool:
        return message.edited_message is not None

    @staticmethod
    def _is_button(message: "Update") -> bool:
        return message.callback_query is not None

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        telegram_webhook = Blueprint("telegram_webhook", __name__)
        out_channel = self.get_output_channel()

        @telegram_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @telegram_webhook.route("/set_webhook", methods=["GET", "POST"])
        async def set_webhook(_: Request) -> HTTPResponse:
            try:
                await self.set_webhook(out_channel)
            except RasaException as exc:
                logger.error(exc)
                return response.text(str(exc))

            logger.info("Webhook Setup Successful")
            return response.text("Webhook setup successful")

        @telegram_webhook.route("/webhook", methods=["GET", "POST"])
        async def message(request: Request) -> Any:
            from aiogram.types import Update

            if request.method == "POST":
                request_dict = request.json
                if isinstance(request_dict, Text):
                    request_dict = json.loads(request_dict)
                update = Update(**request_dict)
                credentials = await out_channel.get_me()
                if not credentials.username == self.verify:
                    logger.debug("Invalid access token, check it matches Telegram")
                    return response.text("failed")

                if self._is_button(update):
                    msg = (
                        update.callback_query.message
                        if update.callback_query is not None
                        else None
                    )
                    text = (
                        update.callback_query.data
                        if update.callback_query is not None
                        else None
                    )
                elif self._is_edited_message(update):
                    msg = update.edited_message
                    text = (
                        update.edited_message.text
                        if update.edited_message is not None
                        else None
                    )
                else:
                    msg = update.message
                    if self._is_user_message(msg):
                        text = (
                            msg.text.replace("/bot", "")
                            if msg is not None and msg.text is not None
                            else None
                        )
                    elif self._is_location(msg):
                        text = (
                            '{{"lng":{0}, "lat":{1}}}'.format(
                                msg.location.longitude, msg.location.latitude
                            )
                            if msg is not None and msg.location is not None
                            else None
                        )
                    else:
                        return response.text("success")

                metadata = self.get_metadata(request)
                try:
                    if msg is not None:
                        sender_id = str(msg.chat.id)
                    else:
                        raise ValueError("No message found.")

                    if text == (INTENT_MESSAGE_PREFIX + USER_INTENT_RESTART):
                        await on_new_message(
                            UserMessage(
                                text,
                                out_channel,
                                sender_id,
                                input_channel=self.name(),
                                metadata=metadata,
                            )
                        )
                        await on_new_message(
                            UserMessage(
                                "/start",
                                out_channel,
                                sender_id,
                                input_channel=self.name(),
                                metadata=metadata,
                            )
                        )
                    else:
                        await on_new_message(
                            UserMessage(
                                text,
                                out_channel,
                                sender_id,
                                input_channel=self.name(),
                                metadata=metadata,
                            )
                        )
                except Exception as e:
                    logger.error(f"Exception when trying to handle message.{e}")
                    logger.debug(e, exc_info=True)
                    if self.debug_mode:
                        raise
                    pass

                return response.text("success")

        return telegram_webhook

    def get_output_channel(self) -> TelegramOutput:
        """Loads the telegram channel."""
        return TelegramOutput(self.access_token)

    async def set_webhook(self, channel: TelegramOutput) -> None:
        from aiogram.exceptions import TelegramAPIError

        try:
            await channel.set_webhook(url=self.webhook_url)
        except TelegramAPIError as error:
            raise RasaException(
                "Failed to set channel webhook: " + str(error)
            ) from error
