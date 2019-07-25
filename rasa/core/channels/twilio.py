# -*- coding: utf-8 -*-
import logging
from sanic import Blueprint, response
from sanic.request import Request
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client
from typing import Dict, Text, Any

from rasa.core.channels.channel import InputChannel
from rasa.core.channels.channel import UserMessage, OutputChannel

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

    async def _send_message(self, message_data: Dict[Text, Any]):
        message = None
        try:
            while not message and self.send_retry < self.max_retry:
                message = self.messages.create(**message_data)
                self.send_retry += 1
        except TwilioRestException as e:
            logger.error("Something went wrong " + repr(e.msg))
        finally:
            self.send_retry = 0

        if not message and self.send_retry == self.max_retry:
            logger.error("Failed to send message. Max number of retires exceeded.")

        return message

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Sends text message"""

        message_data = {"to": recipient_id, "from_": self.twilio_number}
        for message_part in text.split("\n\n"):
            message_data.update({"body": message_part})
            await self._send_message(message_data)

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """Send custom json dict"""

        json_message.setdefault("to", recipient_id)
        if not json_message.get("media_url"):
            json_message.setdefault("body", "")
        if not json_message.get("messaging_service_sid"):
            json_message.setdefault("from", self.twilio_number)

        await self._send_message(json_message)


class TwilioInput(InputChannel):
    """Twilio input channel"""

    @classmethod
    def name(cls):
        return "twilio"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(
            credentials.get("account_sid"),
            credentials.get("auth_token"),
            credentials.get("twilio_number"),
        )

    def __init__(self, account_sid, auth_token, twilio_number, debug_mode=True):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.twilio_number = twilio_number
        self.debug_mode = debug_mode

    def blueprint(self, on_new_message):
        twilio_webhook = Blueprint("twilio_webhook", __name__)

        @twilio_webhook.route("/", methods=["GET"])
        async def health(request: Request):
            return response.json({"status": "ok"})

        @twilio_webhook.route("/webhook", methods=["POST"])
        async def message(request: Request):
            sender = request.form.get("From", None)
            text = request.form.get("Body", None)

            out_channel = TwilioOutput(
                self.account_sid, self.auth_token, self.twilio_number
            )

            if sender is not None and message is not None:
                try:
                    # @ signs get corrupted in SMSes by some carriers
                    text = text.replace("¡", "@")
                    await on_new_message(
                        UserMessage(
                            text, out_channel, sender, input_channel=self.name()
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
            else:
                logger.debug("Invalid message")

            return response.text("success")

        return twilio_webhook
