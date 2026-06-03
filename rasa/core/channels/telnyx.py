import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Text

import aiohttp
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse

from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage

logger = logging.getLogger(__name__)

TELNYX_MESSAGES_URL = "https://api.telnyx.com/v2/messages"


class TelnyxOutput(OutputChannel):
    """Output channel for Telnyx Messaging."""

    @classmethod
    def name(cls) -> Text:
        """Name of the channel."""
        return "telnyx"

    def __init__(self, api_key: Text, from_number: Text) -> None:
        """Create a Telnyx Messaging output channel."""
        self.api_key = api_key
        self.from_number = from_number

    async def _send_message(self, message_data: Dict[Text, Any]) -> None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                TELNYX_MESSAGES_URL, headers=headers, json=message_data
            ) as resp:
                if resp.status >= 400:
                    logger.error(
                        "Failed to send Telnyx message. Status: %s. Response: %s",
                        resp.status,
                        await resp.text(),
                    )

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Sends text messages."""
        for message_part in text.strip().split("\n\n"):
            await self._send_message(
                {
                    "from": self.from_number,
                    "to": recipient_id,
                    "text": message_part,
                }
            )

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """Sends an image."""
        await self._send_message(
            {
                "from": self.from_number,
                "to": recipient_id,
                "text": kwargs.get("text", ""),
                "media_urls": [image],
            }
        )

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """Sends a custom Telnyx message payload."""
        json_message.setdefault("from", self.from_number)
        json_message.setdefault("to", recipient_id)

        await self._send_message(json_message)


class TelnyxInput(InputChannel):
    """Telnyx Messaging input channel."""

    @classmethod
    def name(cls) -> Text:
        """Name of the channel."""
        return "telnyx"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        """Load Telnyx credentials from the credentials file."""
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(credentials.get("api_key"), credentials.get("from_number"))

    def __init__(self, api_key: Text, from_number: Text) -> None:
        """Create a Telnyx Messaging input channel."""
        self.api_key = api_key
        self.from_number = from_number

    @staticmethod
    def _message_payload(request: Request) -> Dict[Text, Any]:
        return (request.json or {}).get("data", {}).get("payload", {})

    @staticmethod
    def _event_type(request: Request) -> Optional[Text]:
        return (request.json or {}).get("data", {}).get("event_type")

    @staticmethod
    def _sender_id(payload: Dict[Text, Any]) -> Optional[Text]:
        return payload.get("from", {}).get("phone_number")

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """Defines the Telnyx webhook routes."""
        telnyx_webhook = Blueprint("telnyx_webhook", __name__)

        @telnyx_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @telnyx_webhook.route("/webhook", methods=["POST"])
        async def message(request: Request) -> HTTPResponse:
            if self._event_type(request) != "message.received":
                return response.text("", status=204)

            payload = self._message_payload(request)
            sender_id = self._sender_id(payload)
            text = payload.get("text")

            if sender_id and text:
                metadata = self.get_metadata(request)
                await on_new_message(
                    UserMessage(
                        text,
                        self.get_output_channel(),
                        sender_id,
                        input_channel=self.name(),
                        metadata=metadata,
                        message_id=payload.get("id"),
                    )
                )
            else:
                logger.debug("Invalid Telnyx message webhook")

            return response.text("", status=204)

        return telnyx_webhook

    def get_metadata(self, request: Request) -> Optional[Dict[Text, Any]]:
        """Extract the Telnyx message payload as metadata."""
        return self._message_payload(request)

    def get_output_channel(self) -> OutputChannel:
        """Create the Telnyx output channel."""
        return TelnyxOutput(self.api_key, self.from_number)
