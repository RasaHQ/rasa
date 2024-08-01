from typing import Any, Awaitable, Callable, Dict, Optional, Text

import structlog
from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage
from rasa.core.channels.voice_aware.jambonz_protocol import (
    send_ws_text_message,
    websocket_message_handler,
)
from rasa.core.channels.voice_aware.utils import validate_voice_license_scope
from rasa.shared.exceptions import RasaException
from sanic import Blueprint, response, Websocket  # type: ignore[attr-defined]
from sanic.request import Request
from sanic.response import HTTPResponse

from rasa.shared.utils.common import mark_as_experimental_feature


structlogger = structlog.get_logger()

CHANNEL_NAME = "jambonz"


class JambonzVoiceAwareInput(InputChannel):
    """Connector for the Jambonz platform."""

    @classmethod
    def name(cls) -> Text:
        return CHANNEL_NAME

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        return cls()

    def __init__(self) -> None:
        """Initializes the JambonzVoiceAwareInput channel."""
        mark_as_experimental_feature("Jambonz Channel")
        validate_voice_license_scope()

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        jambonz_webhook = Blueprint("jambonz_webhook", __name__)

        @jambonz_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            """Server health route."""
            return response.json({"status": "ok"})

        @jambonz_webhook.websocket("/websocket", subprotocols=["ws.jambonz.org"])  # type: ignore
        async def websocket(request: Request, ws: Websocket) -> None:
            """Triggered on new websocket connection."""
            async for message in ws:
                await websocket_message_handler(message, on_new_message, ws)

        return jambonz_webhook


class JambonzWebsocketOutput(OutputChannel):
    @classmethod
    def name(cls) -> Text:
        return CHANNEL_NAME

    def __init__(self, ws: Any, conversation_id: Text) -> None:
        self.ws = ws
        self.conversation_id = conversation_id

    async def add_message(self, message: Dict) -> None:
        """Add metadata and add message.

        Message is added to the list of
        activities to be sent to the Jambonz Websocket server.
        """
        text_message = message.get("text", "")
        structlogger.debug(
            "jambonz.add.message",
            class_name=self.__class__.__name__,
            message=text_message,
        )

        # send message to jambonz
        await send_ws_text_message(self.ws, message.get("text"))

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Send a text message."""
        await self.add_message({"type": "message", "text": text})

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        raise RasaException("Images are not supported by this channel")

    async def send_attachment(
        self, recipient_id: Text, attachment: Text, **kwargs: Any
    ) -> None:
        raise RasaException("Attachments are not supported by this channel")

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """Send an activity."""
        await self.add_message(json_message)
