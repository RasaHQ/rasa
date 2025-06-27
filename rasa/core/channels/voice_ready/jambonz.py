from typing import Any, Awaitable, Callable, Dict, List, Optional, Text

import structlog
from sanic import Blueprint, Websocket, response  # type: ignore[attr-defined]
from sanic.request import Request
from sanic.response import HTTPResponse

from rasa.core.channels.channel import (
    InputChannel,
    OutputChannel,
    UserMessage,
    requires_basic_auth,
)
from rasa.core.channels.voice_ready.jambonz_protocol import (
    CHANNEL_NAME,
    send_ws_hangup_message,
    send_ws_text_message,
    websocket_message_handler,
)
from rasa.core.channels.voice_ready.utils import (
    validate_username_password_credentials,
    validate_voice_license_scope,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.common import mark_as_beta_feature
from rasa.utils.io import remove_emojis

structlogger = structlog.get_logger()

DEFAULT_HANGUP_DELAY_SECONDS = 1


class JambonzVoiceReadyInput(InputChannel):
    """Connector for the Jambonz platform."""

    @classmethod
    def name(cls) -> Text:
        return CHANNEL_NAME

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            return cls()

        username = credentials.get("username")
        password = credentials.get("password")
        validate_username_password_credentials(username, password, "Jambonz")

        return cls(username, password)

    def __init__(
        self, username: Optional[Text] = None, password: Optional[Text] = None
    ) -> None:
        """Initializes the JambonzVoiceReadyInput channel."""
        mark_as_beta_feature("Jambonz Channel")
        validate_voice_license_scope()
        self.username = username
        self.password = password

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        jambonz_webhook = Blueprint("jambonz_webhook", __name__)

        @jambonz_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            """Server health route."""
            return response.json({"status": "ok"})

        @jambonz_webhook.websocket("/websocket", subprotocols=["ws.jambonz.org"])  # type: ignore
        @requires_basic_auth(self.username, self.password)
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
        text = remove_emojis(text)
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

    async def hangup(self, recipient_id: Text, **kwargs: Any) -> None:
        """Indicate that the conversation should be ended."""
        await send_ws_hangup_message(DEFAULT_HANGUP_DELAY_SECONDS, self.ws)

    async def send_text_with_buttons(
        self,
        recipient_id: str,
        text: str,
        buttons: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Uses the concise button output format for voice channels."""
        await self.send_text_with_buttons_concise(recipient_id, text, buttons, **kwargs)
