import logging
from sanic import Blueprint, response
from sanic.request import Request
from typing import Text, Optional, Dict, Any, Callable, Awaitable

from sanic.response import HTTPResponse
from webexteamssdk import WebexTeamsAPI, Webhook

from rasa.core.channels.channel import InputChannel
from rasa.core.channels.channel import UserMessage, OutputChannel

logger = logging.getLogger(__name__)


class WebexTeamsBot(OutputChannel):
    """A Cisco WebexTeams communication channel."""

    @classmethod
    def name(cls) -> Text:
        return "webexteams"

    def __init__(self, access_token: Optional[Text], room: Optional[Text]) -> None:
        self.room = room
        self.api = WebexTeamsAPI(access_token)

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        recipient = self.room or recipient_id
        for message_part in text.split("\n\n"):
            self.api.messages.create(roomId=recipient, text=message_part)

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        recipient = self.room or recipient_id
        return self.api.messages.create(roomId=recipient, files=[image])

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        json_message.setdefault("roomID", recipient_id)
        return self.api.messages.create(**json_message)


class WebexTeamsInput(InputChannel):
    """WebexTeams input channel. Based on the HTTPInputChannel."""

    @classmethod
    def name(cls) -> Text:
        return "webexteams"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        # pytype: disable=attribute-error
        return cls(credentials.get("access_token"), credentials.get("room"))
        # pytype: enable=attribute-error

    def __init__(self, access_token: Text, room: Optional[Text] = None) -> None:
        """Create a Cisco Webex Teams input channel.

        Needs a couple of settings to properly authenticate and validate
        messages. Details here https://developer.webex.com/authentication.html

        Args:
            access_token: Cisco WebexTeams bot access token.
            room: the string identifier for a room to which the bot posts
        """
        self.token = access_token
        self.room = room
        self.api = WebexTeamsAPI(access_token)

    async def process_message(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        text: Optional[Text],
        sender_id: Optional[Text],
        metadata: Optional[Dict],
    ) -> Any:

        try:
            out_channel = self.get_output_channel()
            user_msg = UserMessage(
                text,
                out_channel,
                sender_id,
                input_channel=self.name(),
                metadata=metadata,
            )
            await on_new_message(user_msg)
        except Exception as e:
            logger.error(f"Exception when trying to handle message.{e}")
            logger.error(str(e), exc_info=True)

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        webexteams_webhook = Blueprint("webexteams_webhook", __name__)

        @webexteams_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @webexteams_webhook.route("/webhook", methods=["POST"])
        async def webhook(request: Request) -> HTTPResponse:
            """Respond to inbound webhook HTTP POST from Webex Teams."""

            logger.debug("Received webex webhook call")
            # Get the POST data sent from Webex Teams
            json_data = request.json

            # Create a Webhook object from the JSON data
            webhook_obj = Webhook(json_data)
            # Get the message details
            message = self.api.messages.get(webhook_obj.data.id)

            # This is a VERY IMPORTANT loop prevention control step.
            # If you respond to all messages...  You will respond to the
            # messages that the bot posts and thereby create a loop
            me = self.api.people.me()
            if message.personId == me.id:
                # Message was sent by me (bot); do not respond.
                return response.text("OK")

            else:
                metadata = self.get_metadata(request)
                await self.process_message(
                    on_new_message, message.text, message.personId, metadata
                )
                return response.text("")

        return webexteams_webhook

    def get_output_channel(self) -> OutputChannel:
        return WebexTeamsBot(self.token, self.room)
