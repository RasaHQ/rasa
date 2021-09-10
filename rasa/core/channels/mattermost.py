import json

import logging
import requests
from requests import Response
from sanic import Blueprint, response
from sanic.request import Request
from typing import Text, Dict, Any, List, Callable, Awaitable, Optional

from rasa.core.channels.channel import UserMessage, OutputChannel, InputChannel
from sanic.response import HTTPResponse

logger = logging.getLogger(__name__)


class MattermostBot(OutputChannel):
    """A Mattermost communication channel"""

    @classmethod
    def name(cls) -> Text:
        return "mattermost"

    @classmethod
    def token_from_login(cls, url: Text, user: Text, password: Text) -> Optional[Text]:
        """Retrieve access token for mattermost user."""

        data = {"login_id": user, "password": password}
        r = requests.post(url + "/users/login", data=json.dumps(data))
        if r.status_code == 200:
            return r.headers["Token"]
        else:
            logger.error(f"Failed to login mattermost user {user}. Response: {r}")
            return None

    def __init__(
        self, url: Text, token: Text, bot_channel: Text, webhook_url: Optional[Text]
    ) -> None:
        self.url = url
        self.token = token
        self.bot_channel = bot_channel
        self.webhook_url = webhook_url

        super(MattermostBot, self).__init__()

    def _post_message_to_channel(self, channel_id: Text, message: Text) -> Response:
        return self._post_data_to_channel(
            {"channel_id": channel_id, "message": message}
        )

    def _post_data_to_channel(self, data: Dict[Text, Any]) -> Response:
        """Send a message to a mattermost channel."""

        headers = {"Authorization": "Bearer " + self.token}
        r = requests.post(self.url + "/posts", headers=headers, data=json.dumps(data))
        if not r.status_code == 200:
            logger.error(
                f"Failed to send message to mattermost channel "
                f"{data.get('channel_id')}. Response: {r}"
            )
        return r

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        for message_part in text.strip().split("\n\n"):
            self._post_message_to_channel(self.bot_channel, message_part)

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        json_message.setdefault("channel_id", self.bot_channel)
        json_message.setdefault("message", "")

        self._post_data_to_channel(json_message)

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """Sends an image."""

        self._post_data_to_channel(
            {
                "channel_id": self.bot_channel,
                "props": {"attachments": [{"image_url": image}]},
            }
        )

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """Sends buttons to the output."""
        # buttons are a list of objects: [(option_name, payload)]
        # See https://docs.mattermost.com/developer/interactive-messages.html#message-buttons # noqa: E501, W505

        actions = [
            {
                "name": button["title"],
                "integration": {
                    "url": self.webhook_url,
                    "context": {"action": button["payload"]},
                },
            }
            for button in buttons
        ]

        props = {"attachments": [{"actions": actions}]}

        self._post_data_to_channel(
            {"channel_id": self.bot_channel, "message": text, "props": props}
        )


class MattermostInput(InputChannel):
    """Mattermost input channel implemenation."""

    @classmethod
    def name(cls) -> Text:
        return "mattermost"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if credentials is None:
            cls.raise_missing_credentials_exception()

        token = credentials.get("token")

        return cls(credentials.get("url"), token, credentials.get("webhook_url"))

    def __init__(self, url: Text, token: Text, webhook_url: Text) -> None:
        """Create a Mattermost input channel.
        Needs a couple of settings to properly authenticate and validate
        messages.

        Args:
            url: Your Mattermost team url including /v4 example
                https://mysite.example.com/api/v4
            token: Your mattermost bot token
            webhook_url: The mattermost callback url as specified
                in the outgoing webhooks in mattermost example
                https://mysite.example.com/webhooks/mattermost/webhook
        """
        self.url = url
        self.token = token
        self.webhook_url = webhook_url

    async def message_with_trigger_word(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[None]],
        output: Dict[Text, Any],
        metadata: Optional[Dict],
    ) -> None:
        # splitting to get rid of the @botmention
        # trigger we are using for this
        split_message = output["text"].split(" ", 1)
        if len(split_message) >= 2:
            message = split_message[1]
        else:
            message = output["text"]

        await self._handle_message(
            message, output["user_id"], output["channel_id"], metadata, on_new_message
        )

    async def action_from_button(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[None]],
        output: Dict[Text, Any],
        metadata: Optional[Dict],
    ) -> None:
        # get the action, the buttons triggers
        action = output["context"]["action"]

        await self._handle_message(
            action, output["user_id"], output["channel_id"], metadata, on_new_message
        )

    async def _handle_message(
        self,
        message: Text,
        sender_id: Text,
        bot_channel: Text,
        metadata: Optional[Dict],
        on_new_message: Callable[[UserMessage], Awaitable[None]],
    ) -> None:
        try:
            out_channel = MattermostBot(
                self.url, self.token, bot_channel, self.webhook_url
            )
            user_msg = UserMessage(
                message,
                out_channel,
                sender_id,
                input_channel=self.name(),
                metadata=metadata,
            )
            await on_new_message(user_msg)
        except Exception as e:
            logger.error(f"Exception when trying to handle message.{e}")
            logger.debug(e, exc_info=True)

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Blueprint:
        mattermost_webhook = Blueprint("mattermost_webhook", __name__)

        @mattermost_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @mattermost_webhook.route("/webhook", methods=["POST"])
        async def webhook(request: Request) -> HTTPResponse:
            output = request.json

            if not output:
                return response.text("")

            metadata = self.get_metadata(request)
            # handle normal message with trigger_word
            if "trigger_word" in output:
                await self.message_with_trigger_word(on_new_message, output, metadata)

            # handle context actions from buttons
            elif "context" in output:
                await self.action_from_button(on_new_message, output, metadata)

            return response.text("success")

        return mattermost_webhook
