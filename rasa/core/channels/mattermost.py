import logging
from mattermostwrapper import MattermostAPI
from sanic import Blueprint, response
from sanic.request import Request
from typing import Text, Dict, Any

from rasa.core.channels.channel import UserMessage, OutputChannel, InputChannel

logger = logging.getLogger(__name__)


class MattermostBot(MattermostAPI, OutputChannel):
    """A Mattermost communication channel"""

    @classmethod
    def name(cls):
        return "mattermost"

    def __init__(self, url, team, user, pw, bot_channel):
        self.url = url
        self.team = team
        self.user = user
        self.pw = pw
        self.bot_channel = bot_channel

        super(MattermostBot, self).__init__(url, team)
        super(MattermostBot, self).login(user, pw)

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        for message_part in text.split("\n\n"):
            self.post_channel(self.bot_channel, message_part)

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        json_message.setdefault("channel_id", self.bot_channel)
        json_message.setdefault("message", "")
        self.post("/posts", json_message)


class MattermostInput(InputChannel):
    """Mattermost input channel implemenation."""

    @classmethod
    def name(cls):
        return "mattermost"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(
            credentials.get("url"),
            credentials.get("team"),
            credentials.get("user"),
            credentials.get("pw"),
        )

    def __init__(self, url: Text, team: Text, user: Text, pw: Text) -> None:
        """Create a Mattermost input channel.
        Needs a couple of settings to properly authenticate and validate
        messages.

        Args:
            url: Your Mattermost team url including /v4 example
                https://mysite.example.com/api/v4
            team: Your mattermost team name
            user: Your mattermost userid that will post messages
            pw: Your mattermost password for your user
        """
        self.url = url
        self.team = team
        self.user = user
        self.pw = pw

    def blueprint(self, on_new_message):
        mattermost_webhook = Blueprint("mattermost_webhook", __name__)

        @mattermost_webhook.route("/", methods=["GET"])
        async def health(request: Request):
            return response.json({"status": "ok"})

        @mattermost_webhook.route("/webhook", methods=["POST"])
        async def webhook(request: Request):
            output = request.json
            if output:
                # splitting to get rid of the @botmention
                # trigger we are using for this
                text = output["text"].split(" ", 1)
                text = text[1]
                sender_id = output["user_id"]
                self.bot_channel = output["channel_id"]
                try:
                    out_channel = MattermostBot(
                        self.url, self.team, self.user, self.pw, self.bot_channel
                    )
                    user_msg = UserMessage(
                        text, out_channel, sender_id, input_channel=self.name()
                    )
                    await on_new_message(user_msg)
                except Exception as e:
                    logger.error(
                        "Exception when trying to handle message.{0}".format(e)
                    )
                    logger.debug(e, exc_info=True)
                    pass
            return response.text("")

        return mattermost_webhook
