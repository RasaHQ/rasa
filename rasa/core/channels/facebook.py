import hashlib
import hmac
import logging
from fbmessenger import MessengerClient
from fbmessenger.attachments import Image
from fbmessenger.elements import Text as FBText
from sanic import Blueprint, response
from sanic.request import Request
from typing import Text, List, Dict, Any, Callable, Awaitable, Iterable

from rasa.core.channels.channel import UserMessage, OutputChannel, InputChannel

logger = logging.getLogger(__name__)


class Messenger:
    """Implement a fbmessenger to parse incoming webhooks and send msgs."""

    @classmethod
    def name(cls):
        return "facebook"

    def __init__(
        self,
        page_access_token: Text,
        on_new_message: Callable[[UserMessage], Awaitable[None]],
    ) -> None:

        self.on_new_message = on_new_message
        self.client = MessengerClient(page_access_token)
        self.last_message = {}

    def get_user_id(self):
        return self.last_message["sender"]["id"]

    @staticmethod
    def _is_audio_message(message: Dict[Text, Any]) -> bool:
        """Check if the users message is a recorced voice message."""
        return (
            message.get("message")
            and message["message"].get("attachments")
            and message["message"]["attachments"][0]["type"] == "audio"
        )

    @staticmethod
    def _is_user_message(message: Dict[Text, Any]) -> bool:
        """Check if the message is a message from the user"""
        return (
            message.get("message")
            and message["message"].get("text")
            and not message["message"].get("is_echo")
        )

    async def handle(self, payload):
        for entry in payload["entry"]:
            for message in entry["messaging"]:
                self.last_message = message
                if message.get("message"):
                    return await self.message(message)
                elif message.get("postback"):
                    return await self.postback(message)

    async def message(self, message: Dict[Text, Any]) -> None:
        """Handle an incoming event from the fb webhook."""

        if self._is_user_message(message):
            text = message["message"]["text"]
        elif self._is_audio_message(message):
            attachment = message["message"]["attachments"][0]
            text = attachment["payload"]["url"]
        else:
            logger.warning(
                "Received a message from facebook that we can not "
                "handle. Message: {}".format(message)
            )
            return

        await self._handle_user_message(text, self.get_user_id())

    async def postback(self, message: Dict[Text, Any]) -> None:
        """Handle a postback (e.g. quick reply button)."""

        text = message["postback"]["payload"]
        await self._handle_user_message(text, self.get_user_id())

    async def _handle_user_message(self, text: Text, sender_id: Text) -> None:
        """Pass on the text to the dialogue engine for processing."""

        out_channel = MessengerBot(self.client)
        user_msg = UserMessage(text, out_channel, sender_id, input_channel=self.name())

        # noinspection PyBroadException
        try:
            await self.on_new_message(user_msg)
        except Exception:
            logger.exception(
                "Exception when trying to handle webhook for facebook message."
            )
            pass


class MessengerBot(OutputChannel):
    """A bot that uses fb-messenger to communicate."""

    @classmethod
    def name(cls):
        return "facebook"

    def __init__(self, messenger_client: MessengerClient) -> None:

        self.messenger_client = messenger_client
        super(MessengerBot, self).__init__()

    def send(self, recipient_id: Text, element: Any) -> None:
        """Sends a message to the recipient using the messenger client."""

        # this is a bit hacky, but the client doesn't have a proper API to
        # send messages but instead expects the incoming sender to be present
        # which we don't have as it is stored in the input channel.
        self.messenger_client.send(element.to_dict(), recipient_id, "RESPONSE")

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Send a message through this channel."""

        for message_part in text.split("\n\n"):
            self.send(recipient_id, FBText(text=message_part))

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """Sends an image. Default will just post the url as a string."""

        self.send(recipient_id, Image(url=image))

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any
    ) -> None:
        """Sends buttons to the output."""

        # buttons is a list of tuples: [(option_name,payload)]
        if len(buttons) > 3:
            logger.warning(
                "Facebook API currently allows only up to 3 buttons. "
                "If you add more, all will be ignored."
            )
            await self.send_text_message(recipient_id, text, **kwargs)
        else:
            self._add_postback_info(buttons)

            # Currently there is no predefined way to create a message with
            # buttons in the fbmessenger framework - so we need to create the
            # payload on our own
            payload = {
                "attachment": {
                    "type": "template",
                    "payload": {
                        "template_type": "button",
                        "text": text,
                        "buttons": buttons,
                    },
                }
            }
            self.messenger_client.send(payload, recipient_id, "RESPONSE")

    async def send_quick_replies(
        self,
        recipient_id: Text,
        text: Text,
        quick_replies: List[Dict[Text, Any]],
        **kwargs: Any
    ) -> None:
        """Sends quick replies to the output."""

        self._add_text_info(quick_replies)
        self.send(recipient_id, FBText(text=text, quick_replies=quick_replies))

    async def send_elements(
        self, recipient_id: Text, elements: Iterable[Dict[Text, Any]], **kwargs: Any
    ) -> None:
        """Sends elements to the output."""

        for element in elements:
            self._add_postback_info(element["buttons"])

        payload = {
            "attachment": {
                "type": "template",
                "payload": {"template_type": "generic", "elements": elements},
            }
        }
        self.messenger_client.send(payload, recipient_id, "RESPONSE")

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """Sends custom json data to the output."""

        recipient_id = json_message.pop("sender", {}).pop("id", None) or recipient_id

        self.messenger_client.send(json_message, recipient_id, "RESPONSE")

    @staticmethod
    def _add_text_info(quick_replies: List[Dict[Text, Any]]) -> None:
        """Set quick reply type to text for all buttons without content type.

        Happens in place."""

        for quick_reply in quick_replies:
            if not quick_reply.get("type"):
                quick_reply["content_type"] = "text"

    @staticmethod
    def _add_postback_info(buttons: List[Dict[Text, Any]]) -> None:
        """Make sure every button has a type. Modifications happen in place."""
        for button in buttons:
            if "type" not in button:
                button["type"] = "postback"


class FacebookInput(InputChannel):
    """Facebook input channel implementation. Based on the HTTPInputChannel."""

    @classmethod
    def name(cls):
        return "facebook"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(
            credentials.get("verify"),
            credentials.get("secret"),
            credentials.get("page-access-token"),
        )

    def __init__(self, fb_verify: Text, fb_secret: Text, fb_access_token: Text) -> None:
        """Create a facebook input channel.

        Needs a couple of settings to properly authenticate and validate
        messages. Details to setup:

        https://github.com/rehabstudio/fbmessenger#facebook-app-setup

        Args:
            fb_verify: FB Verification string
                (can be chosen by yourself on webhook creation)
            fb_secret: facebook application secret
            fb_access_token: access token to post in the name of the FB page
        """
        self.fb_verify = fb_verify
        self.fb_secret = fb_secret
        self.fb_access_token = fb_access_token

    def blueprint(self, on_new_message):

        fb_webhook = Blueprint("fb_webhook", __name__)

        @fb_webhook.route("/", methods=["GET"])
        async def health(request: Request):
            return response.json({"status": "ok"})

        @fb_webhook.route("/webhook", methods=["GET"])
        async def token_verification(request: Request):
            if request.args.get("hub.verify_token") == self.fb_verify:
                return response.text(request.args.get("hub.challenge"))
            else:
                logger.warning(
                    "Invalid fb verify token! Make sure this matches "
                    "your webhook settings on the facebook app."
                )
                return response.text("failure, invalid token")

        @fb_webhook.route("/webhook", methods=["POST"])
        async def webhook(request: Request):
            signature = request.headers.get("X-Hub-Signature") or ""
            if not self.validate_hub_signature(self.fb_secret, request.body, signature):
                logger.warning(
                    "Wrong fb secret! Make sure this matches the "
                    "secret in your facebook app settings"
                )
                return response.text("not validated")

            messenger = Messenger(self.fb_access_token, on_new_message)

            await messenger.handle(request.json)
            return response.text("success")

        return fb_webhook

    @staticmethod
    def validate_hub_signature(app_secret, request_payload, hub_signature_header):
        """Make sure the incoming webhook requests are properly signed.

        Args:
            app_secret: Secret Key for application
            request_payload: request body
            hub_signature_header: X-Hub-Signature header sent with request

        Returns:
            bool: indicated that hub signature is validated
        """

        # noinspection PyBroadException
        try:
            hash_method, hub_signature = hub_signature_header.split("=")
        except Exception:
            pass
        else:
            digest_module = getattr(hashlib, hash_method)
            hmac_object = hmac.new(
                bytearray(app_secret, "utf8"), request_payload, digest_module
            )
            generated_hash = hmac_object.hexdigest()
            if hub_signature == generated_hash:
                return True
        return False
