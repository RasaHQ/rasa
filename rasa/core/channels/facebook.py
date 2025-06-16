import hashlib
import hmac
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Text, Union

import structlog
from fbmessenger import MessengerClient
from fbmessenger.attachments import Image
from fbmessenger.elements import Text as FBText
from fbmessenger.quick_replies import QuickReplies, QuickReply
from fbmessenger.sender_actions import SenderAction
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse

import rasa.shared.utils.io
from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage

logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()


class Messenger:
    """Implement a fbmessenger to parse incoming webhooks and send msgs."""

    @classmethod
    def name(cls) -> Text:
        return "facebook"

    def __init__(
        self,
        page_access_token: Text,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> None:
        self.on_new_message = on_new_message
        self.client = MessengerClient(page_access_token)
        self.last_message: Dict[Text, Any] = {}

    def get_user_id(self) -> Text:
        return self.last_message.get("sender", {}).get("id", "")

    @staticmethod
    def _is_audio_message(message: Dict[Text, Any]) -> bool:
        """Check if the users message is a recorded voice message."""
        return (
            "message" in message
            and "attachments" in message["message"]
            and message["message"]["attachments"][0]["type"] == "audio"
        )

    @staticmethod
    def _is_image_message(message: Dict[Text, Any]) -> bool:
        """Check if the users message is an image."""
        return (
            "message" in message
            and "attachments" in message["message"]
            and message["message"]["attachments"][0]["type"] == "image"
        )

    @staticmethod
    def _is_video_message(message: Dict[Text, Any]) -> bool:
        """Check if the users message is a video."""
        return (
            "message" in message
            and "attachments" in message["message"]
            and message["message"]["attachments"][0]["type"] == "video"
        )

    @staticmethod
    def _is_file_message(message: Dict[Text, Any]) -> bool:
        """Check if the users message is a file."""
        return (
            "message" in message
            and "attachments" in message["message"]
            and message["message"]["attachments"][0]["type"] == "file"
        )

    @staticmethod
    def _is_user_message(message: Dict[Text, Any]) -> bool:
        """Check if the message is a message from the user."""
        return (
            "message" in message
            and "text" in message["message"]
            and not message["message"].get("is_echo")
        )

    @staticmethod
    def _is_quick_reply_message(message: Dict[Text, Any]) -> bool:
        """Check if the message is a quick reply message."""
        return (
            message.get("message") is not None
            and message["message"].get("quick_reply") is not None
            and message["message"]["quick_reply"].get("payload")
        )

    async def handle(self, payload: Dict, metadata: Optional[Dict[Text, Any]]) -> None:
        for entry in payload["entry"]:
            for message in entry["messaging"]:
                self.last_message = message
                if message.get("message"):
                    return await self.message(message, metadata)
                elif message.get("postback"):
                    return await self.postback(message, metadata)

    async def message(
        self, message: Dict[Text, Any], metadata: Optional[Dict[Text, Any]]
    ) -> None:
        """Handle an incoming event from the fb webhook."""
        # quick reply and user message both share 'text' attribute
        # so quick reply should be checked first
        if self._is_quick_reply_message(message):
            text = message["message"]["quick_reply"]["payload"]
        elif self._is_user_message(message):
            text = message["message"]["text"]
        elif self._is_audio_message(message):
            attachment = message["message"]["attachments"][0]
            text = attachment["payload"]["url"]
        elif self._is_image_message(message):
            attachment = message["message"]["attachments"][0]
            text = attachment["payload"]["url"]
        elif self._is_video_message(message):
            attachment = message["message"]["attachments"][0]
            text = attachment["payload"]["url"]
        elif self._is_file_message(message):
            attachment = message["message"]["attachments"][0]
            text = attachment["payload"]["url"]
        else:
            structlogger.warning("facebook.message.cannot.handle")
            return

        await self._handle_user_message(text, self.get_user_id(), metadata)

    async def postback(
        self, message: Dict[Text, Any], metadata: Optional[Dict[Text, Any]]
    ) -> None:
        """Handle a postback (e.g. quick reply button)."""
        text = message["postback"]["payload"]
        await self._handle_user_message(text, self.get_user_id(), metadata)

    async def _handle_user_message(
        self, text: Text, sender_id: Text, metadata: Optional[Dict[Text, Any]]
    ) -> None:
        """Pass on the text to the dialogue engine for processing."""
        out_channel = MessengerBot(self.client)
        await out_channel.send_action(sender_id, sender_action="mark_seen")

        user_msg = UserMessage(
            text, out_channel, sender_id, input_channel=self.name(), metadata=metadata
        )
        await out_channel.send_action(sender_id, sender_action="typing_on")
        # noinspection PyBroadException
        try:
            await self.on_new_message(user_msg)
        except Exception:
            logger.exception(
                "Exception when trying to handle webhook for facebook message."
            )
            pass
        finally:
            await out_channel.send_action(sender_id, sender_action="typing_off")


class MessengerBot(OutputChannel):
    """A bot that uses fb-messenger to communicate."""

    @classmethod
    def name(cls) -> Text:
        return "facebook"

    def __init__(self, messenger_client: MessengerClient) -> None:
        self.messenger_client = messenger_client
        super().__init__()

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
        for message_part in text.strip().split("\n\n"):
            self.send(recipient_id, FBText(text=message_part))

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """Sends an image. Default will just post the url as a string."""
        self.send(recipient_id, Image(url=image))

    async def send_action(self, recipient_id: Text, sender_action: Text) -> None:
        """Sends a sender action to facebook (e.g. "typing_on").

        Args:
            recipient_id: recipient
            sender_action: action to send, e.g. "typing_on" or "mark_seen"
        """
        self.messenger_client.send_action(
            SenderAction(sender_action).to_dict(), recipient_id
        )

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """Sends buttons to the output."""
        # buttons is a list of tuples: [(option_name,payload)]
        if len(buttons) > 3:
            rasa.shared.utils.io.raise_warning(
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
        **kwargs: Any,
    ) -> None:
        """Sends quick replies to the output."""
        quick_replies = self._convert_to_quick_reply(quick_replies)
        self.send(recipient_id, FBText(text=text, quick_replies=quick_replies))

    async def send_elements(
        self, recipient_id: Text, elements: Iterable[Dict[Text, Any]], **kwargs: Any
    ) -> None:
        """Sends elements to the output."""
        for element in elements:
            if "buttons" in element:
                self._add_postback_info(element["buttons"])

        payload = {
            "attachment": {
                "type": "template",
                "payload": {"template_type": "generic", "elements": elements},
            }
        }
        self.messenger_client.send(payload, recipient_id, "RESPONSE")

    async def send_custom_json(
        self,
        recipient_id: Text,
        json_message: Union[List, Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """Sends custom json data to the output."""
        if isinstance(json_message, dict) and "sender" in json_message.keys():
            recipient_id = json_message.pop("sender", {}).pop("id", recipient_id)
        elif isinstance(json_message, list):
            for message in json_message:
                if "sender" in message.keys():
                    recipient_id = message.pop("sender", {}).pop("id", recipient_id)
                    break

        self.messenger_client.send(json_message, recipient_id, "RESPONSE")

    @staticmethod
    def _add_postback_info(buttons: List[Dict[Text, Any]]) -> None:
        """Make sure every button has a type. Modifications happen in place."""
        for button in buttons:
            if "type" not in button:
                button["type"] = "postback"

    @staticmethod
    def _convert_to_quick_reply(quick_replies: List[Dict[Text, Any]]) -> QuickReplies:
        """Convert quick reply dictionary to FB QuickReplies object."""
        fb_quick_replies = []
        for quick_reply in quick_replies:
            try:
                fb_quick_replies.append(
                    QuickReply(
                        title=quick_reply["title"],
                        payload=quick_reply["payload"],
                        content_type=quick_reply.get("content_type"),
                    )
                )
            except KeyError as e:
                raise ValueError(
                    'Facebook quick replies must define a "{}" field.'.format(e.args[0])
                )

        return QuickReplies(quick_replies=fb_quick_replies)


class FacebookInput(InputChannel):
    """Facebook input channel implementation. Based on the HTTPInputChannel."""

    @classmethod
    def name(cls) -> Text:
        return "facebook"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
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

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        fb_webhook = Blueprint("fb_webhook", __name__)

        # noinspection PyUnusedLocal
        @fb_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @fb_webhook.route("/webhook", methods=["GET"])
        async def token_verification(request: Request) -> HTTPResponse:
            if request.args.get("hub.verify_token") == self.fb_verify:
                return response.text(request.args.get("hub.challenge"))
            else:
                logger.warning(
                    "Invalid fb verify token! Make sure this matches "
                    "your webhook settings on the facebook app."
                )
                return response.text("failure, invalid token")

        @fb_webhook.route("/webhook", methods=["POST"])
        async def webhook(request: Request) -> HTTPResponse:
            signature = request.headers.get("X-Hub-Signature") or ""
            if not self.validate_hub_signature(self.fb_secret, request.body, signature):
                logger.warning(
                    "Wrong fb secret! Make sure this matches the "
                    "secret in your facebook app settings"
                )
                return response.text("not validated")

            messenger = Messenger(self.fb_access_token, on_new_message)

            metadata = self.get_metadata(request)
            await messenger.handle(request.json, metadata)
            return response.text("success")

        return fb_webhook

    @staticmethod
    def validate_hub_signature(
        app_secret: Text, request_payload: bytes, hub_signature_header: Text
    ) -> bool:
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

    def get_output_channel(self) -> OutputChannel:
        client = MessengerClient(self.fb_access_token)
        return MessengerBot(client)
