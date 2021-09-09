import hashlib
import hmac
import inspect
import logging
from typing import Text, Dict, Any, Callable, Awaitable, Optional, List

from fbmessenger import MessengerClient
from fbmessenger.elements import Text as InstaText
from fbmessenger.attachments import Image
from fbmessenger.quick_replies import QuickReplies, QuickReply
from fbmessenger.sender_actions import SenderAction
from rasa.core.channels.channel import UserMessage, OutputChannel, InputChannel
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse

logger = logging.getLogger(__name__)


class Messenger:
    @classmethod
    def name(cls) -> Text:
        return "instagram"

    def __init__(
        self,
        page_access_token: Text,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> None:
        self.on_new_message = on_new_message
        self.page_access_token = page_access_token
        self.client = MessengerClient(page_access_token)
        self.last_message: Dict[Text, Any] = {}

    def get_user_id(self, metadata) -> Text:
        logger.info(
            f"function-name: {inspect.stack()[0][3]}\n"
            f"metadata: {metadata}"
            f"last message: {self.last_message}"
        )
        return self.last_message.get("sender", {}).get("id", "")

    @staticmethod
    def _is_user_message(message: Dict[Text, Any]) -> bool:
        """Check if the message is a message from the user"""
        print(
            f"function-name: {inspect.stack()[0][3]}\n" f"user_message: {str(message)}"
        )
        return "text" in message["message"]

    @staticmethod
    def _is_image_message(message: Dict[Text, Any]) -> bool:
        """Check if the users message is an image."""
        return (
            "message" in message
            and "attachments" in message["message"]
            and message["message"]["attachments"][0]["type"] == "image"
        )

    @staticmethod
    def _is_quick_reply_message(message: Dict[Text, Any]) -> bool:
        """Check if the message is a quick reply message."""
        return (
            message.get("message") is not None
            and message["message"].get("quick_reply") is not None
            and message["message"]["quick_reply"].get("payload")
        )

    @staticmethod
    def _is_video_message(message: Dict[Text, Any]) -> bool:
        """Check if the users message is a video."""
        return (
            "message" in message
            and "attachments" in message["message"]
            and message["message"]["attachments"][0]["type"] == "video"
        )

    async def handle(self, payload: Dict, metadata: Optional[Dict[Text, Any]]) -> None:
        for entry in payload["entry"]:
            for message in entry["messaging"]:
                self.last_message = message
                if message.get("message"):
                    return await self.message(message, metadata)
                elif message.get("postback"):
                    pass

    async def message(
        self, message: Dict[Text, Any], metadata: Optional[Dict[Text, Any]]
    ) -> None:
        if self._is_user_message(message):
            text = message["message"]["text"]
        elif self._is_image_message(message):
            attachment = message["message"]["attachments"][0]
            text = attachment["payload"]["url"]
        elif self._is_quick_reply_message(message):
            text = message["message"]["quick_reply"]["payload"]
        elif self._is_video_message(message):
            attachment = message["message"]["attachments"][0]
            text = attachment["payload"]["url"]
        else:
            logger.warning(
                f"function-name: {inspect.stack()[0][3]}\n"
                f"Received a message from instagram that we can not "
                f"handle. Message: {message}"
            )
            return
        await self._handle_user_message(text, self.get_user_id(metadata), metadata)

    async def _handle_user_message(
        self, text: Text, sender_id: Text, metadata: Optional[Dict[Text, Any]]
    ) -> None:
        logger.info(
            f"function-name: {inspect.stack()[0][3]}\n"
            f"sender-id: {sender_id}\n"
            f"metadata: {str(metadata)}"
        )
        out_channel = MessengerBot(self.client)
        await out_channel.send_action(sender_id, sender_action="mark_seen")

        user_msg = UserMessage(
            text, out_channel, sender_id, input_channel=self.name(), metadata=metadata
        )
        await out_channel.send_action(sender_id, sender_action="typing_on")

        try:
            await self.on_new_message(user_msg)
        except Exception as e:
            logger.error(f"function-name: {inspect.stack()[0][3]}\n" f"error: {e}")
        finally:
            await out_channel.send_action(sender_id, sender_action="typing_off")


class MessengerBot(OutputChannel):
    @classmethod
    def name(cls) -> Text:
        return "instagram"

    def __init__(self, instagram_client: MessengerClient) -> None:
        self.instagram_client = instagram_client
        super().__init__()

    def send(self, recipient_id: Text, element: Any):
        self.instagram_client.send(element.to_dict(), recipient_id, "RESPONSE")

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        for message_part in text.strip().split("\n\n"):
            print(f"message: {message_part}")
            self.send(recipient_id, InstaText(text=message_part))

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        self.send(recipient_id, Image(url=image))

    async def send_quick_replies(
        self,
        recipient_id: Text,
        text: Text,
        quick_replies: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """Sends quick replies to the output."""

        quick_replies = self._convert_to_quick_reply(quick_replies)
        self.send(recipient_id, InstaText(text=text, quick_replies=quick_replies))

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """Sends buttons to the output.
        Currently there is no predefined way to create a message with
        buttons in the instagram messenger framework - so we need to create the payload on our own
         """
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
        self.instagram_client.send(payload, recipient_id, "RESPONSE")

    @staticmethod
    def _convert_to_quick_reply(quick_replies: List[Dict[Text, Any]]) -> QuickReplies:
        """Convert quick reply dictionary to FB QuickReplies object"""

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

    @staticmethod
    def _add_postback_info(buttons: List[Dict[Text, Any]]) -> None:
        """Make sure every button has a type. Modifications happen in place."""
        for button in buttons:
            if "type" not in button:
                button["type"] = "postback"

    async def send_action(self, recipient_id: Text, sender_action: Text) -> None:
        self.instagram_client.send_action(
            SenderAction(sender_action).to_dict(), recipient_id
        )

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """
        currently instagram supports just heart sticker so added here.
        and for carousels they can be just generic and product so I added validation
        more info can be checked here:
        https://developers.facebook.com/docs/messenger-platform/instagram/features/send-message
        """
        recipient_id = json_message.pop("sender", {}).pop("id", None) or recipient_id
        if json_message["attachment"]["type"] == "like_heart":
            self.instagram_client.send(json_message, recipient_id, "RESPONSE")
        elif json_message["template_type"] in ["generic", "product"]:
            self.instagram_client.send(json_message, recipient_id, "RESPONSE")
        else:
            print(
                "Instagram API currently doesn't support other kind of messages. "
                "If you add, all will be ignored."
            )


class InstagramInput(InputChannel):
    @classmethod
    def name(cls) -> Text:
        return "instagram"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(
            credentials.get("insta-verify"),
            credentials.get("insta-secret"),
            credentials.get("insta-page-access-token"),
        )

    def __init__(
        self, insta_verify: Text, insta_secret: Text, insta_access_token: Text
    ) -> None:
        self.insta_verify = insta_verify
        self.insta_secret = insta_secret
        self.insta_access_token = insta_access_token
        print(
            f"verify: {self.insta_verify}\n"
            f"secret: {self.insta_secret}\n"
            f"token: {self.insta_access_token}"
        )

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:

        insta_webhook = Blueprint("insta_webhook", __name__)

        @insta_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @insta_webhook.route("/webhook", methods=["GET"])
        async def token_verification(request: Request) -> HTTPResponse:
            if request.args.get("hub.verify_token") == self.insta_verify:
                return response.text(request.args.get("hub.challenge"))
            else:
                print(
                    "Invalid instagram verify token! Make sure this matches "
                    "your webhook settings on the facebook app."
                )
                return response.text("failure, invalid token")

        @insta_webhook.route("/webhook", methods=["POST"])
        async def webhook(request: Request) -> HTTPResponse:
            signature = request.headers.get("X-Hub_Signature") or ""
            # if not self.validate_hub_signature(self.insta_secret, request.body, signature):
            #     print(
            #         "Wrong instagram secret! Make sure this matches the "
            #         "secret in your facebook app settings"
            #     )
            #     return response.text("not validated")
            messenger = Messenger(self.insta_access_token, on_new_message)
            metadata = self.get_metadata(request)
            await messenger.handle(request.json, metadata)
            print(
                f"function-name: {inspect.stack()[0][3]}\n" f"response: {response.text}"
            )
            return response.text("success")

        return insta_webhook

    @staticmethod
    def validate_hub_signature(
        app_secret: Text, request_payload: bytes, hub_signature_header: Text
    ) -> bool:
        try:
            hash_method, hub_signature = hub_signature_header.split("=")
        except Exception:
            pass
        else:
            digest_module = getattr(hashlib, hash_method)
            hmac_object = hmac.new(
                bytearray(app_secret, "utf-8"), request_payload, digest_module
            )
            generated_hash = hmac_object.hexdigest()
            if hub_signature == generated_hash:
                return True
            return False

    def get_output_channel(self) -> Optional["OutputChannel"]:
        client = MessengerClient(self.insta_access_token)
        return MessengerBot(client)
