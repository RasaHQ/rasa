from linebot import LineBotApi, WebhookParser
from linebot.exceptions import LineBotApiError
from rasa.core.channels.channel import InputChannel, UserMessage, OutputChannel
from typing import Dict, Text, Any, List, Optional, Callable, Awaitable, Union
from sanic import Blueprint, response
from sanic.response import HTTPResponse
from sanic.request import Request
import json
import logging


from linebot.models import (
    MessageEvent,
    TextSendMessage,
    FlexSendMessage,
    StickerSendMessage,
    ImageSendMessage,
    VideoSendMessage,
    AudioSendMessage,
    LocationSendMessage,
    TemplateSendMessage,
    ConfirmTemplate,
    CarouselTemplate,
    BotInfo,
)

logger = logging.getLogger(__name__)


def has_empty_values(data):
    if isinstance(data, dict):
        for value in data.values():
            if not value or has_empty_values(value):
                return True
    elif isinstance(data, list):
        for item in data:
            if has_empty_values(item):
                return True
    elif not data:
        return True
    return False


class Line:
    """Implement a line to parse incoming webhooks and send msgs."""

    @classmethod
    def name(cls) -> Text:
        return "line"

    def __init__(
        self,
        access_token: Text,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> None:
        self.on_new_message = on_new_message
        self.client = LineBotApi(access_token)
        self.last_message: Dict[Text, Any] = {}
        self.access_token = access_token

    def get_user_id(self) -> Text:
        return self.last_message.source.sender_id

    async def handle(
        self, event: MessageEvent, metadata: Optional[Dict[Text, Any]]
    ) -> None:
        self.last_message = event
        return await self.message(event, metadata)

    @staticmethod
    def _is_user_message(event: MessageEvent) -> bool:
        """Check if the message is a message from the user"""
        logger.debug(f"souce type:{type(event)}")
        return event.source.type == "user"

    async def message(
        self, event: MessageEvent, metadata: Optional[Dict[Text, Any]]
    ) -> None:
        """Handle an incoming event from the line webhook."""

        if self._is_user_message(event):
            if hasattr(event, "message"):
                # extract text from user message
                text = event.message.text
            else:
                # extract data from user postback
                text = event.postback.data
        else:
            logger.warning(
                "Received a message from line that we can not "
                f"handle. Event: {event}"
            )
            return

        await self._handle_user_message(event, text, self.get_user_id(), metadata)

    async def _handle_user_message(
        self,
        event: MessageEvent,
        text: Text,
        sender_id: Text,
        metadata: Optional[Dict[Text, Any]],
    ) -> None:
        """Pass on the text to the dialogue engine for processing."""

        out_channel = LineConnectorOutput(self.access_token, event)

        user_msg = UserMessage(
            text, out_channel, sender_id, input_channel=self.name(), metadata=metadata
        )
        try:
            await self.on_new_message(user_msg)
        except Exception:
            logger.exception(
                "Exception when trying to handle webhook for line message."
            )
            pass


class LineConnectorOutput(OutputChannel):
    """Output channel for Line."""

    @classmethod
    def name(cls) -> Text:
        return "line"

    def __init__(self, channel_access_token: Optional[Text], event: Any) -> None:
        self.line_client = LineBotApi(channel_access_token)
        self.reply_token = event.reply_token
        self.sender_id = event.source.user_id
        super().__init__()

    async def send_to_line(
        self,
        payload_object: Union[
            TextSendMessage,
            FlexSendMessage,
            StickerSendMessage,
            ImageSendMessage,
            VideoSendMessage,
            AudioSendMessage,
            LocationSendMessage,
            TemplateSendMessage,
        ],
        **kwargs: Any,
    ) -> None:
        try:
            if self.reply_token:
                self.line_client.reply_message(
                    self.reply_token, messages=payload_object
                )
            else:
                self.line_client.push_message(
                    to=self.sender_id, messages=payload_object
                )
        except LineBotApiError as e:
            logger.error(f"Line Error: {e.error.message}")
            logger.error(f"Payload: {payload_object}")
            if (
                e.status_code == 400
                or e.error.message == "Invalid reply token, trying to push message."
            ):
                logger.info("Pushing Message...")
                self.line_client.push_message(
                    to=self.sender_id, messages=payload_object
                )

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        try:
            json_converted = json.loads(text)
            logger.debug(f"json_converted:{json_converted}")
            message_type = json_converted.get("type")

            if message_type == "flex":
                # send flex
                await self.send_to_line(
                    FlexSendMessage(
                        alt_text=json_converted.get("alt_text"),
                        contents=json_converted.get("contens"),
                    )
                )
            elif message_type == "sticker":
                # send sticker
                await self.send_to_line(
                    StickerSendMessage(
                        package_id=json_converted.get("package_id"),
                        sticker_id=json_converted.get("sticker_id"),
                    )
                )
            elif message_type == "image":
                # send image
                await self.send_to_line(
                    ImageSendMessage(
                        original_content_url=json_converted.get("original_content_url"),
                        preview_image_url=json_converted.get("preview_image_url"),
                    )
                )
            elif message_type == "video":
                # send video
                await self.send_to_line(
                    VideoSendMessage(
                        original_content_url=json_converted.get("original_content_url"),
                        preview_image_url=json_converted.get("preview_image_url"),
                        tracking_id=json_converted.get("tracking_id"),
                    )
                )
            elif message_type == "audio":
                # send audio
                await self.send_to_line(
                    AudioSendMessage(
                        original_content_url=json_converted.get("original_content_url"),
                        duration=json_converted.get("duration"),
                    )
                )
            elif message_type == "location":
                # send location
                await self.send_to_line(
                    LocationSendMessage(
                        title=json_converted.get("title"),
                        address=json_converted.get("address"),
                        latitude=json_converted.get("latitude"),
                        longitude=json_converted.get("longitude"),
                    )
                )
            elif message_type == "template":
                template = json_converted.get("template")
                template_type = template.get("type")
                logger.debug(f"template:{template} type:{type}")
                # case: does't have type is confirm template
                if has_empty_values(template_type):
                    await self.send_to_line(
                        TemplateSendMessage(
                            alt_text=json_converted.get("alt_text"),
                            template=ConfirmTemplate(
                                text=template.get("text"),
                                actions=template.get("actions"),
                            ),
                        )
                    )
                else:
                    # other is normal template
                    if template_type == "carousel":
                        # handle carousel template
                        await self.send_to_line(
                            TemplateSendMessage(
                                alt_text=json_converted.get("alt_text"),
                                template=CarouselTemplate(
                                    columns=template.get("columns"),
                                    image_aspect_ratio=template.get(
                                        "image_aspect_ratio"
                                    ),
                                    image_size=template.get("image_size"),
                                ),
                            )
                        )
                    else:
                        # heandle normal template
                        await self.send_to_line(
                            TemplateSendMessage(
                                alt_text=json_converted.get("alt_text"),
                                template=json_converted.get("template"),
                            )
                        )
            else:
                # default is handle case with text
                if not has_empty_values(message_type):
                    text = json_converted.get("text")
                await self.send_to_line(
                    TextSendMessage(
                        text=text,
                        quick_reply=json_converted.get("quick_reply"),
                        emojis=json_converted.get("emojis"),
                    )
                )
        except ValueError:
            message_object = TextSendMessage(text=text)
            await self.send_to_line(message_object)

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


class LineConnectorInput(InputChannel):
    """Line input channel"""

    @classmethod
    def name(cls) -> Text:
        return "line"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(
            credentials.get("app_secret"),
            credentials.get("access_token"),
        )

    def __init__(
        self,
        app_secret: Text,
        access_token: Text,
    ) -> None:
        self.app_secret = app_secret
        self.access_token = access_token

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """Send line payload to call back url: https://{HOST}/webhooks/line/callback"""

        line_webhook = Blueprint("line_webhook", __name__)
        parser = self.get_line_message_parser()

        @line_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @line_webhook.route("/bot/info", methods=["GET"])
        async def info(_: Request) -> HTTPResponse:
            bot_info = LineBotApi(self.access_token).get_bot_info()
            return response.json(json.dumps(bot_info, cls=BotInfoEncoder))

        @line_webhook.route("/callback", methods=["POST"])
        async def message(request: Request) -> Any:
            if request.method == "POST":
                # CHECK IF FROM LINE APP
                signature = request.headers.get("X-Line-Signature", None)
                if signature:
                    body = request.body.decode("utf-8")
                    events = parser.parse(body, signature)
                    logger.debug(f"Web Hook Receive:{events}")
                    for event in events:
                        line = Line(self.access_token, on_new_message)
                        metadata = self.get_metadata(request)
                        await line.handle(event, metadata)
                    return response.json({"status": "Line Webhook success"})

                else:
                    return response.json(request.json)

        return line_webhook

    def get_line_message_parser(self) -> WebhookParser:
        """Loads Line WebhookParser"""
        parser = WebhookParser(self.app_secret)
        return parser


class BotInfoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BotInfo):
            # Return a serializable version of the object
            return obj.__dict__
        return super().default(obj)
