import asyncio
import inspect
import json
import logging
import uuid
from asyncio import Queue, CancelledError
from sanic import Sanic, Blueprint, response
from sanic.request import Request
from typing import Text, List, Dict, Any, Optional, Callable, Iterable, Awaitable

import rasa.utils.endpoints
from rasa.cli import utils as cli_utils
from rasa.constants import DOCS_BASE_URL
from rasa.core import utils
from sanic.response import HTTPResponse

from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage

try:
    from urlparse import urljoin  # pytype: disable=import-error
except ImportError:
    from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class HangoutsOutput(OutputChannel):
    @classmethod
    def name(cls) -> Text:
        return "hangouts"

    def __init__(self) -> None:
        self.messages = {}

    def _text_card(self, message: Dict[Text, Any]) -> Dict:

        card = {
            "cards": [
                {
                    "sections": [
                        {"widgets": [{"textParagraph": {"text": message["text"]}}]}
                    ]
                }
            ]
        }
        return card

    def _image_card(self, image: Text) -> Dict:
        card = {
            "cards": [{"sections": [{"widgets": [{"image": {"imageUrl": image}}]}]}]
        }
        return card

    def _text_button_card(self, text: Text, buttons: List) -> Dict:
        hangouts_buttons = []
        for b in buttons:
            try:
                b_txt, b_pl = b["title"], b["payload"]
            except KeyError:
                logger.error(
                    "Buttons must be a list of dicts with 'title' and 'payload' as keys"
                )

            hangouts_buttons.append(
                {
                    "textButton": {
                        "text": b_txt,
                        "onClick": {"action": {"actionMethodName": b_pl}},
                    }
                }
            )

        card = {
            "cards": [
                {
                    "sections": [
                        {
                            "widgets": [
                                {"textParagraph": {"text": text}},
                                {"buttons": hangouts_buttons},
                            ]
                        }
                    ]
                }
            ]
        }
        return card

    def _combine_cards(self, c1: Dict, c2: Dict) -> Dict:
        return {"cards": [*c1["cards"], *c2["cards"]]}

    async def _persist_message(self, message: Dict) -> None:
        """Google Hangouts only accepts single dict with single key 'text'
        for simple text messages. All other responses must be sent as cards.
        
        In case the bot sends multiple messages, all are transformed to either
        cards or text output"""

        # check whether current and previous message will send 'text' or 'card'
        if self.messages.get("text"):
            msg_state = "text"
        elif self.messages.get("cards"):
            msg_state = "cards"
        else:
            msg_state = None

        if message.get("text"):
            msg_new = "text"
        elif message.get("cards"):
            msg_new = "cards"
        else:
            raise Exception(
                "Your message to Hangouts channel must either contain 'text' or 'cards'!"
            )

        # depending on above outcome, convert messages into same type and combine
        if msg_new == msg_state == "text":
            # two text messages are simply appended
            new_text = " ".join([self.messages.get("text", ""), message["text"]])
            new_messages = {"text": new_text}

        elif msg_new == msg_state == "cards":
            # two cards are combined into one
            new_messages = self._combine_cards(self.messages, message)

        elif msg_state == "cards" and msg_new == "text":
            ## if any message is card, turn text message into TextParagraph card and combine cards
            text_card = self._text_card(message)
            new_messages = self._combine_cards(self.messages, text_card)

        elif msg_state == "text" and msg_new == "cards":
            text_card = self._text_card(self.messages)
            new_messages = self._combine_cards(text_card, message)

        elif msg_new == "text":
            new_messages = {"text": message.get("text")}
        else:
            new_messages = message

        self.messages = new_messages

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:

        await self._persist_message({"text": text})

    async def send_image_url(self, recipient_id: Text, image: Text, **kwargs) -> None:

        await self._persist_message(self._image_card(image))

    async def send_text_with_buttons(
        self, recipient_id: Text, text: Text, buttons: List, **kwargs
    ) -> None:

        await self._persist_message(self._text_button_card(text, buttons))

    async def send_attachment(self, recipient_id: Text, attachment: Text):

        await self.send_text_message(recipient_id, attachment)

    async def send_elements(
        self, recipient_id: Text, elements: Iterable[Dict[Text, Any]], **kwargs: Any
    ) -> None:
        raise NotImplementedError

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict, **kwargs
    ) -> None:
        """Custom json payload is simply forwarded to Google Hangouts without
        any modifications. Use this for more complex cards, which can be created
        in actions.py."""
        await self._persist_message(json_message)


## Google Hangouts input channel
class HangoutsInput(InputChannel):
    """
    Channel that uses Google Hangouts Chat API to communicate.
    """

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        credentials = credentials or {}
        return cls(
            credentials.get("user_added_intent"),
            credentials.get("room_added_intent"),
            credentials.get("removed_intent"),
        )

    def __init__(
        self,
        user_added_intent: Optional[Text] = None,
        room_added_intent: Optional[Text] = None,
        removed_intent: Optional[Text] = None,
    ) -> None:

        self.user_added_intent = user_added_intent
        self.room_added_intent = room_added_intent
        self.removed_intent = removed_intent

    @classmethod
    def name(cls) -> Text:
        return "hangouts"

    def _extract_sender(self, req: Request) -> Text:

        if req.json["type"] == "MESSAGE":
            return req.json["message"]["sender"]["displayName"]

        else:
            return req.json["user"]["displayName"]

    # noinspection PyMethodMayBeStatic
    def _extract_message(self, req: Request) -> Text:

        if req.json["type"] == "MESSAGE":
            return req.json["message"]["text"]

        elif req.json["type"] == "CARD_CLICKED":
            return req.json["action"]["actionMethodName"]

        elif req.json["type"] == "ADDED_TO_SPACE":
            if self._extract_room(req) and self.room_added_intent:
                return self.room_added_intent
            elif self.user_added_intent:
                return self.user_added_intent

        elif req.json["type"] == "REMOVED_FROM_SPACE":
            if self.removed_intent:
                return self.removed_intent

    def _extract_room(self, req: Request) -> Text:

        if req.json["space"]["type"] == "ROOM":
            return req.json["space"]["displayName"]

    def _extract_input_channel(self, req: Request) -> Text:
        return self.name()

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Blueprint:

        custom_webhook = Blueprint("hangouts_webhook", __name__)

        @custom_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @custom_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request) -> HTTPResponse:

            sender_id = self._extract_sender(request)
            room_name = self._extract_room(request)
            text = self._extract_message(request)
            if text is None:
                return response.text("OK")
            input_channel = self._extract_input_channel(request)

            collector = HangoutsOutput()

            try:
                await on_new_message(
                    UserMessage(
                        text,
                        collector,
                        sender_id,
                        input_channel=input_channel,
                        metadata={"room": room_name},
                    )
                )
            except CancelledError:
                logger.error(
                    "Message handling timed out for " "user message '{}'.".format(text)
                )
            except Exception:
                logger.exception(
                    "An exception occured while handling "
                    "user message '{}'.".format(text)
                )

            return response.json(collector.messages)

        return custom_webhook
