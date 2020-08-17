import logging
from asyncio import CancelledError
from sanic import Blueprint, response
from sanic.request import Request
from typing import Text, List, Dict, Any, Optional, Callable, Iterable, Awaitable, Union

from sanic.response import HTTPResponse
from sanic.exceptions import abort
from oauth2client import client
from oauth2client.crypt import AppIdentityError

from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage

logger = logging.getLogger(__name__)

CHANNEL_NAME = "hangouts"
CERT_URI = "https://www.googleapis.com/service_accounts/v1/metadata/x509/chat@system.gserviceaccount.com"


class HangoutsOutput(OutputChannel):
    @classmethod
    def name(cls) -> Text:
        return CHANNEL_NAME

    def __init__(self) -> None:
        self.messages = {}

    @staticmethod
    def _text_card(message: Dict[Text, Any]) -> Dict:

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

    @staticmethod
    def _image_card(image: Text) -> Dict:
        card = {
            "cards": [{"sections": [{"widgets": [{"image": {"imageUrl": image}}]}]}]
        }
        return card

    @staticmethod
    def _text_button_card(text: Text, buttons: List) -> Union[Dict, None]:
        hangouts_buttons = []
        for b in buttons:
            try:
                b_txt, b_pl = b["title"], b["payload"]
            except KeyError:
                logger.error(
                    "Buttons must be a list of dicts with 'title' and 'payload' as keys"
                )
                return

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

    @staticmethod
    def _combine_cards(c1: Dict, c2: Dict) -> Dict:
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
            # if any message is card, turn text message into TextParagraph card
            # and combine cards
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

    async def send_attachment(
        self, recipient_id: Text, attachment: Text, **kwargs: Any
    ):

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


# Google Hangouts input channel
class HangoutsInput(InputChannel):
    """
    Channel that uses Google Hangouts Chat API to communicate.
    """

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:

        if credentials:
            return cls(credentials.get("project_id"))  # pytype: disable=attribute-error

        return cls()

    def __init__(
        self,
        project_id: Optional[Text] = None,
        hangouts_user_added_intent_name: Optional[Text] = "/user_added",
        hangouts_room_added_intent_name: Optional[Text] = "/room_added",
        hangouts_removed_intent_name: Optional[Text] = "/bot_removed",
    ) -> None:

        self.project_id = project_id
        self.hangouts_user_added_intent_name = hangouts_user_added_intent_name
        self.hangouts_room_added_intent_name = hangouts_room_added_intent_name
        self.hangouts_user_added_intent_name = hangouts_removed_intent_name

    @classmethod
    def name(cls) -> Text:
        return CHANNEL_NAME

    @staticmethod
    def _extract_sender(req: Request) -> Text:

        if req.json["type"] == "MESSAGE":
            return req.json["message"]["sender"]["displayName"]

        return req.json["user"]["displayName"]

    # noinspection PyMethodMayBeStatic
    def _extract_message(self, req: Request) -> Text:

        if req.json["type"] == "MESSAGE":
            message = req.json["message"]["text"]

        elif req.json["type"] == "CARD_CLICKED":
            message = req.json["action"]["actionMethodName"]

        elif req.json["type"] == "ADDED_TO_SPACE":
            if self._extract_room(req) and self.hangouts_room_added_intent_name:
                message = self.hangouts_room_added_intent_name
            elif not self._extract_room(req) and self.hangouts_user_added_intent_name:
                message = self.hangouts_user_added_intent_name

        elif (
            req.json["type"] == "REMOVED_FROM_SPACE"
            and self.hangouts_user_added_intent_name
        ):
            message = self.hangouts_user_added_intent_name
        else:
            message = ""

        return message

    @staticmethod
    def _extract_room(req: Request) -> Union[Text, None]:

        if req.json["space"]["type"] == "ROOM":
            return req.json["space"]["displayName"]

    def _extract_input_channel(self) -> Text:
        return self.name()

    def _check_token(self, bot_token: Text) -> None:
        # see https://developers.google.com/hangouts/chat/how-tos/bots-develop#verifying_bot_authenticity
        try:
            token = client.verify_id_token(
                bot_token, self.project_id, cert_uri=CERT_URI
            )

            if token["iss"] != "chat@system.gserviceaccount.com":
                abort(401)
        except AppIdentityError:
            abort(401)

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Blueprint:

        custom_webhook = Blueprint("hangouts_webhook", __name__)

        @custom_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @custom_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request) -> HTTPResponse:

            if self.project_id:
                token = request.headers.get("Authorization").replace("Bearer ", "")
                self._check_token(token)

            sender_id = self._extract_sender(request)
            room_name = self._extract_room(request)
            text = self._extract_message(request)
            if text is None:
                return response.text("OK")
            input_channel = self._extract_input_channel()

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
            except Exception as e:
                logger.exception(
                    f"An exception occurred while handling user message: {e}, text: {text}"
                )

            return response.json(collector.messages)

        return custom_webhook
