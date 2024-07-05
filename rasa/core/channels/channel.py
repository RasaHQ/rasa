import json
import logging
import uuid
import jwt
from sanic import Sanic, Blueprint
from sanic.request import Request
from typing import (
    Text,
    List,
    Dict,
    Any,
    Optional,
    Callable,
    Iterable,
    Awaitable,
    NoReturn,
)

from rasa.cli import utils as cli_utils
from rasa.shared.constants import DOCS_BASE_URL, DEFAULT_SENDER_ID
from rasa.core.constants import BEARER_TOKEN_PREFIX
from rasa.shared.exceptions import RasaException
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    EventVerbosity,
)

try:
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class UserMessage:
    """Represents an incoming message.

    Includes the channel the responses should be sent to.
    """

    def __init__(
        self,
        text: Optional[Text] = None,
        output_channel: Optional["OutputChannel"] = None,
        sender_id: Optional[Text] = None,
        parse_data: Optional[Dict[Text, Any]] = None,
        input_channel: Optional[Text] = None,
        message_id: Optional[Text] = None,
        metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Creates a ``UserMessage`` object.

        Args:
            text: the message text content.
            output_channel: the output channel which should be used to send
                bot responses back to the user.
            sender_id: the message owner ID.
            parse_data: rasa data about the message.
            input_channel: the name of the channel which received this message.
            message_id: ID of the message.
            metadata: additional metadata for this message.

        """
        self.text = text.strip() if text else text

        if message_id is not None:
            self.message_id = str(message_id)
        else:
            self.message_id = uuid.uuid4().hex

        if output_channel is not None:
            self.output_channel = output_channel
        else:
            self.output_channel = CollectingOutputChannel()

        if sender_id is not None:
            self.sender_id = str(sender_id)
        else:
            self.sender_id = DEFAULT_SENDER_ID

        self.input_channel = input_channel

        self.parse_data = parse_data
        self.metadata = metadata
        self.headers = kwargs.get("headers", None)

    def __repr__(self) -> Text:
        """Returns event as string for debugging."""
        return f"UserMessage(text: {self.text}, sender_id: {self.sender_id})"

    def __str__(self) -> Text:
        """Returns event as human-readable string."""
        return f"{self.__class__.__name__}({self.text})"


def register(
    input_channels: List["InputChannel"], app: Sanic, route: Optional[Text]
) -> None:
    """Registers input channel blueprints with Sanic."""

    async def handler(message: UserMessage) -> None:
        await app.ctx.agent.handle_message(message)

    for channel in input_channels:
        if route:
            p = urljoin(route, channel.url_prefix())
        else:
            p = None
        app.blueprint(channel.blueprint(handler), url_prefix=p)

    app.ctx.input_channels = input_channels


class InputChannel:
    """Input channel base class."""

    @classmethod
    def name(cls) -> Text:
        """Every input channel needs a name to identify it."""
        return cls.__name__

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> "InputChannel":
        return cls()

    def url_prefix(self) -> Text:
        return self.name()

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """Defines a Sanic blueprint.

        The blueprint will be attached to a running sanic server and handle
        incoming routes it registered for.
        """
        raise NotImplementedError("Component listener needs to provide blueprint.")

    @classmethod
    def raise_missing_credentials_exception(cls) -> NoReturn:
        raise RasaException(
            f"To use the {cls.name()} input channel, you need to "
            f"pass a credentials file using '--credentials'. "
            f"The argument should be a file path pointing to "
            f"a yml file containing the {cls.name()} authentication "
            f"information. Details in the docs: "
            f"{DOCS_BASE_URL}/messaging-and-voice-channels/"
        )

    def get_output_channel(self) -> Optional["OutputChannel"]:
        """Create ``OutputChannel`` based on information provided by the input channel.

        Implementing this function is not required. If this function returns a valid
        ``OutputChannel`` this can be used by Rasa to send bot responses to the user
        without the user initiating an interaction.

        Returns:
            ``OutputChannel`` instance or ``None`` in case creating an output channel
             only based on the information present in the ``InputChannel`` is not
             possible.
        """
        pass

    def get_metadata(self, request: Request) -> Optional[Dict[Text, Any]]:
        """Extracts additional information from the incoming request.

         Implementing this function is not required. However, it can be used to extract
         metadata from the request. The return value is passed on to the
         ``UserMessage`` object and stored in the conversation tracker.

        Args:
            request: incoming request with the message of the user

        Returns:
            Metadata which was extracted from the request.
        """
        pass


def decode_jwt(bearer_token: Text, jwt_key: Text, jwt_algorithm: Text) -> Dict:
    """Decodes a Bearer Token using the specific JWT key and algorithm.

    Args:
        bearer_token: Encoded Bearer token
        jwt_key: Public JWT key for decoding the Bearer token
        jwt_algorithm: JWT algorithm used for decoding the Bearer token

    Returns:
        `Dict` containing the decoded payload if successful or an exception
        if unsuccessful
    """
    authorization_header_value = bearer_token.replace(BEARER_TOKEN_PREFIX, "")
    return jwt.decode(authorization_header_value, jwt_key, algorithms=jwt_algorithm)


def decode_bearer_token(
    bearer_token: Text, jwt_key: Text, jwt_algorithm: Text
) -> Optional[Dict]:
    """Decodes a Bearer Token using the specific JWT key and algorithm.

    Args:
        bearer_token: Encoded Bearer token
        jwt_key: Public JWT key for decoding the Bearer token
        jwt_algorithm: JWT algorithm used for decoding the Bearer token

    Returns:
        `Dict` containing the decoded payload if successful or `None` if unsuccessful
    """
    # noinspection PyBroadException
    try:
        return decode_jwt(bearer_token, jwt_key, jwt_algorithm)
    except jwt.exceptions.InvalidSignatureError:
        logger.error("JWT public key invalid.")
    except Exception:
        logger.exception("Failed to decode bearer token.")

    return None


class OutputChannel:
    """Output channel base class.

    Provides sane implementation of the send methods
    for text only output channels.
    """

    @classmethod
    def name(cls) -> Text:
        """Every output channel needs a name to identify it."""
        return cls.__name__

    def attach_tracker_state(self, tracker: DialogueStateTracker) -> None:
        """Attaches the current tracker state to the output channel."""
        self.tracker_state = tracker.current_state(EventVerbosity.AFTER_RESTART)

    async def send_response(self, recipient_id: Text, message: Dict[Text, Any]) -> None:
        """Send a message to the client."""
        if message.get("quick_replies"):
            await self.send_quick_replies(
                recipient_id,
                message.pop("text"),
                message.pop("quick_replies"),
                **message,
            )
        elif message.get("buttons"):
            await self.send_text_with_buttons(
                recipient_id, message.pop("text"), message.pop("buttons"), **message
            )
        elif message.get("text"):
            await self.send_text_message(recipient_id, message.pop("text"), **message)

        if message.get("custom"):
            await self.send_custom_json(recipient_id, message.pop("custom"), **message)

        # if there is an image we handle it separately as an attachment
        if message.get("image"):
            await self.send_image_url(recipient_id, message.pop("image"), **message)

        if message.get("attachment"):
            await self.send_attachment(
                recipient_id, message.pop("attachment"), **message
            )

        if message.get("elements"):
            await self.send_elements(recipient_id, message.pop("elements"), **message)

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Send a message through this channel."""
        raise NotImplementedError(
            "Output channel needs to implement a send message for simple texts."
        )

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """Sends an image. Default will just post the url as a string."""
        await self.send_text_message(recipient_id, f"Image: {image}")

    async def send_attachment(
        self, recipient_id: Text, attachment: Text, **kwargs: Any
    ) -> None:
        """Sends an attachment. Default will just post as a string."""
        await self.send_text_message(recipient_id, f"Attachment: {attachment}")

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """Sends buttons to the output.

        Default implementation will just post the buttons as a string.
        """
        await self.send_text_message(recipient_id, text)
        for idx, button in enumerate(buttons):
            button_msg = cli_utils.button_to_string(button, idx)
            await self.send_text_message(recipient_id, button_msg)

    async def send_quick_replies(
        self,
        recipient_id: Text,
        text: Text,
        quick_replies: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """Sends quick replies to the output.

        Default implementation will just send as buttons.
        """
        await self.send_text_with_buttons(recipient_id, text, quick_replies)

    async def send_elements(
        self, recipient_id: Text, elements: Iterable[Dict[Text, Any]], **kwargs: Any
    ) -> None:
        """Sends elements to the output.
        Default implementation will just post the elements as a string.
        """
        for element in elements:
            element_msg = "{title} : {subtitle}".format(
                title=element.get("title", ""), subtitle=element.get("subtitle", "")
            )
            await self.send_text_with_buttons(
                recipient_id, element_msg, element.get("buttons", [])
            )

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """Sends json dict to the output channel.
        Default implementation will just post the json contents as a string.
        """
        await self.send_text_message(recipient_id, json.dumps(json_message))


class CollectingOutputChannel(OutputChannel):
    """Output channel that collects send messages in a list.

    (doesn't send them anywhere, just collects them).
    """

    def __init__(self) -> None:
        """Initialise list to collect messages."""
        self.messages: List[Dict[Text, Any]] = []

    @classmethod
    def name(cls) -> Text:
        """Name of the channel."""
        return "collector"

    @staticmethod
    def _message(
        recipient_id: Text,
        text: Optional[Text] = None,
        image: Optional[Text] = None,
        buttons: Optional[List[Dict[Text, Any]]] = None,
        attachment: Optional[Text] = None,
        custom: Optional[Dict[Text, Any]] = None,
    ) -> Dict:
        """Create a message object that will be stored."""
        obj = {
            "recipient_id": recipient_id,
            "text": text,
            "image": image,
            "buttons": buttons,
            "attachment": attachment,
            "custom": custom,
        }

        # filter out any values that are `None`
        return {k: v for k, v in obj.items() if v is not None}

    def latest_output(self) -> Optional[Dict[Text, Any]]:
        if self.messages:
            return self.messages[-1]
        else:
            return None

    async def _persist_message(self, message: Dict[Text, Any]) -> None:
        self.messages.append(message)

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        for message_part in text.strip().split("\n\n"):
            await self._persist_message(self._message(recipient_id, text=message_part))

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """Sends an image. Default will just post the url as a string."""
        await self._persist_message(self._message(recipient_id, image=image))

    async def send_attachment(
        self, recipient_id: Text, attachment: Text, **kwargs: Any
    ) -> None:
        """Sends an attachment. Default will just post as a string."""
        await self._persist_message(self._message(recipient_id, attachment=attachment))

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        await self._persist_message(
            self._message(recipient_id, text=text, buttons=buttons)
        )

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        await self._persist_message(self._message(recipient_id, custom=json_message))
