import copy
import datetime
import json
import logging
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Text, Union

import structlog
from jsonschema import ValidationError, validate
from rasa.core import jobs
from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage
from rasa.core.channels.voice_aware.utils import validate_voice_license_scope
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.shared.exceptions import RasaException
from sanic import Blueprint, response
from sanic.exceptions import NotFound, SanicException, ServerError
from sanic.request import Request
from sanic.response import HTTPResponse


logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()

CHANNEL_NAME = "audiocodes"
KEEP_ALIVE_SECONDS = 120
KEEP_ALIVE_EXPIRATION_FACTOR = 1.5


class Unauthorized(SanicException):
    """**Status**: 401 Not Authorized."""

    status_code = 401
    quiet = True


class Conversation:
    def __init__(self, conversation_id: Text):
        self.activity_ids: List[Text] = []
        self.ws: Any = None
        self.conversation_id: Text = conversation_id
        self.update()

    def update(self) -> None:
        self.last_activity: datetime.datetime = datetime.datetime.utcnow()

    @staticmethod
    def get_metadata(activity: Dict[Text, Any]) -> Optional[Dict[Text, Any]]:
        return activity.get("parameters")

    @staticmethod
    def _handle_event(event: Dict[Text, Any]) -> Text:
        text = f'{INTENT_MESSAGE_PREFIX}vaig_event_{event["name"]}'
        event_params = {}
        if "parameters" in event:
            event_params.update(event["parameters"])
        if "value" in event:
            event_params.update({"value": event["value"]})
        if len(event_params) > 0:
            text += json.dumps(event_params)
        return text

    def is_active_conversation(
        self, now: datetime.datetime, delta: datetime.timedelta
    ) -> bool:
        if now - self.last_activity > delta:
            logger.warning(
                f"Conversation {self.conversation_id} is invalid due to inactivity"
            )
            return False
        return True

    async def handle_activities(
        self,
        message: Dict[Text, Any],
        output_channel: OutputChannel,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> None:
        logger.debug("(handle_activities) --- Activities:")
        for activity in message["activities"]:
            text = None
            if activity["id"] in self.activity_ids:
                logger.warning(
                    "Got activity that already handled. Activity ID:"
                    f' {activity["id"]}'
                )
                continue
            self.activity_ids.append(activity["id"])
            if activity["type"] == "message":
                text = activity["text"]
            elif activity["type"] == "event":
                text = self._handle_event(activity)
            else:
                logger.warning(
                    "Received an activity from audiocodes that we can not "
                    f"handle. Activity: {activity}"
                )
            if not text:
                continue
            metadata = self.get_metadata(activity)
            user_msg = UserMessage(
                text=text,
                output_channel=output_channel,
                sender_id=self.conversation_id,
                metadata=metadata,
            )
            try:
                await on_new_message(user_msg)
            except Exception as e:  # skipcq: PYL-W0703
                if isinstance(user_msg.text, dict):
                    anonymized_info = json.dumps(user_msg.text)
                elif isinstance(user_msg.text, str):
                    anonymized_info = user_msg.text
                else:
                    anonymized_info = "unknown"

                structlogger.exception(
                    "audiocodes.handle.activities.failure",
                    user_message=copy.deepcopy(anonymized_info),
                )
                logger.debug(e, exc_info=True)

                await output_channel.send_custom_json(
                    self.conversation_id,
                    {
                        "type": "event",
                        "name": "hangup",
                        "text": "An error occurred while handling the last message.",
                    },
                )


class AudiocodesInput(InputChannel):
    @classmethod
    def name(cls) -> Text:
        return CHANNEL_NAME

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()
        schema = {
            "type": "object",
            "required": ["token"],
            "properties": {
                "keep_alive": {"type": "number"},
                "keep_alive_expiration_factor": {
                    "type": "number",
                    "minimum": 1,
                },
                "use_websocket": {"type": "boolean"},
                "token": {"type": "string"},
            },
        }

        try:
            validate(instance=credentials, schema=schema)
        except ValidationError as e:
            raise RasaException(f"Invalid credentials: {e.message}")

        return cls(
            credentials.get("token", ""),
            credentials.get("use_websocket", True),
            credentials.get("keep_alive", KEEP_ALIVE_SECONDS),
            credentials.get(
                "keep_alive_expiration_factor", KEEP_ALIVE_EXPIRATION_FACTOR
            ),
        )

    def __init__(
        self,
        token: Text,
        use_websocket: bool,
        keep_alive: int,
        keep_alive_expiration_factor: float,
    ) -> None:
        validate_voice_license_scope()
        self.conversations: Dict[Text, Conversation] = {}
        self.token = token
        self.use_websocket = use_websocket
        self.scheduler_job = None
        self.keep_alive = keep_alive
        self.keep_alive_expiration_factor = keep_alive_expiration_factor

    async def _set_scheduler_job(self) -> None:
        if self.scheduler_job:
            self.scheduler_job.remove()
        self.scheduler_job = (await jobs.scheduler()).add_job(
            self.clean_old_conversations, "interval", minutes=10
        )

    def _check_token(self, token: Optional[Text]) -> None:
        if not token:
            raise Unauthorized("Authentication token required.")

    def _get_conversation(
        self, token: Optional[Text], conversation_id: Text
    ) -> Conversation:
        self._check_token(token)
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            raise NotFound("Conversation not found")
        conversation.update()
        return conversation

    def clean_old_conversations(self) -> None:
        logger.debug(
            "Performing clean old conversations, current number:"
            f" {len(self.conversations)}"
        )
        now = datetime.datetime.utcnow()
        delta = datetime.timedelta(
            seconds=self.keep_alive * self.keep_alive_expiration_factor
        )
        self.conversations = {
            k: v
            for k, v in self.conversations.items()
            if v.is_active_conversation(now, delta)
        }

    def handle_start_conversation(self, body: Dict[Text, Any]) -> Dict[Text, Any]:
        conversation_id = body["conversation"]
        if conversation_id in self.conversations:
            raise ServerError("Conversation already exists")
        logger.debug(
            "(handle_start_conversation) --- New Conversation has arrived."
            f" Conversation: {conversation_id}"
        )
        self.conversations[conversation_id] = Conversation(conversation_id)
        urls = {
            "activitiesURL": f"conversation/{conversation_id}/activities",
            "disconnectURL": f"conversation/{conversation_id}/disconnect",
            "refreshURL": f"conversation/{conversation_id}/keepalive",
            "expiresSeconds": self.keep_alive,
        }
        if self.use_websocket:
            urls.update({"websocketURL": f"conversation/{conversation_id}/websocket"})
        return urls

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        ac_webhook = Blueprint("ac_webhook", __name__)

        @ac_webhook.websocket("/conversation/<conversation_id>/websocket")  # type: ignore[misc]
        async def new_client_connection(
            request: Request, ws: Any, conversation_id: Text
        ) -> None:
            """Triggered on new websocket connection."""
            if self.use_websocket is False:
                raise ConnectionRefusedError("websocket is unavailable")
            logger.debug(
                "(new_client_connection) --- New client is trying to connect."
                f" Conversation: {conversation_id}"
            )
            conversation = self._get_conversation(request.token, conversation_id)
            if conversation:
                if conversation.ws:
                    logger.debug(
                        "(new_client_connection) --- The client was already connected."
                        f" Conversation: {conversation_id}"
                    )
                else:
                    conversation.ws = ws

            try:
                await ws.recv()
            except Exception:
                logger.debug(
                    (
                        "(new_client_connection) --- Websocket was closed by client: "
                        f"{conversation_id}"
                    )
                )
                if conversation:
                    conversation.ws = None

        @ac_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            """Server health route."""
            return response.json({"status": "ok"})

        @ac_webhook.route("/webhook", methods=["GET", "POST"])
        async def receive(request: Request) -> HTTPResponse:
            """Triggered on new conversations.

            Example of payload: {"conversation": <conversation_id>, id, timestamp}.
            """
            if not self.scheduler_job:
                await self._set_scheduler_job()
            self._check_token(request.token)
            if request.method == "GET":
                return response.json({"type": "ac-bot-api", "success": True})
            return response.json(self.handle_start_conversation(request.json))

        @ac_webhook.route(
            "/conversation/<conversation_id>/activities", methods=["POST"]
        )
        async def on_activities(
            request: Request, conversation_id: Text
        ) -> HTTPResponse:
            """Process activities sent by Audiocodes.

            Activities can be:
            - Messages
            - Notifications (e.g: status of success/failure of a transfer)
            - See official documentation for more examples:
            https://techdocs.audiocodes.com/voice-ai-connect/#VAIG_Combined/sending-activities.htm
            Example of payload:
            {"conversation": <conversation_id>, "activities": List[Activity]}.
            """
            logger.debug(
                "(on_activities) --- New activities from the user. Conversation: "
                f"{conversation_id}"
            )
            conversation = self._get_conversation(request.token, conversation_id)
            if conversation is None:
                return response.json({})
            elif conversation.ws:
                ac_output: Union[WebsocketOutput, AudiocodesOutput] = WebsocketOutput(
                    conversation.ws, conversation_id
                )
                await conversation.handle_activities(
                    request.json,
                    output_channel=ac_output,
                    on_new_message=on_new_message,
                )
                return response.json({})
            else:
                # handle non websocket case where messages get returned in json
                ac_output = AudiocodesOutput()
                await conversation.handle_activities(
                    request.json,
                    output_channel=ac_output,
                    on_new_message=on_new_message,
                )
                return response.json(
                    {
                        "conversation": conversation_id,
                        "activities": ac_output.messages,
                    }
                )

        @ac_webhook.route(
            "/conversation/<conversation_id>/disconnect", methods=["POST"]
        )
        async def disconnect(request: Request, conversation_id: Text) -> HTTPResponse:
            """Triggered when the call is disconnected.

            Example of payload:
            {"conversation": <conversation_id>, "reason": Optional[Text]}.
            """
            self._get_conversation(request.token, conversation_id)
            reason = json.dumps({"reason": request.json.get("reason")})
            await on_new_message(
                UserMessage(
                    text=f"{INTENT_MESSAGE_PREFIX}vaig_event_end{reason}",
                    output_channel=None,
                    sender_id=conversation_id,
                )
            )
            del self.conversations[conversation_id]
            logger.debug("(disconnect) --- Conversation was deleted")
            return response.json({})

        @ac_webhook.route("/conversation/<conversation_id>/keepalive", methods=["POST"])
        async def keepalive(request: Request, conversation_id: Text) -> HTTPResponse:
            """Triggered for keeping the connection alive.

            Invoked by VoiceAI Connect every `keep_alive`
            seconds to verify the status of the conversation
            Example of payload: # {"conversation": <conversation_id>}.
            """
            self._get_conversation(request.token, conversation_id)
            return response.json({})

        return ac_webhook


class AudiocodesOutput(OutputChannel):
    @classmethod
    def name(cls) -> Text:
        return CHANNEL_NAME

    def __init__(self) -> None:
        self.messages: List[Dict] = []

    async def add_message(self, message: Dict) -> None:
        """Add metadata and add message.

        Message is added to the list of
        activities to be sent to the VoiceAI Connect server.
        """
        structlogger.debug(
            "audiocodes.add.message",
            class_name=self.__class__.__name__,
            message=copy.deepcopy(message.get("text", "")),
        )
        message.update(
            {
                "timestamp": datetime.datetime.utcnow().isoformat("T")[:-3] + "Z",
                "id": str(uuid.uuid4()),
            }
        )
        await self.do_add_message(message)

    async def do_add_message(self, message: Dict) -> None:
        """Send a list of activities to the VoiceAI Connect server."""
        self.messages.append(message)

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Send a text message."""
        await self.add_message({"type": "message", "text": text})

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        raise RasaException("Images are not supported by this channel")

    async def send_attachment(
        self, recipient_id: Text, attachment: Text, **kwargs: Any
    ) -> None:
        raise RasaException("Attachments are not supported by this channel")

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """Send an activity."""
        await self.add_message(json_message)


class WebsocketOutput(AudiocodesOutput):
    def __init__(self, ws: Any, conversation_id: Text) -> None:
        AudiocodesOutput.__init__(self)
        self.ws = ws
        self.conversation_id = conversation_id

    async def do_add_message(self, message: Dict) -> None:
        """Send a list of activities to the VoiceAI Connect server."""
        await self.ws.send(
            json.dumps(
                {
                    "conversation": self.conversation_id,
                    "activities": [message],
                }
            )
        )
