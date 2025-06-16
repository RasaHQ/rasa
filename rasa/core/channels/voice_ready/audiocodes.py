import asyncio
import copy
import hmac
import json
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Text,
    Tuple,
    Union,
)

import structlog
from jsonschema import ValidationError, validate
from sanic import Blueprint, response
from sanic.exceptions import NotFound, SanicException, ServerError
from sanic.request import Request
from sanic.response import HTTPResponse

from rasa.core import jobs
from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage
from rasa.core.channels.voice_ready.utils import (
    CallParameters,
    validate_voice_license_scope,
)
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.shared.core.constants import USER_INTENT_SESSION_START
from rasa.shared.exceptions import RasaException
from rasa.utils.io import remove_emojis

structlogger = structlog.get_logger()

CHANNEL_NAME = "audiocodes"
KEEP_ALIVE_SECONDS = 120
KEEP_ALIVE_EXPIRATION_FACTOR = 1.5
EVENT_START = "start"
EVENT_DTMF = "DTMF"
ACTIVITY_MESSAGE = "message"
ACTIVITY_EVENT = "event"
INFO_UNKNOWN = "unknown"
ACTIVITY_ID_KEY = "id"
CREDENTIALS_TOKEN_KEY = "token"
CREDENTIALS_USE_WEBSOCKET_KEY = "use_websocket"
CREDENTIALS_KEEP_ALIVE_KEY = "keep_alive"
CREDENTIALS_KEEP_ALIVE_EXPIRATION_FACTOR_KEY = "keep_alive_expiration_factor"
CLEANUP_INTERVAL_MINUTES = 10


def map_call_params(parameters: Dict[Text, Any]) -> CallParameters:
    """Map the Audiocodes parameters to the CallParameters dataclass."""
    return CallParameters(
        call_id=parameters.get("vaigConversationId"),
        user_phone=parameters.get("caller"),
        bot_phone=parameters.get("callee"),
        user_name=parameters.get("callerDisplayName"),
        user_host=parameters.get("callerHost"),
        bot_host=parameters.get("calleeHost"),
    )


class HttpUnauthorized(SanicException):
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
        """Update the last activity time."""
        self.last_activity: datetime = datetime.now(timezone.utc)

    @staticmethod
    def get_metadata(activity: Dict[Text, Any]) -> Optional[Dict[Text, Any]]:
        """Get metadata from the activity.

        ONLY used for activities NOT for events (see _handle_event)."""
        return activity.get("parameters")

    @staticmethod
    def _handle_event(event: Dict[Text, Any]) -> Tuple[Text, Dict[Text, Any]]:
        """Handle events and return a tuple of text and metadata.

        Args:
            event: The event to handle.

        Returns:
            Tuple of text and metadata.
            text is either /session_start or /vaig_event_<event_name>
            metadata is a dictionary with the event parameters.
        """
        structlogger.debug("audiocodes.handle.event", event_payload=event)
        if "name" not in event:
            structlogger.warning(
                "audiocodes.handle.event.no_name_key", event_payload=event
            )
            return "", {}

        if event["name"] == EVENT_START:
            text = f"{INTENT_MESSAGE_PREFIX}{USER_INTENT_SESSION_START}"
            metadata = asdict(map_call_params(event.get("parameters", {})))
        elif event["name"] == EVENT_DTMF:
            text = f"{INTENT_MESSAGE_PREFIX}vaig_event_DTMF"
            metadata = {"value": event["value"]}
        else:
            # handle other events described by Audiocodes
            # https://techdocs.audiocodes.com/voice-ai-connect/#VAIG_Combined/inactivity-detection.htm?TocPath=Bot%2520integration%257CReceiving%2520notifications%257C_____3
            text = f"{INTENT_MESSAGE_PREFIX}vaig_event_{event['name']}"
            metadata = {**event.get("parameters", {})}
            if "value" in event:
                metadata["value"] = event["value"]

        return text, metadata

    def is_active_conversation(self, now: datetime, delta: timedelta) -> bool:
        """Check if the conversation is active."""
        if now - self.last_activity > delta:
            structlogger.warning(
                "audiocodes.conversation.inactive", conversation=self.conversation_id
            )
            return False
        return True

    async def handle_activities(
        self,
        message: Dict[Text, Any],
        input_channel_name: str,
        output_channel: OutputChannel,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> None:
        """Handle activities sent by Audiocodes."""
        structlogger.debug("audiocodes.handle.activities")
        if input_channel_name == "":
            structlogger.warning(
                "audiocodes.handle.activities.empty_input_channel_name",
                event_info=(
                    f"Audiocodes input channel name is empty "
                    f"for conversation {self.conversation_id}"
                ),
            )

        for activity in message["activities"]:
            text = None
            if activity[ACTIVITY_ID_KEY] in self.activity_ids:
                structlogger.warning(
                    "audiocodes.handle.activities.duplicate_activity",
                    activity_id=activity[ACTIVITY_ID_KEY],
                    event_info=(
                        "Audiocodes might send duplicate activities if the bot has not "
                        "responded to the previous one or responded too late. Please "
                        "consider enabling the `use_websocket` option to use"
                        " Audiocodes Asynchronous API."
                    ),
                )
                continue
            self.activity_ids.append(activity[ACTIVITY_ID_KEY])
            if activity["type"] == ACTIVITY_MESSAGE:
                text = activity["text"]
                metadata = self.get_metadata(activity)
            elif activity["type"] == ACTIVITY_EVENT:
                text, metadata = self._handle_event(activity)
            else:
                structlogger.warning(
                    "audiocodes.handle.activities.unknown_activity_type",
                    activity_type=activity["type"],
                )
                continue

            if not text:
                continue
            user_msg = UserMessage(
                text=text,
                input_channel=input_channel_name,
                output_channel=output_channel,
                sender_id=self.conversation_id,
                metadata=metadata,
            )
            try:
                await on_new_message(user_msg)
            except Exception as e:  # skipcq: PYL-W0703
                structlogger.exception(
                    "audiocodes.handle.activities.failure",
                    sender_id=self.conversation_id,
                    error=e,
                    exc_info=True,
                )

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
            credentials.get(CREDENTIALS_TOKEN_KEY, ""),
            credentials.get(CREDENTIALS_USE_WEBSOCKET_KEY, True),
            credentials.get(CREDENTIALS_KEEP_ALIVE_KEY, KEEP_ALIVE_SECONDS),
            credentials.get(
                CREDENTIALS_KEEP_ALIVE_EXPIRATION_FACTOR_KEY,
                KEEP_ALIVE_EXPIRATION_FACTOR,
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
        self.background_tasks: Dict[Text, Set[asyncio.Task]] = defaultdict(set)

    def _create_task(self, conversation_id: Text, coro: Awaitable[Any]) -> asyncio.Task:
        """Create and track an asyncio task for a conversation."""
        task: asyncio.Task = asyncio.create_task(coro)
        self.background_tasks[conversation_id].add(task)
        task.add_done_callback(
            lambda t: self.background_tasks[conversation_id].discard(t)
        )
        return task

    async def _set_scheduler_job(self) -> None:
        if self.scheduler_job:
            self.scheduler_job.remove()
        self.scheduler_job = (await jobs.scheduler()).add_job(
            self.clean_old_conversations, "interval", minutes=CLEANUP_INTERVAL_MINUTES
        )

    def _check_token(self, token: Optional[Text]) -> None:
        if not token:
            structlogger.error("audiocodes.token_not_provided")
            raise HttpUnauthorized("Authentication token required.")

        if not hmac.compare_digest(str(token), str(self.token)):
            structlogger.error("audiocodes.invalid_token", invalid_token=token)
            raise HttpUnauthorized("Invalid authentication token.")

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
        structlogger.debug(
            "audiocodes.clean_old_conversations", current_number=len(self.conversations)
        )
        now = datetime.now(timezone.utc)
        delta = timedelta(seconds=self.keep_alive * self.keep_alive_expiration_factor)

        # clean up conversations
        inactive = [
            conv_id
            for conv_id, conv in self.conversations.items()
            if not conv.is_active_conversation(now, delta)
        ]

        # cancel tasks and remove conversations
        for conv_id in inactive:
            for task in self.background_tasks[conv_id]:
                task.cancel()
            self.background_tasks.pop(conv_id, None)
            self.conversations.pop(conv_id, None)

    def handle_start_conversation(self, body: Dict[Text, Any]) -> Dict[Text, Any]:
        conversation_id = body["conversation"]
        if conversation_id in self.conversations:
            raise ServerError("Conversation already exists")
        structlogger.debug(
            "audiocodes.handle_start_conversation", conversation=conversation_id
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
            structlogger.debug(
                "audiocodes.new_client_connection", conversation=conversation_id
            )
            conversation = self._get_conversation(request.token, conversation_id)
            if conversation:
                if conversation.ws:
                    structlogger.debug(
                        "audiocodes.new_client_connection.already_connected",
                        conversation=conversation_id,
                    )
                else:
                    conversation.ws = ws

            try:
                await ws.recv()
            except Exception:
                structlogger.warning(
                    "audiocodes.new_client_connection.closed",
                    conversation=conversation_id,
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
            structlogger.debug("audiocodes.on_activities", conversation=conversation_id)
            conversation = self._get_conversation(request.token, conversation_id)
            if conversation is None:
                structlogger.warning(
                    "audiocodes.on_activities.no_conversation", request=request.json
                )
                return response.json({})

            if self.use_websocket:
                # send an empty response for this request
                # activities are processed in the background
                # chat response is sent via the websocket
                ac_output: Union[WebsocketOutput, AudiocodesOutput] = WebsocketOutput(
                    conversation.ws, conversation_id
                )
                self._create_task(
                    conversation_id,
                    conversation.handle_activities(
                        request.json,
                        input_channel_name=self.name(),
                        output_channel=ac_output,
                        on_new_message=on_new_message,
                    ),
                )
                return response.json({})

            # without websockets, this becomes a blocking call
            # and the response is sent back to the Audiocodes server
            # after the activities are processed
            ac_output = AudiocodesOutput()
            await conversation.handle_activities(
                request.json,
                input_channel_name=self.name(),
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
            return await self._handle_disconnect(
                request, conversation_id, on_new_message
            )

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

    async def _handle_disconnect(
        self,
        request: Request,
        conversation_id: Text,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> HTTPResponse:
        """Triggered when the call is disconnected."""
        self._get_conversation(request.token, conversation_id)
        reason = {"reason": request.json.get("reason")}
        await on_new_message(
            UserMessage(
                text=f"{INTENT_MESSAGE_PREFIX}session_end",
                output_channel=None,
                input_channel=self.name(),
                sender_id=conversation_id,
                metadata=reason,
            )
        )
        del self.conversations[conversation_id]
        structlogger.debug(
            "audiocodes.disconnect",
            conversation=conversation_id,
            request=request.json,
        )
        return response.json({})


class AudiocodesOutput(OutputChannel):
    @classmethod
    def name(cls) -> Text:
        return CHANNEL_NAME

    def __init__(self) -> None:
        super().__init__()
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
                "timestamp": datetime.now(timezone.utc).isoformat("T")[:-3] + "Z",
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
        text = remove_emojis(text)
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

    async def hangup(self, recipient_id: Text, **kwargs: Any) -> None:
        """Indicate that the conversation should be ended."""
        await self.add_message({"type": "event", "name": "hangup"})

    async def send_text_with_buttons(
        self,
        recipient_id: str,
        text: str,
        buttons: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Uses the concise button output format for voice channels."""
        await self.send_text_with_buttons_concise(recipient_id, text, buttons, **kwargs)


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
