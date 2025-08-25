from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from asyncio import Task
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Text

import structlog
from sanic import Blueprint, Sanic, response
from sanic.request import Request
from sanic.response import HTTPResponse
from socketio import AsyncServer

import rasa.shared.utils.io
from rasa.core.channels.channel import (
    InputChannel,
    OnNewMessageType,
    OutputChannel,
    UserMessage,
)
from rasa.core.channels.constants import USER_CONVERSATION_SILENCE_TIMEOUT
from rasa.shared.core.constants import SILENCE_TIMEOUT_SLOT

logger = structlog.getLogger(__name__)


@dataclass
class SilenceTimeout:
    sender_id: str
    timeout: float


UserSilenceHandlerType = Callable[[SilenceTimeout], Awaitable[Any]]


class SocketBlueprint(Blueprint):
    """Blueprint for socketio connections."""

    def __init__(
        self, sio_server: AsyncServer, socketio_path: Text, *args: Any, **kwargs: Any
    ) -> None:
        """Creates a :class:`sanic.Blueprint` for routing socketio connections.

        :param sio_server: Instance of :class:`socketio.AsyncServer` class
        :param socketio_path: string indicating the route to accept requests on.
        """
        super().__init__(*args, **kwargs)
        self.ctx.sio_server = sio_server
        self.ctx.socketio_path = socketio_path

    def register(self, app: Sanic, options: Dict[Text, Any]) -> None:
        """Attach the Socket.IO webserver to the given Sanic instance.

        :param app: Instance of :class:`sanic.app.Sanic` class
        :param options: Options to be used while registering the
            blueprint into the app.
        """
        if self.ctx.socketio_path:
            path = self.ctx.socketio_path
        else:
            path = options.get("url_prefix", "/socket.io")
        self.ctx.sio_server.attach(app, path)
        super().register(app, options)


class SocketIOOutput(OutputChannel):
    @classmethod
    def name(cls) -> Text:
        return "socketio"

    def __init__(
        self,
        input_channel: SocketIOInput,
        sio_server: AsyncServer,
        bot_message_evt: Text,
    ) -> None:
        super().__init__()
        self._input_channel = input_channel
        self.sio_server = sio_server
        self.bot_message_evt = bot_message_evt

    async def _send_message(self, socket_id: Text, response: Any) -> None:
        """Sends a message to the recipient using the bot event."""
        await self.sio_server.emit(self.bot_message_evt, response, room=socket_id)
        if self.tracker_state and self._input_channel.enable_silence_timeout:
            silence_timeout = self.tracker_state["slots"][SILENCE_TIMEOUT_SLOT]

            logger.debug(
                "socketio_channel.silence_timeout_updated",
                sender_id=socket_id,
                silence_timeout=silence_timeout,
            )

            silence_timeout_counts = self.tracker_state["slots"][
                "consecutive_silence_timeouts"
            ]

            logger.debug(
                "socketio_channel.consecutive_silence_timeouts_updated",
                sender_id=socket_id,
                silence_timeout_counts=silence_timeout_counts,
            )

            self._input_channel.reset_silence_timeout(
                SilenceTimeout(
                    sender_id=socket_id,
                    timeout=silence_timeout,
                ),
            )

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Send a message through this channel."""
        for message_part in text.strip().split("\n\n"):
            await self._send_message(recipient_id, {"text": message_part})

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """Sends an image to the output."""
        message = {"attachment": {"type": "image", "payload": {"src": image}}}
        await self._send_message(recipient_id, message)

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        """Sends buttons to the output."""
        # split text and create a message for each text fragment
        # the `or` makes sure there is at least one message we can attach the quick
        # replies to
        message_parts = text.strip().split("\n\n") or [text]
        messages: List[Dict[Text, Any]] = [
            {"text": message, "quick_replies": []} for message in message_parts
        ]

        # attach all buttons to the last text fragment
        messages[-1]["quick_replies"] = [
            {
                "content_type": "text",
                "title": button["title"],
                "payload": button.get("payload", button["title"]),
            }
            for button in buttons
        ]

        for message in messages:
            await self._send_message(recipient_id, message)

    async def send_elements(
        self, recipient_id: Text, elements: Iterable[Dict[Text, Any]], **kwargs: Any
    ) -> None:
        """Sends elements to the output."""
        for element in elements:
            message = {
                "attachment": {
                    "type": "template",
                    "payload": {"template_type": "generic", "elements": element},
                }
            }

            await self._send_message(recipient_id, message)

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """Sends custom json to the output."""
        if "data" in json_message:
            json_message.setdefault("room", recipient_id)
            await self.sio_server.emit(self.bot_message_evt, **json_message)
        else:
            await self.sio_server.emit(
                self.bot_message_evt, json_message, room=recipient_id
            )

    async def send_attachment(  # type: ignore[override]
        self, recipient_id: Text, attachment: Dict[Text, Any], **kwargs: Any
    ) -> None:
        """Sends an attachment to the user."""
        await self._send_message(recipient_id, {"attachment": attachment})

    async def hangup(self, recipient_id: Text, **kwargs: Any) -> None:
        """Hangs up the call for the given sender."""
        # This method is not applicable for socket.io, but we implement it
        # to satisfy the OutputChannel interface.
        logger.debug(
            "socketio_channel.output.hangup",
            message=f"Hanging up call for user {recipient_id}.",
        )

        self._input_channel.disconnect(recipient_id)


class SocketIOInput(InputChannel):
    """A socket.io input channel."""

    @classmethod
    def name(cls) -> str:
        return "socketio"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        credentials = credentials or {}
        return cls(
            user_message_evt=credentials.get("user_message_evt", "user_uttered"),
            bot_message_evt=credentials.get("bot_message_evt", "bot_uttered"),
            namespace=credentials.get("namespace"),
            session_persistence=credentials.get("session_persistence", False),
            socketio_path=credentials.get("socketio_path", "/socket.io"),
            jwt_key=credentials.get("jwt_key"),
            jwt_method=credentials.get("jwt_method", "HS256"),
            metadata_key=credentials.get("metadata_key", "metadata"),
            enable_silence_timeout=credentials.get("enable_silence_timeout", False),
        )

    def __init__(
        self,
        user_message_evt: Text = "user_uttered",
        bot_message_evt: Text = "bot_uttered",
        namespace: Optional[Text] = None,
        session_persistence: bool = False,
        socketio_path: Optional[Text] = "/socket.io",
        jwt_key: Optional[Text] = None,
        jwt_method: Optional[Text] = "HS256",
        metadata_key: Optional[Text] = "metadata",
        enable_silence_timeout: bool = False,
    ):
        """Creates a ``SocketIOInput`` object."""
        self.bot_message_evt = bot_message_evt
        self.session_persistence = session_persistence
        self.user_message_evt = user_message_evt
        self.namespace = namespace
        self.socketio_path = socketio_path
        self.sio_server: Optional[AsyncServer] = None
        self.metadata_key = metadata_key
        self.enable_silence_timeout = enable_silence_timeout

        self.jwt_key = jwt_key
        self.jwt_algorithm = jwt_method
        self.on_new_message: Optional[OnNewMessageType] = None
        self.sender_silence_map: Dict[str, Task] = {}

    def get_output_channel(self) -> Optional[OutputChannel]:
        """Creates socket.io output channel object."""
        if self.sio_server is None:
            rasa.shared.utils.io.raise_warning(
                "SocketIO output channel cannot be recreated. "
                "This is expected behavior when using multiple Sanic "
                "workers or multiple Rasa Pro instances. "
                "Please use a different channel for external events in these "
                "scenarios."
            )
            return None
        return SocketIOOutput(self, self.sio_server, self.bot_message_evt)

    async def handle_session_request(
        self, sid: Text, data: Optional[Dict] = None
    ) -> None:
        """Handles session requests from the client."""
        if data is None:
            data = {}
        if "session_id" not in data or data["session_id"] is None:
            data["session_id"] = uuid.uuid4().hex
        if self.session_persistence:
            if inspect.iscoroutinefunction(self.sio_server.enter_room):  # type: ignore[union-attr]
                await self.sio_server.enter_room(sid, data["session_id"])  # type: ignore[union-attr]
            else:
                # for backwards compatibility with python-socketio < 5.10.
                # previously, this function was NOT async.
                self.sio_server.enter_room(sid, data["session_id"])  # type: ignore[union-attr]
        await self.sio_server.emit("session_confirm", data["session_id"], room=sid)  # type: ignore[union-attr]
        logger.debug(
            "socketio_channel.input.handle_session_request",
            message=f"User {sid} connected to socketIO endpoint.",
        )

    async def handle_user_message(
        self,
        sid: Text,
        data: Dict,
        on_new_message: OnNewMessageType,
    ) -> None:
        """Handles user messages received from the client."""
        output_channel = self.get_output_channel()

        if self.session_persistence:
            if not data.get("session_id"):
                rasa.shared.utils.io.raise_warning(
                    "A message without a valid session_id "
                    "was received. This message will be "
                    "ignored. Make sure to set a proper "
                    "session id using the "
                    "`session_request` socketIO event."
                )
                return
            sender_id = data["session_id"]
        else:
            sender_id = sid

        # We cancel silence timeout when a new message is received
        # to prevent sending a silence timeout message
        # if the user sends a message after the silence timeout
        self._cancel_silence_timeout(sender_id)

        metadata = data.get(self.metadata_key, {})
        if isinstance(metadata, Text):
            metadata = json.loads(metadata)

        message = UserMessage(
            text=data.get("message", ""),
            output_channel=output_channel,
            sender_id=sender_id,
            input_channel=self.name(),
            metadata=metadata,
        )
        await on_new_message(message)

    def blueprint(self, on_new_message: OnNewMessageType) -> SocketBlueprint:
        """Defines a Sanic blueprint."""
        # Workaround so that socketio works with requests from other origins.
        # https://github.com/miguelgrinberg/python-socketio/issues/205#issuecomment-493769183
        sio_server = AsyncServer(async_mode="sanic", cors_allowed_origins=[])
        socketio_webhook = SocketBlueprint(
            sio_server, self.socketio_path, "socketio_webhook", __name__
        )

        # make sio_server object static to use in get_output_channel
        self.sio_server = sio_server
        # We need to store the on_new_message callback
        # so that we can call it when a silence timeout occurs
        self.on_new_message = on_new_message

        @socketio_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @sio_server.on("connect", namespace=self.namespace)
        async def connect(sid: Text, environ: Dict, auth: Optional[Dict]) -> bool:
            if self.jwt_key:
                jwt_payload = None
                if auth and auth.get("token"):
                    jwt_payload = rasa.core.channels.channel.decode_bearer_token(
                        auth.get("token"), self.jwt_key, self.jwt_algorithm
                    )

                if jwt_payload:
                    logger.debug(
                        "socketio_channel.input.connect.jwt",
                        message=f"User {sid} connected to socketIO endpoint.",
                    )
                    # Store the chat state for this user
                    return True
                else:
                    return False
            else:
                logger.debug(
                    "socketio_channel.input.connect",
                    message=f"User {sid} connected to socketIO endpoint.",
                )
                # Store the chat state for this user
                return True

        @sio_server.on("disconnect", namespace=self.namespace)
        async def disconnect(sid: Text) -> None:
            logger.debug(
                "socketio_channel.input.disconnect",
                message=f"User {sid} disconnected from socketIO endpoint.",
            )

        @sio_server.on("session_request", namespace=self.namespace)
        async def session_request(sid: Text, data: Optional[Dict]) -> None:
            logger.debug(
                "socketio_channel.input.session_request",
                message=f"User {sid} requested a session.",
            )
            await self.handle_session_request(sid, data)

        @sio_server.on(self.user_message_evt, namespace=self.namespace)
        async def handle_message(sid: Text, data: Dict) -> None:
            logger.debug(
                "socketio_channel.input.handle_message",
                message=f"User {sid} sent a message.",
                data=data,
            )
            await self.handle_user_message(sid, data, on_new_message)

        return socketio_webhook

    def reset_silence_timeout(self, silence_timeout: SilenceTimeout) -> None:
        self._cancel_silence_timeout(silence_timeout.sender_id)

        self.sender_silence_map[silence_timeout.sender_id] = (
            asyncio.get_event_loop().create_task(
                self._monitor_silence_timeout(
                    silence_timeout,
                )
            )
        )

    def disconnect(self, sender_id: str) -> None:
        """Disconnects the user with the given sender ID."""
        self._cancel_silence_timeout(sender_id)
        if self.sio_server:
            asyncio.get_event_loop().create_task(self.sio_server.disconnect(sender_id))
            logger.debug(
                "socketio_channel.input.disconnect",
                message=f"User {sender_id} disconnected from socketIO endpoint.",
            )

    async def _monitor_silence_timeout(self, silence_timeout: SilenceTimeout) -> None:
        logger.debug(
            "socketio_channel.input.silence_timeout_watch_started",
            sender_id=silence_timeout.sender_id,
            timeout=silence_timeout.timeout,
        )
        await asyncio.sleep(silence_timeout.timeout)

        # once the timer is up, we call the handler
        # to notify the user about the silence timeout
        # this is important if monitoring trask is cancelled while handler is executed
        asyncio.get_event_loop().create_task(
            self._handle_silence_timeout(silence_timeout)
        )

        logger.debug(
            "socketio_channel.input.silence_timeout_tripped",
            sender_id=silence_timeout.sender_id,
            silence_timeout=silence_timeout.timeout,
        )

    async def _handle_silence_timeout(self, event: SilenceTimeout) -> None:
        if self.on_new_message:
            output_channel = self.get_output_channel()
            message = UserMessage(
                text=USER_CONVERSATION_SILENCE_TIMEOUT,
                output_channel=output_channel,
                sender_id=event.sender_id,
                input_channel=self.name(),
            )
            await self.on_new_message(message)

    def _cancel_silence_timeout(self, sender_id: str) -> None:
        """Cancels the silence timeout task for the given sender."""
        task = self.sender_silence_map.pop(sender_id, None)
        if task and not task.done():
            logger.debug(
                "socketio_channel.input.silence_timeout_cancelled",
                sender_id=sender_id,
            )
            task.cancel()
