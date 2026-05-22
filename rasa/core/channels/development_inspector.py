from __future__ import annotations

import asyncio
import time
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Text,
)
from urllib.parse import urlencode

import orjson
import structlog
from sanic import (  # type: ignore[attr-defined]
    Blueprint,
    Sanic,
    Websocket,
    exceptions,
    response,
)
from sanic.request import Request
from socketio import AsyncServer

from rasa.core.channels.channel import (
    InputChannel,
    OutputChannel,
    RuntimeAgent,
)
from rasa.shared.core.trackers import EventVerbosity
from rasa.shared.utils.cli import print_info

if TYPE_CHECKING:
    from sanic.request import Request
    from sanic.response import HTTPResponse

    from rasa.core.channels.channel import UserMessage
    from rasa.core.processor import MessageProcessor
    from rasa.shared.core.trackers import DialogueStateTracker

from rasa.hooks import hookimpl
from rasa.plugin import plugin_manager

INSPECT_NEXTGEN_TEMPLATE_PATH = "inspector-nextgen/dist"
INSPECT_LEGACY_TEMPLATE_PATH = "inspector/dist"

structlogger = structlog.get_logger()


class DevelopmentInspectorPlugin:
    """Plugin for broadcasting tracker updates to development inspector clients."""

    def __init__(self, inspector: DevelopmentInspectProxy) -> None:
        """Initializes the plugin."""
        self.inspector = inspector
        self.tasks: List[asyncio.Task] = []

    async def _cancel_tasks(self) -> None:
        """Cancel all remaining tasks."""
        for task in self.tasks:
            if not task.done():
                task.cancel()
                await task
        self.tasks = []

    def _cleanup_completed_tasks(self) -> None:
        """Remove tasks that have already completed."""
        self.tasks = [task for task in self.tasks if not task.done()]

    def _create_broadcast_task(self, tracker: DialogueStateTracker) -> None:
        """Creates a task to broadcast tracker updates."""
        task = asyncio.create_task(self.inspector.on_tracker_updated(tracker))
        self.tasks.append(task)
        self._cleanup_completed_tasks()

    @hookimpl
    def after_new_user_message(self, tracker: DialogueStateTracker) -> None:
        """Broadcasts tracker updates after a new user message."""
        self._create_broadcast_task(tracker)

    @hookimpl
    def after_action_executed(self, tracker: DialogueStateTracker) -> None:
        """Broadcasts tracker updates after an action is executed."""
        self._create_broadcast_task(tracker)

    @hookimpl
    def after_response_chunk(
        self, tracker: DialogueStateTracker, accumulated_text: str
    ) -> None:
        """Broadcasts tracker updates with streaming response text."""
        task = asyncio.create_task(
            self.inspector.on_streaming_response(tracker, accumulated_text)
        )
        self.tasks.append(task)
        self._cleanup_completed_tasks()

    @hookimpl
    async def after_server_stop(self) -> None:
        """Cancels all remaining tasks when the server stops."""
        await self._cancel_tasks()


class DevelopmentInspectProxy(InputChannel):
    """Development inspector to inspect channel communication.

    It wraps a Rasa Pro input / output providing an inspect ui showing
    the state of the conversation.
    """

    def __init__(
        self,
        underlying: InputChannel,
        is_voice: bool = False,
        server_url: Optional[Text] = None,
        is_legacy: bool = False,
    ) -> None:
        """Initializes the DevelopmentInspectProxy channel."""
        super().__init__()
        self.underlying = underlying
        self.is_voice = is_voice
        self.is_legacy = is_legacy
        self.server_url = server_url if server_url and server_url != "0.0.0.0" else None
        self.processor: Optional[MessageProcessor] = None
        self.tracker_stream = TrackerStream(get_tracker=self.get_tracker_state)
        self._turn_start_times: Dict[Text, float] = {}
        # Register the plugin to get tracker updates
        plugin_manager().register(DevelopmentInspectorPlugin(self))

    def name(self) -> Text:  # type: ignore[override]
        """Channel name."""
        return self.underlying.name()

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> "InputChannel":
        raise NotImplementedError(
            "Method `from_credentials` not supported for the inspect proxy."
        )

    def url_prefix(self) -> Text:
        return self.underlying.name()

    def get_output_channel(self) -> Optional["OutputChannel"]:
        return self.underlying.get_output_channel()

    def get_metadata(self, request: Request) -> Optional[Dict[Text, Any]]:
        return self.underlying.get_metadata(request)

    def inspect_html_path(self) -> Text:
        """Returns the path to the inspect.html file."""
        import pkg_resources

        if self.is_legacy:
            path = INSPECT_LEGACY_TEMPLATE_PATH
        else:
            path = INSPECT_NEXTGEN_TEMPLATE_PATH

        return pkg_resources.resource_filename(__name__, path)

    async def _get_tracker(self, sender_id: Text) -> DialogueStateTracker:
        """Returns the tracker for the given sender ID."""
        if not self.processor:
            structlogger.error(
                "development_inspector._get_tracker.agent_not_initialized"
            )
            raise ValueError("Agent processor is not initialized.")
        return await self.processor.get_tracker(sender_id)

    async def get_tracker_state(self, sender_id: str) -> str:
        """Returns the state of the tracker as a json string."""
        tracker = await self._get_tracker(sender_id)
        state = tracker.current_state(EventVerbosity.AFTER_RESTART)
        return orjson.dumps(state, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")

    async def on_tracker_updated(self, tracker: DialogueStateTracker) -> None:
        """Notifies all clients about tracker updates in real-time."""
        if self.tracker_stream and tracker.sender_id:
            state = tracker.current_state(EventVerbosity.AFTER_RESTART)
            tracker_dump = orjson.dumps(
                state, option=orjson.OPT_SERIALIZE_NUMPY
            ).decode("utf-8")
            await self.tracker_stream.broadcast(tracker_dump)

    async def on_streaming_response(
        self, tracker: DialogueStateTracker, accumulated_text: str
    ) -> None:
        """Notifies clients about streaming response text in real-time.

        Creates a synthetic tracker state that includes the accumulated
        streaming text as a temporary bot event.
        """
        if self.tracker_stream and tracker.sender_id:
            state = tracker.current_state(EventVerbosity.AFTER_RESTART)
            # Add a synthetic streaming bot event
            from rasa.shared.core.events import BotUttered

            streaming_event = BotUttered(
                text=accumulated_text,
                metadata={"streaming": True},
            ).as_dict()
            state["events"] = state.get("events", []) + [streaming_event]
            tracker_dump = orjson.dumps(
                state, option=orjson.OPT_SERIALIZE_NUMPY
            ).decode("utf-8")
            await self.tracker_stream.broadcast(tracker_dump)

    def _record_turn_start_time(self, sender_id: Text) -> None:
        """Records the start time of a new turn."""
        self._turn_start_times[sender_id] = time.time()

    async def on_message_proxy(
        self,
        on_new_message: Callable[["UserMessage"], Awaitable[Any]],
        message: "UserMessage",
    ) -> None:
        """Proxies the on_new_message call to the underlying channel."""
        await on_new_message(message)

    async def serve_inspect_html(self) -> HTTPResponse:
        """Serves the inspect.html file."""
        return await response.file(self.inspect_html_path() + "/index.html")

    def conversation_blueprint(self, agent: RuntimeAgent) -> "Blueprint":
        """Defines a Sanic blueprint."""
        self.sio_server = AsyncServer(async_mode="sanic", cors_allowed_origins=[])

        async def on_new_message(message: "UserMessage") -> None:
            await agent.handle_message(message)

        underlying_webhook = self.underlying.conversation_blueprint(agent) or (
            self.underlying.blueprint(partial(self.on_message_proxy, on_new_message))
        )

        if underlying_webhook is None:
            raise NotImplementedError(
                f"{self.underlying.__class__.__name__} needs to provide blueprint() "
                f"or conversation_blueprint()."
            )

        underlying_webhook.static("/assets", self.inspect_html_path() + "/assets")

        @underlying_webhook.route("/inspect.html", methods=["GET"], name="inspect")
        async def inspect(_: Request) -> HTTPResponse:
            return await self.serve_inspect_html()

        @underlying_webhook.listener("after_server_start")  # type: ignore[misc]
        async def after_server_start(app: Sanic, _: asyncio.AbstractEventLoop) -> None:
            """Prints a message after the server has started with inspect URL."""
            self.processor = app.ctx.agent.processor

            from rasa.core.channels.socketio import SocketIOInput

            if isinstance(self.underlying, SocketIOInput):

                def _cancel_background_tasks_on_disconnect(sender_id: str) -> None:
                    structlogger.debug(
                        "development_inspector.on_disconnect.cancel_background_tasks",
                        sender_id=sender_id,
                        event_info=f"Client disconnected, cancelling "
                        f"background tasks for senderID {sender_id}.",
                    )
                    app.ctx.agent.cancel_background_tasks(sender_id)

                self.underlying.on_disconnect_callback = (
                    _cancel_background_tasks_on_disconnect
                )

            # allow server_url override (e.g. behind a proxy/tunnel),
            # otherwise fall back to the server's serve_location with
            # 0.0.0.0 replaced by localhost
            serve_location = self.server_url or app.serve_location.replace(
                "0.0.0.0", "localhost"
            )

            if self.is_legacy:
                inspect_path = app.url_for(
                    f"{app.name}.{underlying_webhook.name}.inspect"
                )
                print_info(
                    f"Development inspector for channel {self.name()} is running. To "
                    f"inspect conversations, visit {serve_location}{inspect_path}"
                )
            else:
                query_string = urlencode(
                    {"projectUrl": serve_location, "channel": self.name()}
                )
                print_info(
                    f"Development inspector for channel {self.name()} is running. To "
                    f"inspect conversations, visit "
                    f"{serve_location}/webhooks/{self.name()}/inspect.html?{query_string}"
                )

        underlying_webhook.add_websocket_route(
            self.tracker_stream, "/tracker_stream", name="tracker_stream"
        )

        return underlying_webhook


class TrackerStream:
    """Stream tracker state to connected clients."""

    def __init__(self, get_tracker: Callable[[str], Awaitable[str]]) -> None:
        """Initializes the TrackerStream."""
        self._connected_clients: Set[Websocket] = set()
        self.get_tracker = get_tracker

    def __name__(self) -> str:
        """Name of the stream."""
        return "tracker_stream"

    async def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Starts the stream."""
        await self.stream(*args, **kwargs)

    async def stream(self, request: Request, ws: Websocket) -> None:
        """Handles connection of a new client."""
        self._connected_clients.add(ws)
        try:
            async for message_str in ws:
                message = orjson.loads(message_str)
                # allows frontend to request the tracker state
                # used when websocket begins
                # also used when URL changes (sender updated)
                if message.get("action") == "retrieve":
                    sender_id = message.get("sender_id")
                    if not sender_id:
                        structlogger.warning(
                            "development_insector.tracker_stream.missing_sender_id"
                        )
                        continue
                    tracker_dump = await self.get_tracker(sender_id)
                    await self._send(ws, tracker_dump)
                else:
                    structlogger.warning(
                        "development_inspector.tracker_stream.unknown_action",
                        message=message,  # no pii
                    )
        finally:
            self._connected_clients.remove(ws)

    async def _send(self, ws: Websocket, message: str) -> None:
        """Sends a message to a connected client."""
        try:
            await ws.send(message)
        except (exceptions.WebsocketClosed, asyncio.CancelledError):
            pass

    async def broadcast(self, message: str) -> None:
        """Broadcasts a message to all connected clients."""
        if not self._connected_clients:
            return
        # create & track tasks to avoid orphaned tasks on shutdown
        tasks = [
            asyncio.create_task(self._send(websocket, message))
            for websocket in self._connected_clients
        ]
        if tasks:
            _, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                await task
