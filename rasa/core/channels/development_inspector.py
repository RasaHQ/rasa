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

from rasa.core.channels.channel import InputChannel, OutputChannel
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

INSPECT_TEMPLATE_PATH = "inspector/dist"

structlogger = structlog.get_logger()


class DevelopmentInspectorPlugin:
    """Plugin for broadcasting tracker updates to development inspector clients."""

    def __init__(self, inspector: DevelopmentInspectProxy) -> None:
        """Initializes the plugin."""
        self.inspector = inspector
        self.tasks: List[asyncio.Task] = []

    def _cancel_tasks(self) -> None:
        """Cancel all remaining tasks."""
        [task.cancel() for task in self.tasks]
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
    def after_server_stop(self) -> None:
        """Cancels all remaining tasks when the server stops."""
        self._cancel_tasks()


class DevelopmentInspectProxy(InputChannel):
    """Development inspector to inspect channel communication.

    It wraps a Rasa Pro input / output providing an inspect ui showing
    the state of the conversation.
    """

    def __init__(self, underlying: InputChannel, is_voice: bool = False) -> None:
        """Initializes the DevelopmentInspectProxy channel."""
        super().__init__()
        self.underlying = underlying
        self.is_voice = is_voice
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

    @staticmethod
    def inspect_html_path() -> Text:
        """Returns the path to the inspect.html file."""
        import pkg_resources

        return pkg_resources.resource_filename(__name__, INSPECT_TEMPLATE_PATH)

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

    @classmethod
    async def serve_inspect_html(cls) -> HTTPResponse:
        """Serves the inspect.html file."""
        return await response.file(cls.inspect_html_path() + "/index.html")

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> "Blueprint":
        """Defines a Sanic blueprint."""
        self.sio_server = AsyncServer(async_mode="sanic", cors_allowed_origins=[])
        underlying_webhook: Blueprint = self.underlying.blueprint(
            partial(self.on_message_proxy, on_new_message)
        )
        underlying_webhook.static("/assets", self.inspect_html_path() + "/assets")

        @underlying_webhook.route("/inspect.html", methods=["GET"], name="inspect")
        async def inspect(_: Request) -> HTTPResponse:
            return await self.serve_inspect_html()

        @underlying_webhook.listener("after_server_start")  # type: ignore[misc]
        async def after_server_start(app: Sanic, _: asyncio.AbstractEventLoop) -> None:
            """Prints a message after the server has started with inspect URL."""
            self.processor = app.ctx.agent.processor

            inspect_path = app.url_for(f"{app.name}.{underlying_webhook.name}.inspect")

            # replace 0.0.0.0 with localhost
            serve_location = app.serve_location.replace("0.0.0.0", "localhost")

            print_info(
                f"Development inspector for channel {self.name()} is running. To "
                f"inspect conversations, visit {serve_location}{inspect_path}"
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
        except exceptions.WebsocketClosed:
            pass

    async def broadcast(self, message: str) -> None:
        """Broadcasts a message to all connected clients."""
        if not self._connected_clients:
            return
        await asyncio.wait(
            [
                asyncio.create_task(self._send(websocket, message))
                for websocket in self._connected_clients
            ]
        )
