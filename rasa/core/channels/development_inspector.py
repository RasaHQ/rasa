import asyncio
from functools import partial
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Set,
    Text,
)
from socketio import AsyncServer

import structlog
from sanic import response
from sanic.request import Request
import websockets

from rasa.core.channels.channel import InputChannel, OutputChannel
from rasa.shared.core.trackers import EventVerbosity
from rasa.shared.utils.cli import print_info
from sanic import Blueprint, Websocket, Sanic  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from rasa.core.channels.channel import UserMessage
    from sanic.request import Request
    from sanic.response import HTTPResponse


INSPECT_TEMPLATE_PATH = "inspector/dist"

structlogger = structlog.get_logger()


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
        self.processor = None
        self.tracker_stream = TrackerStream(get_tracker=self.get_tracker_state)

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

    async def get_tracker_state(self, sender_id: str) -> str:
        """Returns the state of the tracker as a json string."""
        if not self.processor:
            return ""

        tracker = await self.processor.get_tracker(sender_id)
        state = tracker.current_state(EventVerbosity.AFTER_RESTART)
        return json.dumps(state)

    async def on_tracker_updated(self, sender_id: str) -> None:
        """Called when a tracker has been updated."""
        if self.tracker_stream:
            tracker_dump = await self.get_tracker_state(sender_id)
            await self.tracker_stream.broadcast(tracker_dump)

    async def on_message_proxy(
        self,
        on_new_message: Callable[["UserMessage"], Awaitable[Any]],
        message: "UserMessage",
    ) -> None:
        """Proxies the on_new_message call to the underlying channel.

        Triggers a tracker update notification after processing the message.
        """
        await on_new_message(message)
        await self.on_tracker_updated(message.sender_id)

    @classmethod
    async def serve_inspect_html(cls) -> "HTTPResponse":
        """Serves the inspect.html file."""
        return await response.file(cls.inspect_html_path() + "/index.html")

    def blueprint(
        self, on_new_message: Callable[["UserMessage"], Awaitable[Any]]
    ) -> "Blueprint":
        """Defines a Sanic blueprint."""
        self.sio = AsyncServer(async_mode="sanic", cors_allowed_origins=[])
        underlying_webhook: "Blueprint" = self.underlying.blueprint(
            partial(self.on_message_proxy, on_new_message)
        )
        underlying_webhook.static("/assets", self.inspect_html_path() + "/assets")

        @underlying_webhook.route("/inspect.html", methods=["GET"], name="inspect")
        async def inspect(_: "Request") -> "HTTPResponse":
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

    def __init__(self, get_tracker: Callable[[str], Awaitable[Dict[str, Any]]]) -> None:
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
                message = json.loads(message_str)
                if message.get("action") == "retrieve":
                    sender_id = message.get("sender_id")
                    if not sender_id:
                        structlogger.warning(
                            "Tried to retrieve tracker without sender_id."
                        )
                        continue
                    tracker_dump = await self.get_tracker(sender_id)
                    await self._send(ws, tracker_dump)
        finally:
            self._connected_clients.remove(ws)

    async def _send(self, ws: Websocket, message: str) -> None:
        """Sends a message to a connected client."""
        try:
            await ws.send(message)
        except websockets.exceptions.ConnectionClosed:
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
