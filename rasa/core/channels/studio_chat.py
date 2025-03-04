import asyncio
import json
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Text,
)

import structlog
from sanic import Sanic

from rasa.core.channels.socketio import SocketBlueprint, SocketIOInput
from rasa.hooks import hookimpl
from rasa.plugin import plugin_manager
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.trackers import EventVerbosity

if TYPE_CHECKING:
    from rasa.core.channels.channel import UserMessage
    from rasa.shared.core.trackers import DialogueStateTracker


structlogger = structlog.get_logger()


def tracker_as_dump(tracker: "DialogueStateTracker") -> str:
    """Create a dump of the tracker state."""
    from rasa.shared.core.trackers import get_trackers_for_conversation_sessions

    multiple_tracker_sessions = get_trackers_for_conversation_sessions(tracker)

    if 0 <= len(multiple_tracker_sessions) <= 1:
        last_tracker = tracker
    else:
        last_tracker = multiple_tracker_sessions[-1]

    state = last_tracker.current_state(EventVerbosity.AFTER_RESTART)
    return json.dumps(state)


def does_need_action_prediction(tracker: "DialogueStateTracker") -> bool:
    """Check if the tracker needs an action prediction."""
    return (
        len(tracker.events) == 0
        or not isinstance(tracker.events[-1], ActionExecuted)
        or tracker.events[-1].action_name != ACTION_LISTEN_NAME
    )


class StudioTrackerUpdatePlugin:
    """Plugin for publishing tracker updates a socketio channel."""

    def __init__(self, socket_channel: "StudioChatInput") -> None:
        self.socket_channel = socket_channel
        self.tasks: List[asyncio.Task] = []

    def _cancel_tasks(self) -> None:
        """Cancel all remaining tasks."""
        for task in self.tasks:
            task.cancel()
        self.tasks = []

    def _cleanup_tasks(self) -> None:
        """Remove tasks that have already completed."""
        self.tasks = [task for task in self.tasks if not task.done()]

    @hookimpl  # type: ignore[misc]
    def after_new_user_message(self, tracker: "DialogueStateTracker") -> None:
        """Triggers a tracker update notification after a new user message."""
        self.handle_tracker_update(tracker)

    @hookimpl  # type: ignore[misc]
    def after_action_executed(self, tracker: "DialogueStateTracker") -> None:
        """Triggers a tracker update notification after an action is executed."""
        self.handle_tracker_update(tracker)

    def handle_tracker_update(self, tracker: "DialogueStateTracker") -> None:
        """Handles a tracker update when triggered by a hook."""
        structlogger.info("studio_chat.after_tracker_update", tracker=tracker)
        # directly create a dump to avoid the tracker getting modified by another
        # function before it gets published (since the publishing is scheduled
        # as an async task)
        tracker_dump = tracker_as_dump(tracker)
        task = asyncio.create_task(
            self.socket_channel.publish_tracker_update(tracker.sender_id, tracker_dump)
        )
        self.tasks.append(task)
        self._cleanup_tasks()

    @hookimpl  # type: ignore[misc]
    def after_server_stop(self) -> None:
        """Cancels all remaining tasks when the server stops."""
        self._cancel_tasks()


class StudioChatInput(SocketIOInput):
    """Input channel for the communication between Rasa Studio and Rasa Pro."""

    @classmethod
    def name(cls) -> Text:
        return "studio_chat"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Creates a ``SocketIOInput`` object."""
        from rasa.core.agent import Agent

        super().__init__(*args, **kwargs)
        self.agent: Optional[Agent] = None

        self._register_tracker_update_hook()

    def _register_tracker_update_hook(self) -> None:
        plugin_manager().register(StudioTrackerUpdatePlugin(self))

    async def on_tracker_updated(self, tracker: "DialogueStateTracker") -> None:
        """Triggers a tracker update notification after a change to the tracker."""
        await self.publish_tracker_update(tracker.sender_id, tracker_as_dump(tracker))

    async def publish_tracker_update(self, sender_id: str, tracker_dump: Dict) -> None:
        """Publishes a tracker update notification to the websocket."""
        if not self.sio:
            structlogger.error("studio_chat.on_tracker_updated.sio_not_initialized")
            return
        await self.sio.emit("tracker", tracker_dump, room=sender_id)

    async def on_message_proxy(
        self,
        on_new_message: Callable[["UserMessage"], Awaitable[Any]],
        message: "UserMessage",
    ) -> None:
        """Proxies the on_new_message call to the underlying channel.

        Triggers a tracker update notification after processing the message.
        """
        await on_new_message(message)

        if not self.agent:
            structlogger.error("studio_chat.on_message_proxy.agent_not_initialized")
            return

        tracker = await self.agent.tracker_store.retrieve(message.sender_id)
        if tracker is None:
            structlogger.error("studio_chat.on_message_proxy.tracker_not_found")
            return

        await self.on_tracker_updated(tracker)

    async def handle_tracker_update(self, sid: str, data: Dict) -> None:
        from rasa.shared.core.trackers import DialogueStateTracker

        structlogger.debug(
            "studio_chat.sio.handle_tracker_update",
            sid=sid,
            sender_id=data["sender_id"],
        )
        if self.agent is None:
            structlogger.error("studio_chat.sio.agent_not_initialized")
            return None

        if not (domain := self.agent.domain):
            structlogger.error("studio_chat.sio.domain_not_initialized")
            return None

        async with self.agent.lock_store.lock(data["sender_id"]):
            tracker = DialogueStateTracker.from_dict(
                data["sender_id"], data["events"], domain.slots
            )

            # will override an existing tracker with the same id!
            await self.agent.tracker_store.save(tracker)

            processor = self.agent.processor
            if processor and does_need_action_prediction(tracker):
                output_channel = self.get_output_channel()

                await processor._run_prediction_loop(output_channel, tracker)
                await processor.run_anonymization_pipeline(tracker)
                await self.agent.tracker_store.save(tracker)

        await self.on_tracker_updated(tracker)

    def blueprint(
        self, on_new_message: Callable[["UserMessage"], Awaitable[Any]]
    ) -> SocketBlueprint:
        socket_blueprint = super().blueprint(
            partial(self.on_message_proxy, on_new_message)
        )

        if not self.sio:
            structlogger.error("studio_chat.blueprint.sio_not_initialized")
            return socket_blueprint

        @socket_blueprint.listener("after_server_start")  # type: ignore[misc]
        async def after_server_start(app: Sanic, _: asyncio.AbstractEventLoop) -> None:
            self.agent = app.ctx.agent

        @self.sio.on("update_tracker", namespace=self.namespace)
        async def on_update_tracker(sid: Text, data: Dict) -> None:
            await self.handle_tracker_update(sid, data)

        return socket_blueprint
