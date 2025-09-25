from __future__ import annotations

import asyncio
import audioop
import base64
import json
import uuid
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
    Tuple,
    Union,
)

import structlog

from rasa.core.channels import UserMessage
from rasa.core.channels.socketio import SocketBlueprint, SocketIOInput
from rasa.core.channels.voice_ready.utils import CallParameters
from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.call_state import call_state
from rasa.core.channels.voice_stream.tts import TTSEngine
from rasa.core.channels.voice_stream.voice_channel import (
    ContinueConversationAction,
    EndConversationAction,
    NewAudioAction,
    VoiceChannelAction,
    VoiceInputChannel,
    VoiceOutputChannel,
)
from rasa.core.exceptions import AgentNotReady
from rasa.hooks import hookimpl
from rasa.plugin import plugin_manager
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.trackers import EventVerbosity

if TYPE_CHECKING:
    from sanic import Sanic, Websocket  # type: ignore[attr-defined]
    from socketio import AsyncServer

    from rasa.shared.core.trackers import DialogueStateTracker


structlogger = structlog.get_logger()


def tracker_as_dump(tracker: "DialogueStateTracker") -> Dict[str, Any]:
    """Create a dump of the tracker state."""
    from rasa.shared.core.trackers import get_trackers_for_conversation_sessions

    multiple_tracker_sessions = get_trackers_for_conversation_sessions(tracker)

    if 0 <= len(multiple_tracker_sessions) <= 1:
        last_tracker = tracker
    else:
        last_tracker = multiple_tracker_sessions[-1]

    state = last_tracker.current_state(EventVerbosity.AFTER_RESTART)
    return state


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

    @hookimpl
    def after_new_user_message(self, tracker: "DialogueStateTracker") -> None:
        """Triggers a tracker update notification after a new user message."""
        self.handle_tracker_update(tracker)

    @hookimpl
    def after_action_executed(self, tracker: "DialogueStateTracker") -> None:
        """Triggers a tracker update notification after an action is executed."""
        self.handle_tracker_update(tracker)

    def handle_tracker_update(self, tracker: "DialogueStateTracker") -> None:
        """Handles a tracker update when triggered by a hook."""
        structlogger.info(
            "studio_chat.after_tracker_update", sender_id=tracker.sender_id
        )
        # directly create a dump to avoid the tracker getting modified by another
        # function before it gets published (since the publishing is scheduled
        # as an async task)
        tracker_dump = tracker_as_dump(tracker)
        task = asyncio.create_task(
            self.socket_channel.publish_tracker_update(tracker.sender_id, tracker_dump)
        )
        self.tasks.append(task)
        self._cleanup_tasks()

    @hookimpl
    def after_server_stop(self) -> None:
        """Cancels all remaining tasks when the server stops."""
        self._cancel_tasks()


class StudioChatInput(SocketIOInput, VoiceInputChannel):
    """Input channel for the communication between Rasa Studio and Rasa Pro."""

    requires_voice_license = False

    @classmethod
    def name(cls) -> Text:
        return "studio_chat"

    def __init__(
        self,
        server_url: str,
        asr_config: Dict,
        tts_config: Dict,
        user_message_evt: Text = "user_uttered",
        bot_message_evt: Text = "bot_uttered",
        namespace: Optional[Text] = None,
        session_persistence: bool = False,
        socketio_path: Optional[Text] = "/socket.io",
        jwt_key: Optional[Text] = None,
        jwt_method: Optional[Text] = "HS256",
        metadata_key: Optional[Text] = "metadata",
        enable_silence_timeout: bool = False,
    ) -> None:
        """Creates a `StudioChatInput` object."""
        from rasa.core.agent import Agent

        self.agent: Optional[Agent] = None

        # Initialize the SocketIO input channel
        SocketIOInput.__init__(
            self,
            user_message_evt=user_message_evt,
            bot_message_evt=bot_message_evt,
            namespace=namespace,
            session_persistence=session_persistence,
            socketio_path=socketio_path,
            jwt_key=jwt_key,
            jwt_method=jwt_method,
            metadata_key=metadata_key,
            enable_silence_timeout=enable_silence_timeout,
        )

        # Initialize the Voice Input Channel
        VoiceInputChannel.__init__(
            self,
            server_url=server_url,
            asr_config=asr_config,
            tts_config=tts_config,
        )

        # Dictionaries to manage active connections and background tasks
        # `active_connections` holds the active voice sessions
        # `background_tasks` holds the asyncio tasks for voice streaming
        self.active_connections: Dict[str, SocketIOVoiceWebsocketAdapter] = {}
        self.background_tasks: Dict[str, asyncio.Task] = {}
        self._turn_start_times: Dict[Text, float] = {}

        self._register_tracker_update_hook()

    @classmethod
    def from_credentials(
        cls, credentials: Optional[Dict[Text, Any]]
    ) -> "StudioChatInput":
        """Creates a StudioChatInput channel from credentials."""
        credentials = credentials or {}

        return cls(
            # Voice specific parameters
            server_url=credentials.get("server_url", ""),
            asr_config=credentials.get("asr", {}),
            tts_config=credentials.get("tts", {}),
            # SocketIO parameters
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

    async def emit(self, event: str, data: Union[Dict, str], room: str) -> None:
        """Emits an event to the websocket."""
        if not self.sio_server:
            structlogger.error("studio_chat.emit.sio_not_initialized")
            return
        await self.sio_server.emit(event, data, room=room)

    def _register_tracker_update_hook(self) -> None:
        plugin_manager().register(StudioTrackerUpdatePlugin(self))

    async def on_tracker_updated(self, tracker: "DialogueStateTracker") -> None:
        """Triggers a tracker update notification after a change to the tracker."""
        await self.publish_tracker_update(tracker.sender_id, tracker_as_dump(tracker))

    async def publish_tracker_update(
        self, sender_id: str, tracker_dump: Dict[str, Any]
    ) -> None:
        """Publishes a tracker update notification to the websocket."""
        await self.emit("tracker", tracker_dump, room=sender_id)

    async def on_message_proxy(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        message: UserMessage,
    ) -> None:
        """Proxies the on_new_message call to the underlying channel.

        Triggers a tracker update notification after processing the message.
        """
        try:
            await on_new_message(message)
        except Exception as e:
            structlogger.exception(
                "studio_chat.on_new_message.error",
                error=str(e),
                sender_id=message.sender_id,
            )

        if not self.agent or not self.agent.is_ready():
            structlogger.error("studio_chat.on_message_proxy.agent_not_initialized")
            await self.emit_error(
                "The Rasa Pro model could not be loaded. "
                "Please check the training and deployment logs "
                "for more information.",
                message.sender_id,
                AgentNotReady("The Rasa Pro model could not be loaded."),
            )
            return

        tracker = await self.agent.tracker_store.retrieve(message.sender_id)
        if tracker is None:
            structlogger.error("studio_chat.on_message_proxy.tracker_not_found")
            return

        await self.on_tracker_updated(tracker)

    async def emit_error(self, message: str, room: str, e: Exception) -> None:
        await self.emit(
            "error",
            {
                "message": message,
                "error": str(e),
                "exception": str(type(e).__name__),
            },
            room=room,
        )

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

        tracker: Optional[DialogueStateTracker] = None

        async with self.agent.lock_store.lock(data["sender_id"]):
            try:
                tracker = DialogueStateTracker.from_dict(
                    data["sender_id"], data["events"], domain.slots
                )

                # will override an existing tracker with the same id!
                await self.agent.tracker_store.save(tracker)

                processor = self.agent.processor
                if processor and does_need_action_prediction(tracker):
                    output_channel = self.get_output_channel()

                    await processor._run_prediction_loop(output_channel, tracker)
                    await self.agent.tracker_store.save(tracker)
            except Exception as e:
                structlogger.error(
                    "studio_chat.sio.handle_tracker_update.error",
                    error=e,
                    sender_id=data["sender_id"],
                )
                await self.emit_error(
                    "An error occurred while updating the conversation.",
                    data["sender_id"],
                    e,
                )

        if not tracker:
            # in case the tracker couldn't be updated, we retrieve the prior
            # version and use that to populate the update
            tracker = await self.agent.tracker_store.get_or_create_tracker(
                data["sender_id"]
            )
        await self.on_tracker_updated(tracker)

    def channel_bytes_to_rasa_audio_bytes(self, input_bytes: bytes) -> RasaAudioBytes:
        """Voice method to convert channel bytes to RasaAudioBytes."""
        return RasaAudioBytes(audioop.lin2ulaw(input_bytes, 4))

    async def collect_call_parameters(
        self, channel_websocket: "Websocket"
    ) -> Optional[CallParameters]:
        """Voice method to collect call parameters."""
        session_id = channel_websocket.session_id
        return CallParameters(session_id, "local", "local", stream_id=session_id)

    def map_input_message(
        self,
        message: Any,
        ws: "Websocket",
    ) -> VoiceChannelAction:
        """Voice method to map websocket messages to actions."""
        if "audio" in message:
            channel_bytes = base64.b64decode(message["audio"])
            audio_bytes = self.channel_bytes_to_rasa_audio_bytes(channel_bytes)
            return NewAudioAction(audio_bytes)
        elif "marker" in message:
            if message["marker"] == call_state.latest_bot_audio_id:
                # Just finished streaming last audio bytes
                call_state.is_bot_speaking = False
                if call_state.should_hangup:
                    structlogger.debug(
                        "studio_chat.hangup", marker=call_state.latest_bot_audio_id
                    )
                    return EndConversationAction()
            else:
                call_state.is_bot_speaking = True
        return ContinueConversationAction()

    def create_output_channel(
        self, voice_websocket: "Websocket", tts_engine: TTSEngine
    ) -> VoiceOutputChannel:
        """Create a voice output channel. This is used by VoiceInputChannel."""
        return StudioVoiceOutputChannel(
            voice_websocket,
            tts_engine,
            self.tts_cache,
        )

    async def interrupt_playback(
        self, ws: Websocket, call_parameters: CallParameters
    ) -> None:
        """Interrupt the current playback of audio."""
        structlogger.debug("studio_chat.interrupt_playback")
        await ws.send(json.dumps({"interruptPlayback": True}))

    def _start_voice_session(
        self,
        session_id: str,
        sid: str,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ) -> None:
        """Create SocketIO WebSocket Adaptor & start async task for voice streaming."""
        if sid in self.active_connections:
            structlogger.warning(
                "studio_chat._start_voice_session.session_already_active",
                session_id=sid,
            )
            return

        structlogger.info(
            "studio_chat._start_voice_session.starting_session", session_id=sid
        )

        # Create a websocket adapter for this connection
        ws_adapter = SocketIOVoiceWebsocketAdapter(
            sio_server=self.sio_server,
            session_id=session_id,
            sid=sid,
            bot_message_evt=self.bot_message_evt,
        )
        self.active_connections[sid] = ws_adapter

        # Start voice streaming in an async task
        task = asyncio.create_task(
            self._handle_voice_streaming(on_new_message, ws_adapter, sid)
        )
        self.background_tasks[sid] = task
        task.add_done_callback(lambda _: self._cleanup_tasks_for_sid(sid))

    async def _handle_voice_streaming(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        ws_adapter: "Websocket",
        sid: str,
    ) -> None:
        """Handle voice streaming for a Socket.IO connection."""
        try:
            await self.run_audio_streaming(on_new_message, ws_adapter)
        except Exception as e:
            structlogger.exception(
                "studio_voice.voice_streaming.error",
                error=str(e),
                sid=sid,
            )
            if sid in self.active_connections:
                del self.active_connections[sid]

    def _cleanup_tasks_for_sid(self, sid: str) -> None:
        if sid in self.background_tasks:
            task = self.background_tasks.pop(sid)
            task.cancel()
        if sid in self.active_connections:
            del self.active_connections[sid]

    @hookimpl
    def after_server_stop(self) -> None:
        """Cleanup background tasks and active connections when the server stops."""
        structlogger.info("studio_chat.after_server_stop.cleanup")
        self.active_connections.clear()
        for task in self.background_tasks.values():
            task.cancel()

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> SocketBlueprint:
        socket_blueprint = super().blueprint(
            partial(self.on_message_proxy, on_new_message)
        )

        if not self.sio_server:
            structlogger.error("studio_chat.blueprint.sio_not_initialized")
            return socket_blueprint

        @socket_blueprint.listener("after_server_start")  # type: ignore[misc]
        async def after_server_start(
            app: "Sanic", _: asyncio.AbstractEventLoop
        ) -> None:
            if hasattr(app.ctx, "agent"):
                self.agent = app.ctx.agent

        @self.sio_server.on("disconnect", namespace=self.namespace)
        async def disconnect(sid: Text) -> None:
            structlogger.debug("studio_chat.sio.disconnect", sid=sid)
            self._cleanup_tasks_for_sid(sid)

        @self.sio_server.on("session_request", namespace=self.namespace)
        async def session_request(sid: Text, data: Optional[Dict]) -> None:
            """Overrides the base SocketIOInput session_request handler.

            Args:
              sid: ID of the session (from SocketIO).
              data:
                - session_id: Studio Chat channel is used with a Bridge Architecture
                  (Model Service's Socket Bridge), so we use session_id to remain
                  consistent across the bridge. Session ID becomes the sender_id
                  for the UserMessage.
                - is_voice: Boolean indicating if its a voice session.
            """
            # Call parent session_request handler first
            await self.handle_session_request(sid, data)

            # start a voice session if requested
            if data and data.get("is_voice", False):
                self._start_voice_session(data["session_id"], sid, on_new_message)

        @self.sio_server.on(self.user_message_evt, namespace=self.namespace)
        async def handle_message(sid: Text, data: Dict) -> None:
            """Overrides the base SocketIOInput handle_message handler."""
            # Handle voice messages
            if "audio" in data or "marker" in data:
                if sid in self.active_connections:
                    # Route audio messages to the voice adapter queue
                    ws = self.active_connections[sid]
                    ws.put_message(data)
                return

            try:
                # Handle text messages
                await self.handle_user_message(sid, data, on_new_message)
            except Exception as e:
                structlogger.exception(
                    "studio_chat.sio.handle_message.error",
                    error=str(e),
                    sid=sid,
                )
                await self.emit("error", str(e), room=sid)

        @self.sio_server.on("update_tracker", namespace=self.namespace)
        async def on_update_tracker(sid: Text, data: Dict) -> None:
            await self.handle_tracker_update(sid, data)

        return socket_blueprint


class StudioVoiceOutputChannel(VoiceOutputChannel):
    @classmethod
    def name(cls) -> str:
        return "studio_chat"

    def rasa_audio_bytes_to_channel_bytes(
        self, rasa_audio_bytes: RasaAudioBytes
    ) -> bytes:
        return audioop.ulaw2lin(rasa_audio_bytes, 4)

    def channel_bytes_to_message(self, recipient_id: str, channel_bytes: bytes) -> str:
        return json.dumps({"audio": base64.b64encode(channel_bytes).decode("utf-8")})

    def create_marker_message(self, recipient_id: str) -> Tuple[str, str]:
        message_id = uuid.uuid4().hex
        marker_data: Dict[str, Any] = {"marker": message_id}

        # Include comprehensive latency information if available
        latency_data = {
            "asr_latency_ms": call_state.asr_latency_ms,
            "rasa_processing_latency_ms": call_state.rasa_processing_latency_ms,
            "tts_first_byte_latency_ms": call_state.tts_first_byte_latency_ms,
            "tts_complete_latency_ms": call_state.tts_complete_latency_ms,
        }

        # Filter out None values from latency data
        latency_data = {k: v for k, v in latency_data.items() if v is not None}

        # Add latency data to marker if any metrics are available
        if latency_data:
            marker_data["latency"] = latency_data

        return json.dumps(marker_data), message_id


class SocketIOVoiceWebsocketAdapter:
    """Adapter to make Socket.IO work like a Sanic WebSocket for voice channels."""

    def __init__(
        self, sio_server: "AsyncServer", session_id: str, sid: str, bot_message_evt: str
    ) -> None:
        self.sio_server = sio_server
        self.bot_message_evt = bot_message_evt
        self._closed = False
        self._receive_queue: asyncio.Queue[Any] = asyncio.Queue()

        # the messages need to be emitted on room=sid
        self.sid = sid

        # used by collect_call_parameters
        # ultimately, this becomes the sender_id
        self.session_id = session_id

    @property
    def closed(self) -> bool:
        return self._closed

    async def send(self, data: Any) -> None:
        """Send data to the client."""
        if not self.closed:
            await self.sio_server.emit(self.bot_message_evt, data, room=self.sid)

    async def recv(self) -> Any:
        """Receive data from the client."""
        if self.closed:
            raise ConnectionError("WebSocket is closed")
        return await self._receive_queue.get()

    def put_message(self, message: Any) -> None:
        """Put message in the internal receive queue."""
        self._receive_queue.put_nowait(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the connection."""
        self._closed = True
        # at this point, the client should have disconnected

    def __aiter__(self) -> "SocketIOVoiceWebsocketAdapter":
        """Allow the adapter to be used in an async for loop."""
        return self

    async def __anext__(self) -> Any:
        if self.closed:
            raise StopAsyncIteration
        try:
            message = await self.recv()
            return message
        except Exception:
            raise StopAsyncIteration
