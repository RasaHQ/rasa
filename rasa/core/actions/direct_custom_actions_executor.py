import asyncio
import contextlib
from functools import lru_cache
from importlib.util import find_spec
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Optional,
    Text,
)

import structlog
from rasa_sdk.executor import ActionExecutor
from rasa_sdk.interfaces import ActionMissingDomainException

import rasa
from rasa.core.actions.action_exceptions import DomainNotFound
from rasa.core.actions.constants import STREAMING_QUEUE_MAX_SIZE
from rasa.core.actions.custom_action_executor import (
    CustomActionExecutor,
    dispatch_stream_chunk,
    warn_on_duplicate_streamed_responses,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from rasa.core.channels.channel import OutputChannel

structlogger = structlog.get_logger(__name__)


class DirectCustomActionExecutor(CustomActionExecutor):
    _actions_module_registered: ClassVar[bool] = False
    supports_streaming: ClassVar[bool] = True

    def __init__(self, action_name: str, action_endpoint: EndpointConfig):
        """Initializes the direct custom action executor.

        Args:
            action_name: Name of the custom action.
            action_endpoint: The endpoint to execute custom actions.
        """
        self.action_name = action_name
        self.action_endpoint = action_endpoint
        self.action_executor = self._create_action_executor()

    @staticmethod
    @lru_cache(maxsize=1)
    def _create_action_executor() -> ActionExecutor:
        """Creates and returns a cached ActionExecutor instance.

        Returns:
            ActionExecutor: The cached ActionExecutor instance.
        """
        return ActionExecutor()

    def register_actions_from_a_module(self) -> None:
        """Registers actions from the specified module if not already registered.

        This method checks if the actions module has already been registered to prevent
        duplicate registrations. If not registered, it attempts to register the actions
        module specified in the action endpoint configuration. If the module does not
        exist, it raises a RasaException.

        Raises:
            RasaException: If the actions module specified does not exist.
        """
        if DirectCustomActionExecutor._actions_module_registered:
            return

        module_name = self.action_endpoint.actions_module
        if not find_spec(module_name):
            raise RasaException(
                f"You've provided the custom actions module '{module_name}' "
                f"to run directly by the rasa server, however this module does "
                f"not exist. Please check for typos in your `endpoints.yml` file."
            )

        self.action_executor.register_package(module_name)
        DirectCustomActionExecutor._actions_module_registered = True

    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        """Executes the custom action directly.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            The response from the execution of the custom action.

        Raises:
            RasaException: If the actions module specified does not exist.
        """
        structlogger.debug(
            "action.direct_custom_action_executor.run",
            action_name=self.action_name,
        )

        # Register actions module if not already registered.
        # This is done here instead of __init__ to allow proper
        # exception handling and avoid hanging conversations.
        self.register_actions_from_a_module()
        self.action_executor.reload()

        tracker_state = tracker.current_state(EventVerbosity.ALL)
        action_call = {
            "next_action": self.action_name,
            "sender_id": tracker.sender_id,
            "tracker": tracker_state,
            "version": rasa.__version__,
        }

        if domain:
            action_call["domain"] = domain.as_dict()

        result = await self.action_executor.run(action_call)
        return result.model_dump() if result else {}

    async def run_streaming(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        output_channel: "OutputChannel",
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        """Executes the custom action directly with streaming output.

        Runs the action via :meth:`ActionExecutor.run_streaming`, consuming
        chunk events from the sink queue as they are produced.  Each text
        chunk is forwarded to *output_channel* in real time so the end user
        receives tokens incrementally.  Returns the same final result dict as
        :meth:`run` so that callers can use either method interchangeably.

        Chunk events emitted by the SDK:

        * ``stream_start``  -- forwarded as
          :meth:`~OutputChannel.send_response_chunk_start`.
        * ``stream_chunk``  -- routed by
          :func:`~rasa.core.actions.custom_action_executor.dispatch_stream_chunk`:
          rich payloads (buttons, image, attachment, custom, elements) are
          delivered via :meth:`~OutputChannel.send_response`; plain text
          payloads are forwarded as incremental tokens via
          :meth:`~OutputChannel.send_response_chunk`.
        * ``stream_end``    -- forwarded as
          :meth:`~OutputChannel.send_response_chunk_end`.
        * ``stream_done``   -- terminal sentinel carrying the full result;
          not forwarded to the channel.

        Note: when the SDK sink is attached, ``stream_end`` does **not** call
        ``utter_message`` on the dispatcher, so the streamed text does not
        appear in ``responses``.  No deduplication is therefore needed in
        ``RemoteAction._utter_responses``.

        If the action raises an exception (e.g.
        :class:`~rasa_sdk.interfaces.ActionExecutionRejection`) the background
        task fails before placing ``stream_done`` on the queue.  A done-callback
        puts an ``_error`` sentinel on the queue so the consumer loop always
        terminates and the original exception is re-raised.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            output_channel: The output channel to forward streaming chunks to.
            include_domain: Ignored for the direct executor — the domain is always
                included because the SDK runs in-process and there is no network
                payload overhead to avoid.

        Returns:
            The final ``{events, responses}`` dict from the completed action.

        Raises:
            RasaException: If the actions module specified does not exist.
        """
        structlogger.debug(
            "action.direct_custom_action_executor.run_streaming",
            action_name=self.action_name,
        )

        self.register_actions_from_a_module()
        self.action_executor.reload()

        tracker_state = tracker.current_state(EventVerbosity.ALL)
        action_call = {
            "next_action": self.action_name,
            "sender_id": tracker.sender_id,
            "tracker": tracker_state,
            "version": rasa.__version__,
        }

        if domain:
            action_call["domain"] = domain.as_dict()

        sink: asyncio.Queue = asyncio.Queue(maxsize=STREAMING_QUEUE_MAX_SIZE)
        executor_task = asyncio.create_task(
            self.action_executor.run_streaming(action_call, sink=sink)
        )

        # If the task fails before placing stream_done (e.g. ActionExecutionRejection),
        # put an error sentinel so the consumer loop below always terminates.
        async def _put_error_on_sink(exc: BaseException) -> None:
            await sink.put({"event": "_error", "exc": exc})

        # Keeps strong references to fire-and-forget tasks so they are not
        # garbage-collected before they complete (required by RUF006).
        _background_tasks: set = set()

        def _on_task_done(task: "asyncio.Task[Any]") -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is None:
                return
            try:
                sink.put_nowait({"event": "_error", "exc": exc})
            except asyncio.QueueFull:
                bg = asyncio.create_task(_put_error_on_sink(exc))
                _background_tasks.add(bg)
                bg.add_done_callback(_background_tasks.discard)

        executor_task.add_done_callback(_on_task_done)

        try:
            final_result = await self._consume_stream_events(
                sink, output_channel, tracker.sender_id
            )
            await executor_task
            return final_result
        except BaseException as exc:
            # Cancel the background SDK task so it does not block forever on a
            # full sink queue after the consumer has already exited.
            executor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await executor_task
            raise exc

    def _raise_for_sdk_error(self, exc: BaseException) -> None:
        """Re-raise *exc*, translating domain-missing errors to :class:`DomainNotFound`.

        Args:
            exc: The exception placed on the sink by the SDK background task.

        Raises:
            DomainNotFound: When *exc* is an :class:`ActionMissingDomainException`.
        """
        if isinstance(exc, ActionMissingDomainException):
            raise DomainNotFound() from exc
        raise exc

    async def _consume_stream_events(
        self,
        sink: "asyncio.Queue[Dict[Text, Any]]",
        output_channel: "OutputChannel",
        sender_id: str,
    ) -> Dict[Text, Any]:
        """Consume SDK stream events from *sink* until ``stream_done`` is received.

        Processes each event type in order, forwarding chunks to *output_channel*
        in real time.  Raises the original exception if an ``_error`` sentinel is
        received.

        Args:
            sink: The queue populated by the SDK ``ActionExecutor.run_streaming``
                background task.
            output_channel: The output channel to forward streaming chunks to.
            sender_id: The sender ID of the current conversation.

        Returns:
            The final ``{events, responses}`` dict from the completed action.
        """
        final_result: Dict[Text, Any] = {}
        streamed_payloads: list[Dict[Text, Any]] = []

        # Tri-state session tracker:
        #   False – stream definitely not started (initial value)
        #   None  – chunk_start is being attempted (set before the call)
        #   True  – chunk_start succeeded and session is open
        # Cleanup fires whenever the state is not False, covering both a
        # failed chunk_start attempt and a mid-stream channel error.
        chunk_session_open: Optional[bool] = False

        try:
            while True:
                chunk: Dict[Text, Any] = await sink.get()
                event = chunk.get("event")

                if event == "stream_start":
                    structlogger.debug(
                        "action.direct_custom_action_executor.run_streaming.stream_start",
                        action_name=self.action_name,
                    )
                    chunk_session_open = None  # about to attempt
                    await output_channel.send_response_chunk_start(sender_id)
                    chunk_session_open = True  # confirmed
                elif event == "stream_chunk":
                    payload = {
                        key: value
                        for key, value in chunk.items()
                        if key not in ("event", "response_id")
                    }
                    structlogger.debug(
                        "action.direct_custom_action_executor.run_streaming.chunk",
                        action_name=self.action_name,
                    )
                    await dispatch_stream_chunk(output_channel, sender_id, payload)
                    streamed_payloads.append(payload)
                elif event == "stream_end":
                    structlogger.debug(
                        "action.direct_custom_action_executor.run_streaming.stream_end",
                        action_name=self.action_name,
                    )
                    await output_channel.send_response_chunk_end(sender_id)
                    chunk_session_open = False  # confirmed ended
                elif event == "stream_done":
                    sdk_result = chunk.get("result")
                    final_result = sdk_result.model_dump() if sdk_result else {}
                    warn_on_duplicate_streamed_responses(
                        action_name=self.action_name,
                        streamed_payloads=streamed_payloads,
                        final_responses=final_result.get("responses", []),
                    )
                    break
                elif event == "_error":
                    self._raise_for_sdk_error(chunk["exc"])
        except BaseException as exc:
            # Close the chunk session whenever it was started or attempted,
            # so the client is not left waiting indefinitely.
            if chunk_session_open is not False:
                with contextlib.suppress(BaseException):
                    await output_channel.send_response_chunk_end(sender_id)
            raise exc

        return final_result
