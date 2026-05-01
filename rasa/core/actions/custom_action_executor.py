from __future__ import annotations

import abc
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Text

import structlog
from pydantic import BaseModel

import rasa
from rasa.core.actions.action_exceptions import DomainNotFound
from rasa.core.actions.constants import DEFAULT_SELECTIVE_DOMAIN, SELECTIVE_DOMAIN
from rasa.shared.constants import DOCS_BASE_URL
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from rasa.core.channels.channel import OutputChannel
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker


logger = logging.getLogger(__name__)
structlogger = structlog.get_logger(__name__)

# Keys that indicate a stream_chunk payload carries rich content (buttons, image,
# attachment, custom JSON, etc.) and should be delivered via send_response() rather
# than as an incremental text token via send_response_chunk().
RICH_KEYS: frozenset = frozenset(
    {"buttons", "image", "attachment", "custom", "elements", "quick_replies"}
)


async def dispatch_stream_chunk(
    output_channel: "OutputChannel",
    sender_id: str,
    payload: Dict[str, Any],
) -> None:
    """Route a stream_chunk payload to the correct OutputChannel method.

    A chunk is treated as **rich content** if its payload contains any key from
    :data:`RICH_KEYS` (``buttons``, ``image``, ``attachment``, ``custom``,
    ``elements``, ``quick_replies``).  Rich chunks are delivered immediately and
    completely via :meth:`~OutputChannel.send_response`.

    All other chunks (typically carrying only ``text``) are treated as
    incremental text tokens and forwarded via
    :meth:`~OutputChannel.send_response_chunk`.

    Args:
        output_channel: The output channel to deliver the chunk to.
        sender_id: The recipient / conversation ID.
        payload: The chunk payload dict, with protocol-internal keys
            (``event``, ``response_id``) already removed by the caller.
    """
    if RICH_KEYS & set(payload.keys()):
        await output_channel.send_response(sender_id, payload.copy())
    else:
        await output_channel.send_response_chunk(sender_id, payload.get("text", ""))


def warn_on_duplicate_streamed_responses(
    action_name: str,
    streamed_payloads: List[Dict[str, Any]],
    final_responses: List[Dict[str, Any]],
) -> None:
    """Log a warning if any response in *final_responses* was already streamed.

    An action that calls both ``stream_chunk()`` and ``utter_message()`` for the
    same content will cause the user to receive that message twice — once during
    execution and again when ``RemoteAction._utter_responses()`` processes the
    final result.  This function detects that mistake and surfaces it as a
    structured warning so action authors can fix it during development.

    Only exact payload matches trigger the warning.  Incremental text tokens
    (e.g. ``{"text": "Hello"}``) will not match a concatenated final response
    (``{"text": "Hello world"}``), so normal token-streaming does not produce
    false positives.

    Args:
        action_name: Name of the action being executed (for log context).
        streamed_payloads: Payloads already delivered to the output channel
            mid-stream (collected by the executor during ``run_streaming()``).
        final_responses: The ``responses`` list from the action's final result
            dict (``final_result["responses"]``).
    """
    for response in final_responses:
        if response in streamed_payloads:
            structlogger.warning(
                "rasa.core.actions.custom_action_executor"
                ".run_streaming.duplicate_response",
                action_name=action_name,
                event_info=(
                    f"Action '{action_name}' included a response in its final "
                    f"result that was already delivered mid-stream. The user will "
                    f"receive this message twice. Use stream_chunk() for mid-stream "
                    f"delivery or utter_message() for the final result, not both "
                    f"for the same content."
                ),
            )
            break


class ActionResultType(Enum):
    SUCCESS = "success"
    RETRY_WITH_DOMAIN = "retry_with_domain"


class ActionResult(BaseModel):
    """Result of custom action execution.

    This is used to avoid raising exceptions for expected conditions
    like missing domain (449 status code), which would otherwise be
    captured by tracing as errors.
    """

    result_type: ActionResultType
    response: Optional[Dict[Text, Any]] = None


class CustomActionExecutor(abc.ABC):
    """Interface for custom action executors.

    Provides an abstraction layer for executing custom actions
    regardless of the communication protocol.

    Set ``supports_streaming = True`` on a concrete subclass to advertise that
    it overrides :meth:`run_streaming`.  :meth:`RemoteAction._can_stream` uses
    this flag instead of ``hasattr`` so that the inherited default
    ``run_streaming`` (which raises :class:`NotImplementedError`) does not
    accidentally enable the streaming path for executors that have not
    implemented it.
    """

    supports_streaming: ClassVar[bool] = False

    @abc.abstractmethod
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        """Executes the custom action.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            The response from the execution of the custom action.
        """
        pass

    async def run_streaming(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        output_channel: "OutputChannel",
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        """Execute the custom action with server-streaming support.

        The default implementation raises :class:`NotImplementedError`.
        Concrete subclasses that support streaming must:

        1. Override this method with an actual implementation.
        2. Set ``supports_streaming = True`` at the class level so that
           :meth:`~rasa.core.actions.action.RemoteAction._can_stream`
           activates the streaming path for them.

        The ``supports_streaming`` flag is the authoritative capability signal
        used by ``_can_stream``.  Using a flag rather than ``hasattr`` means
        that this inherited stub does **not** accidentally make non-streaming
        executors appear to support streaming.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            output_channel: Channel to forward incremental chunks to.
            include_domain: If True, the domain is included in the request.

        Returns:
            The final action result dict (events + responses).

        Raises:
            NotImplementedError: When the executor does not support streaming.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_streaming(). "
            f"Set supports_streaming = True and override run_streaming() "
            f"to enable streaming for this executor."
        )

    async def run_with_result(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> ActionResult:
        """Executes the custom action and returns a result.

        This method is used to avoid raising exceptions for expected conditions
        like missing domain, which would otherwise be captured by tracing as errors.

        By default, this method calls the run method and wraps the response
        for backward compatibility.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            ActionResult containing the response and result type.
        """
        try:
            response = await self.run(tracker, domain, include_domain)
            return ActionResult(result_type=ActionResultType.SUCCESS, response=response)
        except DomainNotFound:
            return ActionResult(result_type=ActionResultType.RETRY_WITH_DOMAIN)


class NoEndpointCustomActionExecutor(CustomActionExecutor):
    """Implementation of a custom action executor when endpoint is not set.

    Used to handle the case where no endpoint is configured.

    Raises RasaException when executed.
    """

    def __init__(self, action_name: str) -> None:
        """Initializes the custom action executor.

        Args:
            action_name: The name of the custom action.
        """
        self.action_name = action_name

    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        """Executes the custom action.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            The response from the execution of the custom action.
        """
        raise RasaException(
            f"Failed to execute custom action '{self.action_name}' "
            f"because no endpoint is configured to run this "
            f"custom action. Please take a look at "
            f"the docs and set an endpoint configuration via the "
            f"--endpoints flag. "
            f"{DOCS_BASE_URL}/action-server/actions"
        )


class CustomActionRequestWriter:
    """Writes the request payload for a custom action."""

    def __init__(self, action_name: str, action_endpoint: EndpointConfig) -> None:
        """Initializes the request writer.

        Args:
            action_name: The name of the custom action.
            action_endpoint: The endpoint configuration for the action server.
        """
        self.action_name = action_name
        self.action_endpoint = action_endpoint

    def _is_selective_domain_enabled(self) -> bool:
        """Check if selective domain handling is enabled.

        Returns:
            True if selective domain handling is enabled, otherwise False.
        """
        if self.action_endpoint is None:
            return False
        return bool(
            self.action_endpoint.kwargs.get(SELECTIVE_DOMAIN, DEFAULT_SELECTIVE_DOMAIN)
        )

    def create(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> Dict[str, Any]:
        """Create the JSON payload for the action server request.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            A JSON payload to be sent to the action server.
        """
        from rasa.shared.core.trackers import EventVerbosity

        tracker_state = tracker.current_state(EventVerbosity.ALL)

        result = {
            "next_action": self.action_name,
            "sender_id": tracker.sender_id,
            "tracker": tracker_state,
            "version": rasa.__version__,
        }

        if include_domain and (
            not self._is_selective_domain_enabled()
            or domain.does_custom_action_explicitly_need_domain(self.action_name)
        ):
            result["domain"] = domain.as_dict()

        result["domain_digest"] = domain.fingerprint()

        return result


class RetryCustomActionExecutor(CustomActionExecutor):
    """Retries the execution of a custom action."""

    def __init__(self, custom_action_executor: CustomActionExecutor) -> None:
        self._custom_action_executor = custom_action_executor

    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        """Runs the wrapped custom action executor with retry logic.

        First request to the action server is made with/without the domain
        as specified by the `include_domain` parameter.

        If the action server responds with a missing domain indication,
        retries the request with the domain included.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            The response from the execution of the custom action.

        Raises:
            DomainNotFound: If the action server still requires domain after retry.
        """
        result = await self._custom_action_executor.run_with_result(
            tracker,
            domain,
            include_domain=include_domain,
        )

        if result.result_type == ActionResultType.RETRY_WITH_DOMAIN:
            # Retry with domain included
            result = await self._custom_action_executor.run_with_result(
                tracker, domain, include_domain=True
            )

            # If still missing domain after retry, raise error
            if result.result_type == ActionResultType.RETRY_WITH_DOMAIN:
                raise DomainNotFound()

        return result.response if result.response is not None else {}

    async def run_streaming(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        output_channel: "OutputChannel",
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        """Streaming variant of :meth:`run` with domain-not-found retry.

        Delegates to the wrapped executor's ``run_streaming()`` method.
        This method should only be called when ``RemoteAction._can_stream()``
        has already confirmed that the inner executor exposes ``run_streaming``
        (the check looks through this wrapper).  Calling it on a wrapper whose
        inner executor does not implement ``run_streaming`` will raise
        ``AttributeError``; that is an implementation bug in the caller, not a
        runtime-recoverable error.

        Domain retry semantics differ from the unary path: because partial
        chunks may already have been forwarded to *output_channel* before the
        error is detected, the retry re-opens the stream from scratch with
        ``include_domain=True``.  In practice the SDK validates the domain
        before emitting any chunks, so a mid-stream ``DomainNotFound`` should
        not occur.

        Args:
            tracker: Current dialogue tracker.
            domain: Domain of the assistant.
            output_channel: Channel to forward incremental chunks to.
            include_domain: Whether to include the full domain in the payload.

        Returns:
            Final action result dict (events + responses).

        Raises:
            DomainNotFound: If the domain is still missing after the retry.
        """
        try:
            return await self._custom_action_executor.run_streaming(
                tracker=tracker,
                domain=domain,
                output_channel=output_channel,
                include_domain=include_domain,
            )
        except DomainNotFound:
            # Retry once with the domain included.
            return await self._custom_action_executor.run_streaming(
                tracker=tracker,
                domain=domain,
                output_channel=output_channel,
                include_domain=True,
            )
