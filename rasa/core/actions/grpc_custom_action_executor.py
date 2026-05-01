import contextlib
import json
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import grpc
import grpc.aio
import structlog
from google.protobuf.json_format import MessageToDict, Parse, ParseDict, ParseError
from rasa_sdk.grpc_errors import ResourceNotFound, ResourceNotFoundType
from rasa_sdk.grpc_py import action_webhook_pb2, action_webhook_pb2_grpc

from rasa.core.actions.action_exceptions import DomainNotFound
from rasa.core.actions.constants import (
    SSL_CLIENT_CERT_FIELD,
    SSL_CLIENT_KEY_FIELD,
    STREAM_ERROR_MISSING_DOMAIN,
)
from rasa.core.actions.custom_action_executor import (
    ActionResult,
    ActionResultType,
    CustomActionExecutor,
    CustomActionRequestWriter,
    dispatch_stream_chunk,
    warn_on_duplicate_streamed_responses,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.io import file_as_bytes
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from rasa.core.channels.channel import OutputChannel
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger(__name__)


class GRPCCustomActionExecutor(CustomActionExecutor):
    """gRPC-based implementation of the CustomActionExecutor.

    Executes custom actions by making gRPC requests to the action endpoint.
    """

    supports_streaming: ClassVar[bool] = True

    def __init__(
        self,
        action_name: str,
        action_endpoint: EndpointConfig,
    ) -> None:
        """Initializes the gRPC custom action executor.

        Args:
            action_name: Name of the custom action.
            action_endpoint: Endpoint configuration of the custom action.
        """
        self.action_name = action_name
        self.request_writer = CustomActionRequestWriter(action_name, action_endpoint)
        self.action_endpoint = action_endpoint

        parsed_url = urlparse(self.action_endpoint.url)
        self.request_url = parsed_url.netloc

        self.cert_ca = (
            file_as_bytes(self.action_endpoint.cafile)
            if self.action_endpoint.cafile
            else None
        )

        self.client_cert = None
        self.client_key = None

        client_cert_file = self.action_endpoint.kwargs.get(SSL_CLIENT_CERT_FIELD)
        client_key_file = self.action_endpoint.kwargs.get(SSL_CLIENT_KEY_FIELD)
        if client_cert_file and client_key_file:
            self.client_cert = file_as_bytes(client_cert_file)
            self.client_key = file_as_bytes(client_key_file)
        elif client_key_file and not client_cert_file:
            structlogger.error(
                f"rasa.core.actions.grpc_custom_action_executor.{SSL_CLIENT_CERT_FIELD}_missing",
                event_info=(
                    f"Client key file '{SSL_CLIENT_KEY_FIELD}' is provided but "
                    f"client certificate file '{SSL_CLIENT_CERT_FIELD}' "
                    f"is not provided in the endpoint configuration. "
                    f"Both fields are required for client TLS authentication."
                    f"Continuing without client TLS authentication."
                ),
            )
        elif client_cert_file and not client_key_file:
            structlogger.error(
                f"rasa.core.actions.grpc_custom_action_executor.{SSL_CLIENT_KEY_FIELD}_missing",
                event_info=(
                    f"Client certificate file '{SSL_CLIENT_CERT_FIELD}' "
                    f" is provided but client key file '{SSL_CLIENT_KEY_FIELD}'"
                    f" is not provided in the endpoint configuration. "
                    f"Both fields are required for client TLS authentication."
                    f"Continuing without client TLS authentication."
                ),
            )

    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> Dict[str, Any]:
        """Execute the custom action using a gRPC request.

        Args:
            tracker: Tracker for the current conversation.
            domain: Domain of the assistant.
            include_domain: If True, the domain is included in the request.

        Returns:
            Response from the action server.
            Returns empty dict if domain is missing.

        Raises:
            RasaException: If an error occurs while making the gRPC request
                (other than missing domain).
        """
        result = await self.run_with_result(tracker, domain, include_domain)

        # Return empty dict for retry cases to avoid raising exceptions
        # RetryCustomActionExecutor will handle the retry logic
        if result.result_type == ActionResultType.RETRY_WITH_DOMAIN:
            return {}

        return result.response if result.response is not None else {}

    async def run_with_result(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> ActionResult:
        """Execute the custom action and return an ActionResult.

        This method avoids raising DomainNotFound exception for missing domain,
        instead returning an ActionResult with RETRY_WITH_DOMAIN type.
        This prevents tracing from capturing this expected condition as an error.

        Args:
            tracker: Tracker for the current conversation.
            domain: Domain of the assistant.
            include_domain: If True, the domain is included in the request.

        Returns:
            ActionResult containing the response and result type.
        """
        request = self._create_payload(
            tracker=tracker, domain=domain, include_domain=include_domain
        )

        try:
            response = await self._request(request)
            return ActionResult(result_type=ActionResultType.SUCCESS, response=response)
        except DomainNotFound:
            # Return retry result instead of raising DomainNotFound
            return ActionResult(result_type=ActionResultType.RETRY_WITH_DOMAIN)

    async def _request(
        self,
        request: action_webhook_pb2.WebhookRequest,
    ) -> Dict[str, Any]:
        """Perform a single async gRPC request to the action server.

        Uses an async channel via ``grpc.aio`` so the call does not block the
        event loop.  The channel is opened and closed within this method using
        an async context manager.

        Args:
            request: gRPC Request to be sent to the action server.

        Returns:
            Response from the action server as a plain dictionary.

        Raises:
            DomainNotFound: When the server signals the domain is missing.
            RasaException: For all other gRPC errors.
        """
        metadata = self._build_metadata()
        async with self._create_channel() as channel:
            client = self._create_grpc_client(channel)
            try:
                response = await client.Webhook(request, metadata=metadata)
                return MessageToDict(response)
            except grpc.aio.AioRpcError as rpc_error:
                # grpc.aio.AioRpcError always carries code() and details();
                # the grpc.Call isinstance check used by the sync client is
                # not needed here.
                status_code = rpc_error.code()
                details = rpc_error.details()
                if status_code is not grpc.StatusCode.NOT_FOUND:
                    raise RasaException(
                        f"Failed to execute custom action '{self.action_name}'. "
                        f"Error: {details}"
                    )

                resource_not_found_error = self._parse_not_found_error(details)
                if resource_not_found_error is None:
                    raise RasaException(
                        f"Failed to execute custom action '{self.action_name}'. "
                        f"Error: {details}"
                    )
                if (
                    resource_not_found_error.resource_type
                    == ResourceNotFoundType.DOMAIN
                ):
                    structlogger.error(
                        "rasa.core.actions.grpc_custom_action_executor.domain_not_found",
                        event_info=(
                            f"Failed to execute custom action '{self.action_name}'. "
                            f"Could not find domain. {resource_not_found_error.message}"
                        ),
                    )
                    raise DomainNotFound()
                raise RasaException(
                    f"Failed to execute custom action '{self.action_name}'. "
                    f"Error: {details}"
                )
            except grpc.RpcError:
                raise RasaException(
                    f"Failed to execute custom action '{self.action_name}'. "
                    f"Unknown error occurred while calling the "
                    f"action server over gRPC protocol."
                )

    async def run_streaming(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        output_channel: "OutputChannel",
        include_domain: bool = False,
    ) -> Dict[str, Any]:
        """Execute the custom action using a gRPC server-streaming call.

        Opens the ``WebhookStream`` RPC and forwards each ``WebhookStreamEvent``
        to *output_channel* as it arrives:

        * ``chunk_start``  → :meth:`OutputChannel.send_response_chunk_start`
        * ``chunk``        → :meth:`OutputChannel.send_response_chunk`
        * ``chunk_end``    → :meth:`OutputChannel.send_response_chunk_end`
        * ``final_result`` → converted to a plain dict and returned to the caller
        * ``error``        → re-raised as :class:`~rasa.shared.exceptions.RasaException`

        The ``final_result`` event MUST be the last message emitted by the SDK
        servicer; arriving without one raises ``RasaException``.

        Args:
            tracker: Current dialogue tracker (provides the sender ID and state).
            domain: Domain of the assistant.
            output_channel: Channel to forward incremental chunks to.
            include_domain: Whether to include the full domain in the request
                payload (used on domain-not-found retry).

        Returns:
            The action result as a plain dictionary (events + responses), as
            produced by ``MessageToDict`` on the ``WebhookResponse`` carried
            inside the ``final_result`` stream event.

        Raises:
            DomainNotFound: When the server signals the domain is missing.
            RasaException: For all other gRPC or application-level errors.
        """
        request = self._create_payload(
            tracker=tracker, domain=domain, include_domain=include_domain
        )
        metadata = self._build_metadata()

        async with self._create_channel() as channel:
            client = self._create_grpc_client(channel)
            response_stream = client.WebhookStream(request, metadata=metadata)
            # Tri-state session tracker:
            #   False – stream definitely not started (initial value)
            #   None  – chunk_start is being attempted (set before the call)
            #   True  – chunk_start succeeded and session is open
            # Cleanup fires whenever the state is not False, covering both a
            # failed chunk_start attempt and a mid-stream channel error.
            chunk_session_open: Optional[bool] = False
            try:
                streamed_payloads: List[Dict[str, Any]] = []
                async for event in response_stream:
                    which = event.WhichOneof("event")
                    if which == "chunk_start":
                        chunk_session_open = None  # about to attempt
                    result = await self._process_stream_event(
                        event, output_channel, tracker.sender_id, streamed_payloads
                    )
                    # Update only after a successful dispatch.
                    if which == "chunk_start":
                        chunk_session_open = True
                    elif which == "chunk_end":
                        chunk_session_open = False
                    if result is not None:
                        return result
            except grpc.aio.AioRpcError as rpc_error:
                self._raise_for_streaming_rpc_error(rpc_error)
            except grpc.RpcError:
                raise RasaException(
                    f"Failed to execute custom action '{self.action_name}' "
                    f"via streaming. Unknown error occurred while calling the "
                    f"action server over gRPC protocol."
                )
            except BaseException as exception:
                # Close the chunk session whenever it was started or attempted,
                # so the client is not left waiting indefinitely.
                if chunk_session_open is not False:
                    with contextlib.suppress(BaseException):
                        await output_channel.send_response_chunk_end(tracker.sender_id)
                raise exception

        # Stream ended without a final_result event — this is a server-side bug.
        raise RasaException(
            f"Action '{self.action_name}' streaming call ended without a "
            f"'final_result' event. The action server may have crashed or "
            f"sent an incomplete response."
        )

    async def _process_stream_event(
        self,
        event: Any,
        output_channel: "OutputChannel",
        sender_id: str,
        streamed_payloads: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Process a single ``WebhookStreamEvent`` received from the SDK.

        Dispatches the event to the appropriate handler and returns the
        ``final_result`` dict when the terminal event arrives, or ``None``
        for all intermediate events.

        Args:
            event: Protobuf ``WebhookStreamEvent`` message.
            output_channel: Channel to forward chunk events to.
            sender_id: Conversation sender ID.
            streamed_payloads: Accumulator for chunk payloads (mutated in place).

        Returns:
            The ``final_result`` dict when a ``final_result`` event is received,
            ``None`` for every other event type.

        Raises:
            DomainNotFound: On a ``StreamError`` signalling missing domain.
            RasaException: On any other ``StreamError``.
        """
        which = event.WhichOneof("event")

        if which == "chunk_start":
            await output_channel.send_response_chunk_start(sender_id)
        elif which == "chunk":
            chunk_dict = MessageToDict(event.chunk, preserving_proto_field_name=True)
            chunk_dict.pop("response_id", None)
            await dispatch_stream_chunk(output_channel, sender_id, chunk_dict)
            streamed_payloads.append(chunk_dict)
        elif which == "chunk_end":
            await output_channel.send_response_chunk_end(sender_id)
        elif which == "final_result":
            return self._consume_final_result(event, streamed_payloads)
        elif which == "error":
            self._raise_for_stream_error(event.error.message)
        else:
            structlogger.warning(
                "rasa.core.actions.grpc_custom_action_executor"
                ".run_streaming.unknown_event",
                event_info=(
                    f"Received an unknown streaming event type "
                    f"'{which}' for action '{self.action_name}'. Ignoring."
                ),
            )
        return None

    def _consume_final_result(
        self,
        event: Any,
        streamed_payloads: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Convert a ``final_result`` event to a plain dict and check for duplicates.

        Args:
            event: Protobuf ``WebhookStreamEvent`` carrying a ``final_result``.
            streamed_payloads: Chunk payloads already forwarded to the channel.

        Returns:
            The action result as a plain dictionary (``events`` + ``responses``).
        """
        final_result = MessageToDict(event.final_result)
        warn_on_duplicate_streamed_responses(
            action_name=self.action_name,
            streamed_payloads=streamed_payloads,
            final_responses=final_result.get("responses", []),
        )
        return final_result

    def _raise_for_stream_error(self, error_message: str) -> None:
        """Raise the appropriate exception for a ``StreamError`` event.

        The SDK sends a ``StreamError`` with a ``STREAM_ERROR_MISSING_DOMAIN``
        prefix when the request did not include the domain payload.  This is
        translated to :class:`DomainNotFound` so that
        :class:`RetryCustomActionExecutor` can retry with the domain attached.

        Args:
            error_message: The ``message`` field of the ``StreamError`` proto.

        Raises:
            DomainNotFound: When the error signals a missing domain.
            RasaException: For all other streaming errors.
        """
        if error_message.startswith(STREAM_ERROR_MISSING_DOMAIN):
            raise DomainNotFound()
        raise RasaException(
            f"Failed to execute custom action '{self.action_name}'. "
            f"Action server reported a streaming error: {error_message}"
        )

    def _raise_for_streaming_rpc_error(self, rpc_error: grpc.aio.AioRpcError) -> None:
        """Translate an ``AioRpcError`` from the streaming call into a typed exception.

        Mirrors the logic of the unary ``_request`` error handler but uses
        streaming-specific log keys and error messages.

        Args:
            rpc_error: The ``AioRpcError`` raised by ``grpc.aio``.

        Raises:
            DomainNotFound: When the server returns NOT_FOUND with a domain error.
            RasaException: For all other gRPC status codes or parse failures.
        """
        status_code = rpc_error.code()
        details = rpc_error.details()
        if status_code is not grpc.StatusCode.NOT_FOUND:
            raise RasaException(
                f"Failed to execute custom action '{self.action_name}' "
                f"via streaming. Error: {details}"
            )

        resource_not_found_error = self._parse_not_found_error(details)
        if resource_not_found_error is None:
            raise RasaException(
                f"Failed to execute custom action '{self.action_name}' "
                f"via streaming. Error: {details}"
            )
        if resource_not_found_error.resource_type == ResourceNotFoundType.DOMAIN:
            structlogger.error(
                "rasa.core.actions.grpc_custom_action_executor"
                ".run_streaming.domain_not_found",
                event_info=(
                    f"Failed to execute custom action '{self.action_name}'. "
                    f"Could not find domain. {resource_not_found_error.message}"
                ),
            )
            raise DomainNotFound()
        raise RasaException(
            f"Failed to execute custom action '{self.action_name}' "
            f"via streaming. Error: {details}"
        )

    @staticmethod
    def _parse_not_found_error(details: Optional[str]) -> Optional[ResourceNotFound]:
        """Try to parse a gRPC ``NOT_FOUND`` detail string as a ``ResourceNotFound``.

        ``details`` is an arbitrary string supplied by the action server; it is
        not guaranteed to be valid JSON or to conform to the ``ResourceNotFound``
        schema.  Any parse failure (invalid JSON, schema mismatch, …) is caught
        and ``None`` is returned so callers can fall back to a generic
        ``RasaException`` instead of letting an unrelated exception escape the
        error handler.

        Args:
            details: The raw detail string from ``AioRpcError.details()``.

        Returns:
            A parsed ``ResourceNotFound`` instance, or ``None`` if parsing fails.
        """
        if not details:
            return None
        try:
            return ResourceNotFound.model_validate_json(details)
        except Exception:
            return None

    def _build_metadata(self) -> List[Tuple[str, Any]]:
        """Build metadata for the gRPC request.

        Returns:
            Metadata for the gRPC request.
        """
        metadata = []
        for key, value in self.action_endpoint.headers.items():
            metadata.append((key, value))
        return metadata

    @staticmethod
    def _sanitize_payload(
        obj: Dict[str, Any] | List[Any] | Any,
    ) -> Any:
        """Recursively normalise types that ``ParseDict`` cannot handle.

        ``google.protobuf.json_format.ParseDict`` rejects Python ``tuple``
        values with ``ParseError: Value ... has unexpected type <class 'tuple'>``.
        Tracker state frequently contains tuples (e.g. ``ListSlot`` values,
        event tuples) which are semantically identical to lists for protobuf.
        This method converts them before the payload reaches ``ParseDict``,
        avoiding the JSON round-trip fallback.

        Args:
            obj: Arbitrary nested Python object (dict / list / tuple / scalar).

        Returns:
            The same structure with all tuples replaced by lists, recursively.
        """
        if isinstance(obj, dict):
            return {
                k: GRPCCustomActionExecutor._sanitize_payload(v) for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)):
            return [GRPCCustomActionExecutor._sanitize_payload(item) for item in obj]
        return obj

    def _create_payload(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> action_webhook_pb2.WebhookRequest:
        """Create the gRPC payload for the action server.

        Args:
            tracker: Tracker for the current conversation.
            domain: Domain of the assistant.
            include_domain: If True, the domain is included in the request.

        Returns:
            gRPC payload for the action server.
        """
        json_body = self.request_writer.create(
            tracker=tracker, domain=domain, include_domain=include_domain
        )
        # Normalise tuple values before handing off to ParseDict.
        sanitized = self._sanitize_payload(json_body)

        request_proto = action_webhook_pb2.WebhookRequest()

        try:
            return ParseDict(
                js_dict=sanitized, message=request_proto, ignore_unknown_fields=True
            )
        except ParseError as exc:
            # Only reach here for payload shapes not covered by _sanitize_payload.
            # Log the actual error so future occurrences are easy to diagnose.
            structlogger.warning(
                (
                    "rasa.core.actions.grpc_custom_action_executor."
                    "create_grpc_payload_from_dict_failed"
                ),
                event_info=(
                    "Failed to create gRPC payload from Python dictionary. "
                    "Falling back to create payload from JSON intermediary."
                ),
                error=str(exc),
            )
            json_text = json.dumps(sanitized)
            return Parse(
                text=json_text, message=request_proto, ignore_unknown_fields=True
            )

    def _create_grpc_client(
        self,
        channel: grpc.aio.Channel,
    ) -> action_webhook_pb2_grpc.ActionServiceStub:
        """Create a gRPC async stub for the action server.

        Args:
            channel: An open ``grpc.aio.Channel`` to bind the stub to.

        Returns:
            Async gRPC stub for the action server.
        """
        return action_webhook_pb2_grpc.ActionServiceStub(channel)

    def _create_channel(
        self,
    ) -> grpc.aio.Channel:
        """Create an async gRPC channel for the action server.

        Uses ``grpc.aio`` so the channel can be used with ``await`` and
        ``async with`` without blocking the event loop.

        Returns:
            An async gRPC channel (``grpc.aio.Channel``) for the action server.
        """
        compression = grpc.Compression.Gzip

        if self.cert_ca:
            credentials = grpc.ssl_channel_credentials(
                root_certificates=self.cert_ca,
                private_key=self.client_key,
                certificate_chain=self.client_cert,
            )
            return grpc.aio.secure_channel(
                target=self.request_url,
                credentials=credentials,
                compression=compression,
            )
        return grpc.aio.insecure_channel(
            target=self.request_url, compression=compression
        )
