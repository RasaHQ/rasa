import json
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from urllib.parse import urlparse

import grpc
import structlog
from google.protobuf.json_format import MessageToDict, Parse, ParseDict
from rasa_sdk.grpc_errors import ResourceNotFound, ResourceNotFoundType
from rasa_sdk.grpc_py import action_webhook_pb2, action_webhook_pb2_grpc

from rasa.core.actions.action_exceptions import DomainNotFound
from rasa.core.actions.constants import SSL_CLIENT_CERT_FIELD, SSL_CLIENT_KEY_FIELD
from rasa.core.actions.custom_action_executor import (
    CustomActionExecutor,
    CustomActionRequestWriter,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.io import file_as_bytes
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger(__name__)


class GRPCCustomActionExecutor(CustomActionExecutor):
    """gRPC-based implementation of the CustomActionExecutor.

    Executes custom actions by making gRPC requests to the action endpoint.
    """

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
        """

        request = self._create_payload(
            tracker=tracker, domain=domain, include_domain=include_domain
        )

        return self._request(request)

    def _request(
        self,
        request: action_webhook_pb2.WebhookRequest,
    ) -> Dict[str, Any]:
        """Perform a single gRPC request to the action server.

        Args:
            request: gRPC Request to be sent to the action server.

        Returns:
            Response from the action server.
        """

        client = self._create_grpc_client()
        metadata = self._build_metadata()
        try:
            response = client.Webhook(request, metadata=metadata)
            return MessageToDict(response)
        except grpc.RpcError as rpc_error:
            if not isinstance(rpc_error, grpc.Call):
                raise RasaException(
                    f"Failed to execute custom action '{self.action_name}'. "
                    f"Unknown error occurred while calling the "
                    f"action server over gRPC protocol."
                )

            status_code = rpc_error.code()
            details = rpc_error.details()
            if status_code is not grpc.StatusCode.NOT_FOUND:
                raise RasaException(
                    f"Failed to execute custom action '{self.action_name}'. "
                    f"Error: {details}"
                )

            resource_not_found_error = ResourceNotFound.model_validate_json(details)
            if (
                resource_not_found_error
                and resource_not_found_error.resource_type
                == ResourceNotFoundType.DOMAIN
            ):
                structlogger.error(
                    "rasa.core.actions.grpc_custom_action_executor.domain_not_found",
                    event_info=(
                        f"Failed to execute custom action '{self.action_endpoint}'. "
                        f"Could not find domain. {resource_not_found_error.message}"
                    ),
                )
                raise DomainNotFound()
            raise RasaException(
                f"Failed to execute custom action '{self.action_name}'. "
                f"Error: {details}"
            )

    def _build_metadata(self) -> List[Tuple[str, Any]]:
        """Build metadata for the gRPC request.

        Returns:
            Metadata for the gRPC request.
        """
        metadata = []
        for key, value in self.action_endpoint.headers.items():
            metadata.append((key, value))
        return metadata

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

        request_proto = action_webhook_pb2.WebhookRequest()

        try:
            return ParseDict(
                js_dict=json_body, message=request_proto, ignore_unknown_fields=True
            )
        except Exception:
            structlogger.warning(
                (
                    "rasa.core.actions.grpc_custom_action_executor."
                    "create_grpc_payload_from_dict_failed"
                ),
                event_info=(
                    "Failed to create gRPC payload from Python dictionary. "
                    "Falling back to create payload from JSON intermediary."
                ),
            )
            json_text = json.dumps(json_body)
            return Parse(
                text=json_text, message=request_proto, ignore_unknown_fields=True
            )

    def _create_grpc_client(
        self,
    ) -> action_webhook_pb2_grpc.ActionServiceStub:
        """Create a gRPC client for the action server.

        Returns:
            gRPC client for the action server.
        """
        channel = self._create_channel()
        return action_webhook_pb2_grpc.ActionServiceStub(channel)

    def _create_channel(
        self,
    ) -> grpc.Channel:
        """Create a gRPC channel for the action server.

        Returns:
            gRPC channel for the action server.
        """

        compression = grpc.Compression.Gzip

        if self.cert_ca:
            credentials = grpc.ssl_channel_credentials(
                root_certificates=self.cert_ca,
                private_key=self.client_key,
                certificate_chain=self.client_cert,
            )
            return grpc.secure_channel(
                target=self.request_url,
                credentials=credentials,
                compression=compression,
            )
        return grpc.insecure_channel(target=self.request_url, compression=compression)
