from typing import Any, Dict, Optional, TYPE_CHECKING
from urllib.parse import urlparse

import grpc
import structlog
from google.protobuf.json_format import ParseDict, MessageToDict

from rasa.core.actions.action_exceptions import DomainNotFound
from rasa.core.actions.constants import SSL_CLIENT_CERT_FIELD, SSL_CLIENT_KEY_FIELD
from rasa.core.actions.custom_action_executor import (
    CustomActionExecutor,
    CustomActionRequestWriter,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.io import file_as_bytes
from rasa.utils.endpoints import EndpointConfig
from rasa_sdk.grpc_errors import ResourceNotFound, ResourceNotFoundType
from rasa_sdk.grpc_py import action_webhook_pb2_grpc, action_webhook_pb2

if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker
    from rasa.shared.core.domain import Domain

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
        elif client_cert_file:
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
        elif client_key_file:
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

    async def run(
        self, tracker: "DialogueStateTracker", domain: Optional["Domain"] = None
    ) -> Dict[str, Any]:
        """Execute the custom action using a gRPC request.

        Args:
            tracker: Tracker for the current conversation.
            domain: Domain of the assistant.

        Returns:
            Response from the action server.
        """

        json_body = self.request_writer.create(tracker=tracker, domain=domain)

        return self._request(json_body)

    def _request(
        self,
        json_body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform a single gRPC request to the action server.

        Args:
            json_body: JSON body of the request.

        Returns:
            Response from the action server.
        """

        client = self._create_grpc_client()
        request_proto = action_webhook_pb2.WebhookRequest()
        request = ParseDict(
            js_dict=json_body, message=request_proto, ignore_unknown_fields=True
        )
        try:
            response = client.Webhook(request)
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
