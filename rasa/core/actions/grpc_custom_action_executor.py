from os.path import abspath
from typing import Dict, Text, Any, Optional, TYPE_CHECKING

import grpc
import structlog
from google.protobuf.json_format import ParseDict, MessageToDict

from rasa.core.actions.action_exceptions import DomainNotFound
from rasa.core.actions.custom_action_executor import (
    CustomActionExecutor,
    CustomActionRequestWriter,
)
from rasa.shared.exceptions import FileNotFoundException
from rasa.utils.endpoints import EndpointConfig, ClientResponseError
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

    def __init__(self, action_name: str, action_endpoint: EndpointConfig) -> None:
        """Initializes the gRPC custom action executor.

        Args:
            action_name: Name of the custom action.
            action_endpoint: Endpoint configuration of the custom action.
        """
        self.action_name = action_name
        self.action_endpoint = action_endpoint
        self.request_writer = CustomActionRequestWriter(action_name, action_endpoint)

    async def run(
        self, tracker: "DialogueStateTracker", domain: Optional["Domain"] = None
    ) -> Dict[Text, Any]:
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
        json_body: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Perform a single gRPC request to the action server.

        Args:
            json_body: JSON body of the request.

        Returns:
            Response from the action server.
        """

        client = self._create_grpc_client(
            self.action_endpoint.url, self.action_endpoint.cafile
        )
        request_proto = action_webhook_pb2.WebhookRequest()
        request = ParseDict(
            js_dict=json_body, message=request_proto, ignore_unknown_fields=True
        )
        try:
            response = client.webhook(request)
            return MessageToDict(response)
        except grpc.RpcError as e:
            status_code = e.code()
            details = e.details()
            if status_code is not grpc.StatusCode.NOT_FOUND:
                raise ClientResponseError(status=status_code, message=details, text="")

            resource_not_found_error = ResourceNotFound.model_validate_json(details)
            if not resource_not_found_error:
                raise ClientResponseError(status=status_code, message=details, text="")

            if resource_not_found_error.resource_type == ResourceNotFoundType.DOMAIN:
                structlogger.error(
                    "rasa.core.actions.grpc_custom_action_executor.domain_not_found",
                    event_info=(
                        f"Failed to execute custom action '{self.action_endpoint}'. "
                        f"Could not find domain. {resource_not_found_error.message}"
                    ),
                )
                raise DomainNotFound()
            elif resource_not_found_error.resource_type == ResourceNotFoundType.ACTION:
                structlogger.error(
                    "rasa.core.actions.grpc_custom_action_executor.action_not_found",
                    event_info=(
                        f"Failed to execute custom action '{self.action_name}'. "
                        f"Could not find action. {resource_not_found_error.message}"
                    ),
                )
            raise ClientResponseError(status=status_code, message=details, text="")

    @staticmethod
    def _create_grpc_client(
        url: Text, cert_file: Optional[Text] = None
    ) -> action_webhook_pb2_grpc.ActionServerWebhookStub:
        """Create a gRPC client for the action server.

        Args:
            url: URL of the action server.
            cert_file: Path to the certificate file for TLS encryption.

        Returns:
            gRPC client for the action server.
        """
        channel = GRPCCustomActionExecutor._create_channel(url, cert_file)
        return action_webhook_pb2_grpc.ActionServerWebhookStub(channel)

    @staticmethod
    def _create_channel(
        url: Optional[Text], cert_ca_file: Optional[Text] = None
    ) -> grpc.Channel:
        """Create a gRPC channel for the action server.

        Args:
            url: URL of the action server.
            cert_ca_file: Path to the certificate file for TLS encryption.

        Returns:
            gRPC channel for the action server.
        """
        if cert_ca_file:
            try:
                credentials = grpc.ssl_channel_credentials(
                    root_certificates=open(cert_ca_file, "rb").read()
                    if cert_ca_file
                    else None
                )
                return grpc.secure_channel(url, credentials)
            except FileNotFoundError as e:
                raise FileNotFoundException(
                    f"Failed to find certificate file, "
                    f"'{abspath(cert_ca_file)}' does not exist."
                ) from e

        else:
            return grpc.insecure_channel(url.lstrip("grpc://"))
