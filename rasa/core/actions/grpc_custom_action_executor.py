import logging
from os.path import abspath
from typing import Dict, Text, Any, Optional, TYPE_CHECKING

import grpc
from google.protobuf.json_format import ParseDict, MessageToDict

from rasa.core.actions.custom_action_executor import CustomActionExecutor
from rasa.shared.exceptions import FileNotFoundException
from rasa.utils.endpoints import EndpointConfig, ClientResponseError
from rasa_sdk.grpc_py import action_webhook_pb2_grpc, action_webhook_pb2

if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker
    from rasa.shared.core.domain import Domain

logger = logging.getLogger(__name__)


class GRPCCustomActionExecutor(CustomActionExecutor):
    """gRPC-based implementation of the CustomActionExecutor.

    Executes custom actions by making gRPC requests to the action endpoint.
    """

    def __init__(self, name: str, action_endpoint: EndpointConfig) -> None:
        super().__init__(name, action_endpoint)

    async def run(
        self, tracker: "DialogueStateTracker", domain: "Domain"
    ) -> Dict[Text, Any]:
        """Execute the custom action using a gRPC request."""

        json_body = self._action_call_format(
            tracker=tracker, domain=domain, should_include_domain=False
        )

        return self._request(json_body)

    def _request(
        self,
        json_body: Dict[Text, Any],
    ) -> Dict[Text, Any]:
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
            raise ClientResponseError(status=status_code, message=details, text="")

    @staticmethod
    def _create_grpc_client(
        url: Text, cert_file: Optional[Text] = None
    ) -> action_webhook_pb2_grpc.ActionServerWebhookStub:
        channel = GRPCCustomActionExecutor._create_channel(url, cert_file)
        return action_webhook_pb2_grpc.ActionServerWebhookStub(channel)

    @staticmethod
    def _create_channel(
        url: Optional[Text], cert_ca_file: Optional[Text] = None
    ) -> grpc.Channel:
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
