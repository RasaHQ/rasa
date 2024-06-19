from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Text,
)

import structlog
from rasa.core.channels import SocketIOInput
from sanic import response

if TYPE_CHECKING:
    from rasa.core.channels.channel import InputChannel, UserMessage
    from sanic import Blueprint
    from sanic.request import Request
    from sanic.response import HTTPResponse


INSPECT_TEMPLATE_PATH = "inspector/dist"

structlogger = structlog.get_logger()


class DevelopmentInspectInput(SocketIOInput):
    """Internal Rasa Pro-only SocketIO-based channel.

    It extends the Rasa Pro SocketIO channel by
    adding the /inspect endpoint, which works hand in hand with `rasa inspect`.
    """

    @classmethod
    def name(cls) -> Text:
        """Channel name."""
        return "inspector"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> "InputChannel":
        """Override function to set default parameters."""
        credentials = credentials or {}
        if credentials.get("session_persistence") is False:
            structlogger.error("development_inspector.init.session_persistence_ignored")

        # session_persistence needs to be True
        credentials["session_persistence"] = True
        return super().from_credentials(credentials)

    def __init__(
        self,
        user_message_evt: Text = "user_uttered",
        bot_message_evt: Text = "bot_uttered",
        namespace: Optional[Text] = None,
        # this argument is kept to keep the signature compatible with the parent class
        # especially with from_credentials()
        session_persistence: bool = True,
        socketio_path: Optional[Text] = "/socket.io",
        jwt_key: Optional[Text] = None,
        jwt_method: Optional[Text] = "HS256",
        metadata_key: Optional[Text] = "metadata",
    ):
        """Override function to set default parameters."""
        super().__init__(
            user_message_evt,
            bot_message_evt,
            namespace,
            True,  # session_persistence needs to be True
            socketio_path,
            jwt_key,
            jwt_method,
            metadata_key,
        )

    @staticmethod
    def inspect_html_path() -> Text:
        """Returns the path to the inspect.html file."""
        import pkg_resources

        return pkg_resources.resource_filename(__name__, INSPECT_TEMPLATE_PATH)

    def blueprint(
        self, on_new_message: Callable[["UserMessage"], Awaitable[Any]]
    ) -> "Blueprint":
        """Defines a Sanic blueprint."""
        socketio_webhook: "Blueprint" = super().blueprint(on_new_message)
        socketio_webhook.static("/assets", self.inspect_html_path() + "/assets")

        @socketio_webhook.route("/inspect.html", methods=["GET"])
        async def inspect(_: "Request") -> "HTTPResponse":
            return await response.file(self.inspect_html_path() + "/index.html")

        return socketio_webhook
