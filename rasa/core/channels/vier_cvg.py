import asyncio
import base64
import copy
import json
import logging
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, Optional, Text

import rasa.shared.utils.io
import structlog
from rasa.core.channels.channel import InputChannel, OutputChannel, UserMessage

from rasa.utils.beta import ensure_beta_feature_is_enabled

# ignore ResourceWarning, InsecureRequestWarning
warnings.filterwarnings("ignore", category=ResourceWarning)

from cvg_sdk.api.call_api import CallApi  # noqa: E402
from cvg_sdk.api_client import ApiClient  # noqa: E402
from cvg_sdk.configuration import Configuration  # noqa: E402
from cvg_sdk.model.outbound_call_result import OutboundCallResult  # noqa: E402
from cvg_sdk.model.say_parameters import SayParameters  # noqa: E402
from sanic import Blueprint, response  # noqa: E402
from sanic.request import Request  # noqa: E402
from sanic.response import HTTPResponse  # noqa: E402

logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()

CHANNEL_NAME = "vier_cvg"
FEATURE_NAME = "VIER Voice Connector"
RASA_PRO_BETA_VIER = "RASA_PRO_BETA_VIER"


def make_metadata(payload: Any) -> Dict[Text, Any]:
    return {"cvg_body": payload}


@dataclass
class Recipient:
    dialog_id: str
    project_token: str
    reseller_token: str


def parse_recipient_id(recipient_id: Text) -> Recipient:
    parsed_json = json.loads(
        base64.b64decode(bytes(recipient_id, "utf-8")).decode("utf-8")
    )
    project_context = parsed_json["projectContext"]
    return Recipient(
        parsed_json["dialogId"],
        project_context["projectToken"],
        project_context["resellerToken"],
    )


class CVGOutput(OutputChannel):
    """Output channel for the Cognitive Voice Gateway."""

    on_message: Callable[[UserMessage], Awaitable[Any]]

    @classmethod
    def name(cls) -> Text:
        return CHANNEL_NAME

    def __init__(
        self,
        callback_base_url: Text,
        on_message: Callable[[UserMessage], Awaitable[Any]],
        proxy: Optional[Text] = None,
    ) -> None:
        configuration = Configuration.get_default_copy()
        configuration.host = callback_base_url
        configuration.proxy = proxy
        configuration.verify_ssl = False

        self.on_message = on_message

        self.api_client = ApiClient(configuration=configuration)
        self.call_api = CallApi(self.api_client)

    async def send_text_message(
        self,
        recipient_id: Text,
        text: Text,
        **kwargs: Any,
    ) -> None:
        structlogger.debug("vier_cvg.send.text.message", text=copy.deepcopy(text))

        dialog_id = parse_recipient_id(recipient_id).dialog_id
        self.call_api.say(SayParameters(dialog_id=dialog_id, text=text))

    async def _execute_operation_by_name(
        self,
        operation_name: Text,
        body: Any,
        recipient_id: Text,
    ) -> Any:
        recipient = parse_recipient_id(recipient_id)
        dialog_id = recipient.dialog_id
        reseller_token = recipient.reseller_token

        async def handle_outbound_call_result(
            outbound_call_result: OutboundCallResult,
        ) -> Any:
            if outbound_call_result.status == "Success":
                user_message = UserMessage(
                    text="/cvg_outbound_success",
                    output_channel=self,
                    sender_id=recipient_id,
                    input_channel=CHANNEL_NAME,
                    metadata=make_metadata(outbound_call_result.to_dict()),
                )
            elif outbound_call_result.status == "Failure":
                user_message = UserMessage(
                    text="/cvg_outbound_failure",
                    output_channel=self,
                    sender_id=recipient_id,
                    input_channel=CHANNEL_NAME,
                    metadata=make_metadata(outbound_call_result.to_dict()),
                )
            else:
                return response.text(
                    f"Invalid OutboundCallResult status: {outbound_call_result.status}",
                    status=400,
                )

            logger.info(
                "Creating incoming UserMessage: {text=%s, output_channel=%s, sender_id=%s, metadata=%s}"  # noqa: E501
                % (
                    user_message.text,
                    user_message.output_channel,
                    user_message.sender_id,
                    user_message.metadata,
                )
            )
            await self.on_message(user_message)

        newBody = copy.deepcopy(body)
        if newBody is None:
            newBody = {}
        path = operation_name[3:].replace("_", "/")
        if operation_name.startswith("cvg_call_"):
            if "dialogId" not in body:
                newBody["dialogId"] = dialog_id

            # The response from forward and bridge must be handled
            handle_result_outbound_call_result_for = [
                "cvg_call_forward",
                "cvg_call_bridge",
            ]
            if operation_name in handle_result_outbound_call_result_for:
                # The request must be async: We cannot trigger another intent,
                # while the send_ function of OutputChannel is not finished yet.
                # (conversation is locked)
                self.api_client.pool.apply_async(
                    self.api_client.call_api,
                    (path, "POST"),
                    {"body": newBody, "response_type": (OutboundCallResult,)},
                    callback=lambda result: asyncio.run(
                        handle_outbound_call_result(result[0])
                    ),
                )

            return self.api_client.call_api(path, "POST", body=newBody)
        elif operation_name.startswith("cvg_dialog_"):
            if operation_name == "cvg_dialog_delete":
                return self.api_client.call_api(
                    f"/dialog/{reseller_token}/{dialog_id}", "DELETE", body=newBody
                )
            elif operation_name == "cvg_dialog_data":
                return self.api_client.call_api(
                    f"/dialog/{reseller_token}/{dialog_id}/data", "POST", body=newBody
                )
            else:
                logger.error(
                    f"Dialog operation {operation_name} not found/not implemented yet. "
                    f"Please consider using the cvg-python-sdk in one of your actions."
                )
        else:
            logger.error(f"Operation {operation_name} not found/not implemented yet")
        logger.info("Ran operation: " + operation_name)

    async def send_custom_json(
        self,
        recipient_id: Text,
        json_message: Dict[Text, Any],
        **kwargs: Any,
    ) -> None:
        structlogger.debug(
            "vier_cvg.send.json",
            json_message=copy.deepcopy(json.dumps(json_message)),
            recipient_id=recipient_id,
        )
        for operation_name, body in json_message.items():
            await self._execute_operation_by_name(operation_name, body, recipient_id)

    async def send_image_url(*args: Any, **kwargs: Any) -> None:
        # We do not support images.
        rasa.shared.utils.io.raise_warning(
            "Ignoring image URL."
            "We cannot represent images as a voice bot."
            "Please define a voice-friendly alternative."
        )


class CVGInput(InputChannel):
    """Input channel for the Cognitive Voice Gateway."""

    @classmethod
    def name(cls) -> Text:
        return CHANNEL_NAME

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if credentials is None:
            cls.raise_missing_credentials_exception()
        token = credentials.get("token")
        if token is None or isinstance(token, str) or len(token) == 0:
            raise ValueError(
                "No authentication token has been configured in your credentials.yml!"
            )
        proxy = credentials.get("proxy")
        start_intent = credentials.get("start_intent")
        if start_intent is None:
            start_intent = "/cvg_session"

        return cls(token, start_intent, proxy)

    def __init__(
        self, token: Text, start_intent: Text, proxy: Optional[Text] = None
    ) -> None:
        ensure_beta_feature_is_enabled(
            feature_name=FEATURE_NAME,
            env_flag=RASA_PRO_BETA_VIER,
        )

        self.callback = None
        self.expected_authorization_header_value = f"Bearer {token}"
        self.proxy = proxy
        self.start_intent = start_intent

    async def process_message(
        self,
        request: Request,
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        text: Text,
        sender_id: Optional[Text],
    ) -> Any:
        try:
            if text[-1] == ".":
                text = text[:-1]

            metadata = make_metadata(request.json)
            user_msg = UserMessage(
                text=text,
                output_channel=CVGOutput(
                    request.json["callback"], on_new_message, self.proxy
                ),
                sender_id=sender_id,
                input_channel=CHANNEL_NAME,
                metadata=metadata,
            )

            structlogger.debug(
                "vier_cvg.process.message",
                text=copy.deepcopy(text),
                output_channel=user_msg.output_channel,
                sender_id=sender_id,
                metadata=metadata,
            )

            await on_new_message(user_msg)
        except Exception as e:
            logger.error(f"Exception when trying to handle message: {e}")
            logger.error(str(e), exc_info=True)

        return response.text("")

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        def valid_request(func: Any) -> Any:
            def decorator(f: Any) -> Any:
                @wraps(f)
                async def decorated_function(
                    request: HTTPResponse, *args: Any, **kwargs: Any
                ) -> Any:
                    if (
                        request.headers.get("authorization")
                        != self.expected_authorization_header_value
                    ):
                        return response.text("bot token is invalid!", status=401)

                    if not request.headers.get("content-type") == "application/json":
                        return response.text(
                            "content-type is not supported. "
                            "Please use application/json",
                            status=415,
                        )
                    json_body = request.json  # type: ignore[attr-defined]
                    if json_body is None:
                        return response.text("body is not valid json.", status=400)
                    if json_body["dialogId"] is None:
                        return response.text("dialogId is required", status=400)
                    if json_body["callback"] is None:
                        return response.text("callback is required", status=400)
                    if json_body["projectContext"] is None:
                        return response.text("projectContext is required", status=400)
                    return await f(request, *args, **kwargs)

                return decorated_function

            return decorator(func)

        async def process_message_oneline(request: Request, text: Text) -> Any:
            sender_id_json = json.dumps(
                {
                    "dialogId": request.json["dialogId"],
                    "projectContext": request.json["projectContext"],
                }
            )
            sender_id_base64 = base64.b64encode(bytes(sender_id_json, "utf-8")).decode(
                "utf-8"
            )

            return await self.process_message(
                request,
                on_new_message,
                text=text,
                sender_id=sender_id_base64,
            )

        cvg_webhook = Blueprint(
            "vier_cvg_webhook",
            __name__,
        )

        @cvg_webhook.post("/session")
        @valid_request
        async def session(request: Request) -> HTTPResponse:
            return await process_message_oneline(request, self.start_intent)

        @cvg_webhook.post("/message")
        @valid_request
        async def message(request: Request) -> HTTPResponse:
            return await process_message_oneline(request, request.json["text"])

        @cvg_webhook.post("/answer")
        @valid_request
        async def answer(request: Request) -> HTTPResponse:
            return await process_message_oneline(
                request, "/cvg_answer_" + request.json["type"]["name"].lower()
            )

        @cvg_webhook.post("/inactivity")
        @valid_request
        async def inactivity(request: Request) -> HTTPResponse:
            return await process_message_oneline(request, "/cvg_inactivity")

        @cvg_webhook.post("/terminated")
        @valid_request
        async def terminated(request: Request) -> HTTPResponse:
            return await process_message_oneline(request, "/cvg_terminated")

        @cvg_webhook.post("/recording")
        @valid_request
        async def recording(request: Request) -> HTTPResponse:
            return await process_message_oneline(request, "/cvg_recording")

        return cvg_webhook
