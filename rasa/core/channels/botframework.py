import datetime
import json
import logging
import re
from http import HTTPStatus
from typing import Text, Dict, Any, List, Iterable, Callable, Awaitable, Optional

import jwt
import requests
from jwt import InvalidKeyError, PyJWTError
from jwt.algorithms import RSAAlgorithm
from requests import HTTPError
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse

from rasa.core.channels.channel import UserMessage, OutputChannel, InputChannel

MICROSOFT_OPEN_ID_URI = (
    "https://login.botframework.com/v1/.well-known/openidconfiguration"
)

BEARER_REGEX = re.compile(r"Bearer\s+(.*)")

logger = logging.getLogger(__name__)

MICROSOFT_OAUTH2_URL = "https://login.microsoftonline.com"

MICROSOFT_OAUTH2_PATH = "botframework.com/oauth2/v2.0/token"


class BotFramework(OutputChannel):
    """A Microsoft Bot Framework communication channel."""

    token_expiration_date = datetime.datetime.now()

    headers = None

    @classmethod
    def name(cls) -> Text:
        return "botframework"

    def __init__(
        self,
        app_id: Text,
        app_password: Text,
        conversation: Dict[Text, Any],
        bot: Text,
        service_url: Text,
    ) -> None:

        service_url = (
            f"{service_url}/" if not service_url.endswith("/") else service_url
        )

        self.app_id = app_id
        self.app_password = app_password
        self.conversation = conversation
        self.global_uri = f"{service_url}v3/"
        self.bot = bot

    async def _get_headers(self) -> Optional[Dict[Text, Any]]:
        if BotFramework.token_expiration_date < datetime.datetime.now():
            uri = f"{MICROSOFT_OAUTH2_URL}/{MICROSOFT_OAUTH2_PATH}"
            grant_type = "client_credentials"
            scope = "https://api.botframework.com/.default"
            payload = {
                "client_id": self.app_id,
                "client_secret": self.app_password,
                "grant_type": grant_type,
                "scope": scope,
            }

            token_response = requests.post(uri, data=payload)

            if token_response.ok:
                token_data = token_response.json()
                access_token = token_data["access_token"]
                token_expiration = token_data["expires_in"]

                delta = datetime.timedelta(seconds=int(token_expiration))
                BotFramework.token_expiration_date = datetime.datetime.now() + delta

                BotFramework.headers = {
                    "content-type": "application/json",
                    "Authorization": "Bearer %s" % access_token,
                }
                return BotFramework.headers
            else:
                logger.error("Could not get BotFramework token")
                return None
        else:
            return BotFramework.headers

    def prepare_message(
        self, recipient_id: Text, message_data: Dict[Text, Any]
    ) -> Dict[Text, Any]:
        data = {
            "type": "message",
            "recipient": {"id": recipient_id},
            "from": self.bot,
            "channelData": {"notification": {"alert": "true"}},
            "text": "",
        }
        data.update(message_data)
        return data

    async def send(self, message_data: Dict[Text, Any]) -> None:
        post_message_uri = "{}conversations/{}/activities".format(
            self.global_uri, self.conversation["id"]
        )
        headers = await self._get_headers()
        send_response = requests.post(
            post_message_uri, headers=headers, data=json.dumps(message_data)
        )

        if not send_response.ok:
            logger.error(
                "Error trying to send botframework messge. Response: %s",
                send_response.text,
            )

    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        for message_part in text.strip().split("\n\n"):
            text_message = {"text": message_part}
            message = self.prepare_message(recipient_id, text_message)
            await self.send(message)

    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        hero_content = {
            "contentType": "application/vnd.microsoft.card.hero",
            "content": {"images": [{"url": image}]},
        }

        image_message = {"attachments": [hero_content]}
        message = self.prepare_message(recipient_id, image_message)
        await self.send(message)

    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        hero_content = {
            "contentType": "application/vnd.microsoft.card.hero",
            "content": {"subtitle": text, "buttons": buttons},
        }

        buttons_message = {"attachments": [hero_content]}
        message = self.prepare_message(recipient_id, buttons_message)
        await self.send(message)

    async def send_elements(
        self, recipient_id: Text, elements: Iterable[Dict[Text, Any]], **kwargs: Any
    ) -> None:
        for e in elements:
            message = self.prepare_message(recipient_id, e)
            await self.send(message)

    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        json_message.setdefault("type", "message")
        json_message.setdefault("recipient", {}).setdefault("id", recipient_id)
        json_message.setdefault("from", self.bot)
        json_message.setdefault("channelData", {}).setdefault(
            "notification", {}
        ).setdefault("alert", "true")
        json_message.setdefault("text", "")
        await self.send(json_message)


class BotFrameworkInput(InputChannel):
    """Bot Framework input channel implementation."""

    @classmethod
    def name(cls) -> Text:
        return "botframework"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(credentials.get("app_id"), credentials.get("app_password"))

    def __init__(self, app_id: Text, app_password: Text) -> None:
        """Create a Bot Framework input channel.

        Args:
            app_id: Bot Framework's API id
            app_password: Bot Framework application secret
        """

        self.app_id = app_id
        self.app_password = app_password

        self.jwt_keys: Dict[Text, Any] = {}
        self.jwt_update_time = datetime.datetime.fromtimestamp(0)

        self._update_cached_jwk_keys()

    def _update_cached_jwk_keys(self) -> None:
        logger.debug("Updating JWT keys for the Botframework.")
        response = requests.get(MICROSOFT_OPEN_ID_URI)
        response.raise_for_status()
        conf = response.json()

        jwks_uri = conf["jwks_uri"]

        keys_request = requests.get(jwks_uri)
        keys_request.raise_for_status()
        keys_list = keys_request.json()
        self.jwt_keys = {key["kid"]: key for key in keys_list["keys"]}
        self.jwt_update_time = datetime.datetime.now()

    def _validate_jwt_token(self, jwt_token: Text) -> None:
        jwt_header = jwt.get_unverified_header(jwt_token)
        key_id = jwt_header["kid"]
        if key_id not in self.jwt_keys:
            raise InvalidKeyError(f"JWT Key with ID {key_id} not found.")

        key_json = self.jwt_keys[key_id]
        public_key = RSAAlgorithm.from_jwk(key_json)  # type: ignore
        jwt.decode(
            jwt_token,
            key=public_key,
            audience=self.app_id,
            algorithms=jwt_header["alg"],
        )

    def _validate_auth(self, auth_header: Optional[Text]) -> Optional[HTTPResponse]:
        if not auth_header:
            return response.text(
                "No authorization header provided.", status=HTTPStatus.UNAUTHORIZED
            )

        # Update the JWT keys daily
        if datetime.datetime.now() - self.jwt_update_time > datetime.timedelta(days=1):
            try:
                self._update_cached_jwk_keys()
            except HTTPError as error:
                logger.warning(
                    f"Could not update JWT keys from {MICROSOFT_OPEN_ID_URI}."
                )
                logger.exception(error, exc_info=True)

        auth_match = BEARER_REGEX.match(auth_header)
        if not auth_match:
            return response.text(
                "No Bearer token provided in Authorization header.",
                status=HTTPStatus.UNAUTHORIZED,
            )

        (jwt_token,) = auth_match.groups()

        try:
            self._validate_jwt_token(jwt_token)
        except PyJWTError as error:
            logger.error("Bot framework JWT token could not be verified.")
            logger.exception(error, exc_info=True)
            return response.text(
                "Could not validate JWT token.", status=HTTPStatus.UNAUTHORIZED
            )

        return None

    @staticmethod
    def add_attachments_to_metadata(
        postdata: Dict[Text, Any], metadata: Optional[Dict[Text, Any]]
    ) -> Optional[Dict[Text, Any]]:
        """Merge the values of `postdata['attachments']` with `metadata`."""
        if postdata.get("attachments"):
            attachments = {"attachments": postdata["attachments"]}
            if metadata:
                metadata.update(attachments)
            else:
                metadata = attachments
        return metadata

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        """Defines the Sanic blueprint for the bot framework integration."""
        botframework_webhook = Blueprint("botframework_webhook", __name__)

        @botframework_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @botframework_webhook.route("/webhook", methods=["POST"])
        async def webhook(request: Request) -> HTTPResponse:
            validation_response = self._validate_auth(
                request.headers.get("Authorization")
            )
            if validation_response:
                return validation_response

            postdata = request.json
            metadata = self.get_metadata(request)

            metadata_with_attachments = self.add_attachments_to_metadata(
                postdata, metadata
            )

            try:
                if postdata["type"] == "message":
                    out_channel = BotFramework(
                        self.app_id,
                        self.app_password,
                        postdata["conversation"],
                        postdata["recipient"],
                        postdata["serviceUrl"],
                    )

                    user_msg = UserMessage(
                        text=postdata.get("text", ""),
                        output_channel=out_channel,
                        sender_id=postdata["from"]["id"],
                        input_channel=self.name(),
                        metadata=metadata_with_attachments,
                    )

                    await on_new_message(user_msg)
                else:
                    logger.info("Not received message type")
            except Exception as e:
                logger.error(f"Exception when trying to handle message.{e}")
                logger.debug(e, exc_info=True)
                pass

            return response.text("success")

        return botframework_webhook
