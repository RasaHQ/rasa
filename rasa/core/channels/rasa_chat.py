import json
from typing import Text, Optional, Dict

import aiohttp
import logging
from sanic.exceptions import abort
import jwt

from rasa.core import constants
from rasa.core.channels.channel import RestInput
from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from sanic.request import Request

logger = logging.getLogger(__name__)


class RasaChatInput(RestInput):
    """Chat input channel for Rasa X"""

    @classmethod
    def name(cls):
        return "rasa"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(credentials.get("url"))

    def __init__(self, url):
        self.base_url = url
        self.jwt_key = None
        self.jwt_algorithm = None

    async def _fetch_public_key(self) -> None:
        public_key_url = "{}/version".format(self.base_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(
                public_key_url, timeout=DEFAULT_REQUEST_TIMEOUT
            ) as resp:
                status_code = resp.status
                if status_code != 200:
                    logger.error(
                        "Failed to fetch JWT public key from URL '{}' with "
                        "status code {}: {}"
                        "".format(public_key_url, status_code, await resp.text())
                    )
                    return
                rjs = await resp.json()
                public_key_field = "keys"
                if public_key_field in rjs:
                    self.jwt_key = rjs["keys"][0]["key"]
                    self.jwt_algorithm = rjs["keys"][0]["alg"]
                    logger.debug(
                        "Fetched JWT public key from URL '{}' for algorithm '{}':\n{}"
                        "".format(public_key_url, self.jwt_algorithm, self.jwt_key)
                    )
                else:
                    logger.error(
                        "Retrieved json response from URL '{}' but could not find "
                        "'{}' field containing the JWT public key. Please make sure "
                        "you use an up-to-date version of Rasa X (>= 0.20.2). "
                        "Response was: {}"
                        "".format(public_key_url, public_key_field, json.dumps(rjs))
                    )

    def _decode_jwt(self, bearer_token: Text) -> Dict:
        authorization_header_value = bearer_token.replace(
            constants.BEARER_TOKEN_PREFIX, ""
        )
        return jwt.decode(
            authorization_header_value, self.jwt_key, algorithms=self.jwt_algorithm
        )

    async def _decode_bearer_token(self, bearer_token: Text) -> Optional[Dict]:
        if self.jwt_key is None:
            await self._fetch_public_key()

        # noinspection PyBroadException
        try:
            return self._decode_jwt(bearer_token)
        except jwt.exceptions.InvalidSignatureError:
            logger.error("JWT public key invalid, fetching new one.")
            await self._fetch_public_key()
            return self._decode_jwt(bearer_token)
        except Exception:
            logger.exception("Failed to decode bearer token.")

    async def _extract_sender(self, req: Request) -> Optional[Text]:
        """Fetch user from the Rasa X Admin API"""

        if req.headers.get("Authorization"):
            user = await self._decode_bearer_token(req.headers["Authorization"])
            if user:
                return user["username"]

        user = await self._decode_bearer_token(req.args.get("token", default=None))
        if user:
            return user["username"]

        abort(401)
