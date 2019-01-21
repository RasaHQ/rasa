import aiohttp
import logging
from sanic.exceptions import abort

from rasa_core.channels.channel import RestInput
from rasa_core.constants import DEFAULT_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


class RasaChatInput(RestInput):
    """Chat input channel for Rasa Platform"""

    @classmethod
    def name(cls):
        return "rasa"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        return cls(credentials.get("url"),
                   credentials.get("admin_token"))

    def __init__(self, url, admin_token=None):
        self.base_url = url
        self.admin_token = admin_token

    async def _check_token(self, token):
        url = "{}/users/me".format(self.base_url)
        headers = {"Authorization": token}
        logger.debug("Requesting user information from auth server {}."
                     "".format(url))

        async with aiohttp.ClientSession() as session:
            async with session.get(url,
                                   headers=headers,
                                   timeout=DEFAULT_REQUEST_TIMEOUT) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.info("Failed to check token: {}. "
                                "Content: {}".format(token, await resp.text()))
                    return None

    async def _extract_sender(self, req):
        """Fetch user from the Rasa Platform Admin API"""

        if req.headers.get("Authorization"):
            user = await self._check_token(req.headers.get("Authorization"))
            if user:
                return user["username"]

        user = await self._check_token(req.args.get('token', default=None))
        if user:
            return user["username"]

        abort(401)
