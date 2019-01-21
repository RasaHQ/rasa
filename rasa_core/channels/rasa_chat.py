import logging

import requests
from flask import request, abort

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

    def _check_token(self, token):
        url = "{}/users/me".format(self.base_url)
        headers = {"Authorization": token}
        logger.debug("Requesting user information from auth server {}."
                     "".format(url))
        result = requests.get(url,
                              headers=headers,
                              timeout=DEFAULT_REQUEST_TIMEOUT)

        if result.status_code == 200:
            return result.json()
        else:
            logger.info("Failed to check token: {}. "
                        "Content: {}".format(token, request.data))
            return None

    def _extract_sender(self, req):
        """Fetch user from the Rasa Platform Admin API"""

        if req.headers.get("Authorization"):
            user = self._check_token(req.headers.get("Authorization"))
            if user:
                return user["username"]

        user = self._check_token(req.args.get('token', default=None))
        if user:
            return user["username"]

        abort(401)
