import aiohttp

import logging

from typing import Text, Dict, Any, Optional

from rasa.core import constants
from rasa.core.channels import UserMessage
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class RasaNLUHttpInterpreter:
    """Allows for an HTTP endpoint to be used to parse messages."""

    def __init__(self, endpoint_config: Optional[EndpointConfig] = None) -> None:
        """Initializes a `RasaNLUHttpInterpreter`."""
        if endpoint_config:
            self.endpoint_config = endpoint_config
        else:
            self.endpoint_config = EndpointConfig(constants.DEFAULT_SERVER_URL)

    async def parse(self, message: UserMessage) -> Dict[Text, Any]:
        """Parse a text message.

        Return a default value if the parsing of the text failed.
        """
        default_return = {
            "intent": {INTENT_NAME_KEY: "", "confidence": 0.0},
            "entities": [],
            "text": "",
        }

        result = await self._rasa_http_parse(message.text, message.sender_id)
        return result if result is not None else default_return

    async def _rasa_http_parse(
        self, text: Text, message_id: Optional[Text] = None
    ) -> Optional[Dict[Text, Any]]:
        """Send a text message to a running rasa NLU http server.

        Return `None` on failure.
        """
        if not self.endpoint_config or self.endpoint_config.url is None:
            logger.error(
                f"Failed to parse text '{text}' using rasa NLU over http. "
                f"No rasa NLU server specified!"
            )
            return None

        params = {
            "token": self.endpoint_config.token,
            "text": text,
            "message_id": message_id,
        }

        if self.endpoint_config.url.endswith("/"):
            url = self.endpoint_config.url + "model/parse"
        else:
            url = self.endpoint_config.url + "/model/parse"

        # noinspection PyBroadException
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        response_text = await resp.text()
                        logger.error(
                            f"Failed to parse text '{text}' using rasa NLU over "
                            f"http. Error: {response_text}"
                        )
                        return None
        except Exception:  # skipcq: PYL-W0703
            # need to catch all possible exceptions when doing http requests
            # (timeouts, value errors, parser errors, ...)
            logger.exception(f"Failed to parse text '{text}' using rasa NLU over http.")
            return None
