import json
import logging
from typing import Dict, Any, Text, TYPE_CHECKING

import aiohttp

from rasa.core.actions.action import ActionExecutionRejection
from rasa.core.actions.custom_action_executor import CustomActionExecutor
from rasa.core.constants import (
    DEFAULT_REQUEST_TIMEOUT,
    COMPRESS_ACTION_SERVER_REQUEST_ENV_NAME,
    DEFAULT_COMPRESS_ACTION_SERVER_REQUEST,
)
from rasa.shared.exceptions import RasaException
from rasa.utils.common import get_bool_env_variable
from rasa.utils.endpoints import EndpointConfig, ClientResponseError

if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker
    from rasa.shared.core.domain import Domain


logger = logging.getLogger(__name__)


class HTTPCustomActionExecutor(CustomActionExecutor):
    """HTTP-based implementation of the CustomActionExecutor.

    Executes custom actions by making HTTP POST requests to the action endpoint.
    """

    def __init__(self, name: str, action_endpoint: EndpointConfig) -> None:
        super().__init__(name, action_endpoint)

    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> Dict[str, Any]:
        """Execute the custom action using an HTTP POST request.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.

        Returns:
            A dictionary containing the response from the custom action endpoint.

        Raises:
            RasaException: If an error occurs while making the HTTP request.
        """
        try:
            logger.debug(
                "Calling action endpoint to run action '{}'.".format(self.name())
            )

            json_body = self._action_call_format(
                tracker=tracker, domain=domain, should_include_domain=False
            )

            should_compress = get_bool_env_variable(
                COMPRESS_ACTION_SERVER_REQUEST_ENV_NAME,
                DEFAULT_COMPRESS_ACTION_SERVER_REQUEST,
            )

            response = await self._perform_request_with_retries(
                json_body, should_compress, tracker, domain
            )

            if response is None:
                response = {}

            return response

        except ClientResponseError as e:
            if e.status == 400:
                response_data = json.loads(e.text)
                exception = ActionExecutionRejection(
                    response_data["action_name"], response_data.get("error")
                )
                logger.error(exception.message)
                raise exception
            else:
                raise RasaException(
                    f"Failed to execute custom action '{self.name()}'"
                ) from e

        except aiohttp.ClientConnectionError as e:
            logger.error(
                f"Failed to run custom action '{self.name()}'. Couldn't connect "
                f"to the server at '{self.action_endpoint.url}'. "
                f"Is the server running? "
                f"Error: {e}"
            )
            raise RasaException(
                f"Failed to execute custom action '{self.name()}'. Couldn't connect "
                f"to the server at '{self.action_endpoint.url}."
            )

        except aiohttp.ClientError as e:
            # not all errors have a status attribute, but
            # helpful to log if they got it

            # noinspection PyUnresolvedReferences
            status = getattr(e, "status", None)
            raise RasaException(
                "Failed to run custom action '{}'. Action server "
                "responded with a non 200 status code of {}. "
                "Make sure your action server properly runs actions "
                "and returns a 200 once the action is executed. "
                "Error: {}".format(self.name(), status, e)
            )

    def name(self) -> Text:
        return self._name

    async def _perform_request_with_retries(
        self,
        json_body: Dict[str, Any],
        should_compress: bool,
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> Any:
        """Attempts to perform the request with retries if necessary."""
        assert self.action_endpoint is not None
        number_of_retries = 2
        for _ in range(number_of_retries):
            try:
                return await self.action_endpoint.request(
                    json=json_body,
                    method="post",
                    timeout=DEFAULT_REQUEST_TIMEOUT,
                    compress=should_compress,
                )
            except ClientResponseError as e:
                # Repeat the request because Domain was not in the payload
                if e.status == 449:
                    json_body = self._action_call_format(
                        tracker=tracker, domain=domain, should_include_domain=True
                    )
                    continue
                raise e
