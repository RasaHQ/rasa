import json
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import aiohttp

from rasa.core.actions.action_exceptions import ActionExecutionRejection
from rasa.core.actions.constants import MISSING_DOMAIN_MARKER
from rasa.core.actions.custom_action_executor import (
    ActionResult,
    ActionResultType,
    CustomActionExecutor,
    CustomActionRequestWriter,
)
from rasa.core.constants import (
    COMPRESS_ACTION_SERVER_REQUEST_ENV_NAME,
    DEFAULT_COMPRESS_ACTION_SERVER_REQUEST,
    DEFAULT_REQUEST_TIMEOUT,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.common import get_bool_env_variable
from rasa.utils.endpoints import ClientResponseError, EndpointConfig

if TYPE_CHECKING:
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker


logger = logging.getLogger(__name__)


class HTTPCustomActionExecutor(CustomActionExecutor):
    """HTTP-based implementation of the CustomActionExecutor.

    Executes custom actions by making HTTP POST requests to the action endpoint.
    """

    def __init__(
        self,
        action_name: str,
        action_endpoint: EndpointConfig,
    ) -> None:
        self.action_name = action_name
        self.action_endpoint = action_endpoint
        self.request_writer = CustomActionRequestWriter(action_name, action_endpoint)
        self.should_compress = get_bool_env_variable(
            COMPRESS_ACTION_SERVER_REQUEST_ENV_NAME,
            DEFAULT_COMPRESS_ACTION_SERVER_REQUEST,
        )

    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        include_domain: bool = False,
    ) -> Dict[str, Any]:
        """Execute the custom action using an HTTP POST request.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            A dictionary containing the response from the custom action endpoint.
            Returns empty dict if domain is missing (449 status).

        Raises:
            RasaException: If an error occurs while making the HTTP request
                (other than missing domain).
        """
        result = await self.run_with_result(tracker, domain, include_domain)

        # Return empty dict for retry cases to avoid raising exceptions
        # RetryCustomActionExecutor will handle the retry logic
        if result.result_type == ActionResultType.RETRY_WITH_DOMAIN:
            return {}

        return result.response if result.response is not None else {}

    async def run_with_result(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
        include_domain: bool = False,
    ) -> ActionResult:
        """Execute the custom action and return an ActionResult.

        This method avoids raising DomainNotFound exception for 449 status code,
        instead returning an ActionResult with RETRY_WITH_DOMAIN type.
        This prevents tracing from capturing this expected condition as an error.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            ActionResult containing the response and result type.
        """
        from rasa.core.actions.action import RemoteActionJSONValidator

        try:
            logger.debug(
                "Calling action endpoint to run action '{}'.".format(self.action_name)
            )

            json_body = self.request_writer.create(
                tracker=tracker, domain=domain, include_domain=include_domain
            )

            assert self.action_endpoint is not None
            response = await self.action_endpoint.request(
                json=json_body,
                method="post",
                timeout=DEFAULT_REQUEST_TIMEOUT,
                compress=self.should_compress,
            )

            # Check if we got the special marker for 449 status (missing domain)
            if isinstance(response, dict) and response.get(MISSING_DOMAIN_MARKER):
                return ActionResult(result_type=ActionResultType.RETRY_WITH_DOMAIN)

            if response is None:
                response = {}

            RemoteActionJSONValidator.validate(response)
            return ActionResult(result_type=ActionResultType.SUCCESS, response=response)

        except ClientResponseError as e:
            if e.status == 400:
                response_data = json.loads(e.text)
                exception = ActionExecutionRejection(
                    response_data["action_name"], response_data.get("error")
                )
                logger.error(exception.message)
                raise exception
            elif e.status == 404:
                message = (
                    f"Custom action implementation with name '{self.action_name}' "
                    f"not found."
                )
                logger.error(message)
                raise RasaException(message) from e
            else:
                raise RasaException(
                    f"Failed to execute custom action '{self.action_name}'"
                ) from e

        except aiohttp.ClientConnectionError as e:
            logger.error(
                f"Failed to run custom action '{self.action_name}'. Couldn't connect "
                f"to the server at '{self.action_endpoint.url}'. "
                f"Is the server running? "
                f"Error: {e}"
            )
            raise RasaException(
                f"Failed to execute custom action '{self.action_name}'. "
                f"Couldn't connect to the server at '{self.action_endpoint.url}."
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
                "Error: {}".format(self.action_name, status, e)
            )
