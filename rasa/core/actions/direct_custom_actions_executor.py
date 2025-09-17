from functools import lru_cache
from importlib.util import find_spec
from typing import (
    Any,
    ClassVar,
    Dict,
    Text,
)

import structlog
from rasa_sdk.executor import ActionExecutor

import rasa
from rasa.core.actions.custom_action_executor import (
    CustomActionExecutor,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

structlogger = structlog.get_logger(__name__)


class DirectCustomActionExecutor(CustomActionExecutor):
    _actions_module_registered: ClassVar[bool] = False

    def __init__(self, action_name: str, action_endpoint: EndpointConfig):
        """Initializes the direct custom action executor.

        Args:
            action_name: Name of the custom action.
            action_endpoint: The endpoint to execute custom actions.
        """
        self.action_name = action_name
        self.action_endpoint = action_endpoint
        self.action_executor = self._create_action_executor()

    @staticmethod
    @lru_cache(maxsize=1)
    def _create_action_executor() -> ActionExecutor:
        """Creates and returns a cached ActionExecutor instance.

        Returns:
            ActionExecutor: The cached ActionExecutor instance.
        """
        return ActionExecutor()

    def register_actions_from_a_module(self) -> None:
        """Registers actions from the specified module if not already registered.

        This method checks if the actions module has already been registered to prevent
        duplicate registrations. If not registered, it attempts to register the actions
        module specified in the action endpoint configuration. If the module does not
        exist, it raises a RasaException.

        Raises:
            RasaException: If the actions module specified does not exist.
        """
        if DirectCustomActionExecutor._actions_module_registered:
            return

        module_name = self.action_endpoint.actions_module
        if not find_spec(module_name):
            raise RasaException(
                f"You've provided the custom actions module '{module_name}' "
                f"to run directly by the rasa server, however this module does "
                f"not exist. Please check for typos in your `endpoints.yml` file."
            )

        self.action_executor.register_package(module_name)
        DirectCustomActionExecutor._actions_module_registered = True

    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        """Executes the custom action directly.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            The response from the execution of the custom action.

        Raises:
            RasaException: If the actions module specified does not exist.
        """
        structlogger.debug(
            "action.direct_custom_action_executor.run",
            action_name=self.action_name,
        )

        # Register actions module if not already registered.
        # This is done here instead of __init__ to allow proper
        # exception handling and avoid hanging conversations.
        self.register_actions_from_a_module()
        self.action_executor.reload()

        tracker_state = tracker.current_state(EventVerbosity.ALL)
        action_call = {
            "next_action": self.action_name,
            "sender_id": tracker.sender_id,
            "tracker": tracker_state,
            "version": rasa.__version__,
        }

        if domain:
            action_call["domain"] = domain.as_dict()

        result = await self.action_executor.run(action_call)
        return result.model_dump() if result else {}
