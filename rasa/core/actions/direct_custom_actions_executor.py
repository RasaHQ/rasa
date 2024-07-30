from importlib.util import find_spec
from typing import (
    Any,
    Dict,
    Text,
    Optional,
)

import structlog
from rasa_sdk.executor import ActionExecutor

import rasa
from rasa.core.actions.custom_action_executor import (
    CustomActionExecutor,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.trackers import EventVerbosity
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

structlogger = structlog.get_logger(__name__)


class DirectCustomActionExecutor(CustomActionExecutor):
    def __init__(self, action_name: str, action_endpoint: EndpointConfig):
        """Initializes the direct custom action executor.

        Args:
            action_name: Name of the custom action.
            action_endpoint: The endpoint to execute custom actions.
        """
        self.action_name = action_name
        self.action_endpoint = action_endpoint
        self.action_executor = ActionExecutor()
        self.register_actions_from_a_module()

    def register_actions_from_a_module(self) -> None:
        module_name = self.action_endpoint.actions_module
        if not find_spec(module_name):
            raise RasaException(
                f"You've provided the custom actions module '{module_name}' "
                f"to run directly by the rasa server, however this module does "
                f"not exist. Please check for typos in your `endpoints.yml` file."
            )

        self.action_executor.register_package(module_name)

    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
    ) -> Dict[Text, Any]:
        structlogger.debug(
            "action.direct_custom_action_executor.run",
            action_name=self.action_name,
        )
        tracker_state = tracker.current_state(EventVerbosity.ALL)

        action_call = {
            "next_action": self.action_name,
            "sender_id": tracker.sender_id,
            "tracker": tracker_state,
            "version": rasa.__version__,
        }

        if domain:
            action_call["domain"] = domain.as_dict()

        return await self.action_executor.run(action_call)
