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

structlogger = structlog.get_logger(__name__)


class DirectCustomActionExecutor(CustomActionExecutor):
    def __init__(self, action_name: str, actions_module: Text):
        """Initializes the direct custom action executor.

        Args:
            action_name: Name of the custom action.
            actions_module: The name of the module containing all custom actions.
        """
        self.action_name = action_name
        self.action_executor = ActionExecutor()
        self.action_executor.register_package(actions_module)

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
