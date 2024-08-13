from typing import (
    Any,
    Dict,
    Text,
    Optional,
)

import structlog

from rasa.core.actions.custom_action_executor import (
    CustomActionExecutor,
)
from rasa.e2e_test.constants import KEY_MOCK_CUSTOM_ACTIONS
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

structlogger = structlog.get_logger(__name__)


class E2EStubCustomActionExecutor(CustomActionExecutor):
    def __init__(
        self,
        action_name: str,
        action_endpoint: EndpointConfig,
    ):
        """Initializes the e2e stub custom action executor.

        Args:
            action_name: Name of the custom action.
            action_endpoint: The endpoint to execute custom actions.
        """
        self.action_name = action_name
        self.action_endpoint = action_endpoint
        self.mock_custom_action = self.get_and_validate_mocked_custom_action()

    def get_and_validate_mocked_custom_action(self):
        mock_custom_actions = self.action_endpoint.kwargs.get(KEY_MOCK_CUSTOM_ACTIONS)
        if mock_custom_action := mock_custom_actions.get(self.action_name):
            return mock_custom_action
        else:
            # TODO Update message below with reference to the docs
            raise RasaException(f"Action `{self.action_name}` has not been mocked.")

    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: Optional["Domain"] = None,
    ) -> Dict[Text, Any]:
        structlogger.debug(
            "action.e2e_stub_custom_action_executor.run",
            action_name=self.action_name,
        )
        return self.mock_custom_action.as_dict()
