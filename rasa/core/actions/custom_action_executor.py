import abc
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, Text

import rasa
from rasa.core.actions.constants import SELECTIVE_DOMAIN, DEFAULT_SELECTIVE_DOMAIN
from rasa.exceptions import ModelNotFound
from rasa.model import get_local_model
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker
    from rasa.shared.core.domain import Domain


logger = logging.getLogger(__name__)


class CustomActionExecutor(abc.ABC):
    """Interface for custom action executors.

    Provides an abstraction layer for executing custom actions
    regardless of the communication protocol.
    """

    def __init__(self, name: str, action_endpoint: EndpointConfig) -> None:
        self._name = name
        self.action_endpoint = action_endpoint

    def _get_domain_digest(self) -> Optional[str]:
        try:
            return get_local_model()
        except ModelNotFound as e:
            logger.warning(
                f"Model not found while running the action '{self._name}'.", exc_info=e
            )
            return None

    def _is_selective_domain_enabled(self) -> bool:
        """Check if selective domain handling is enabled.

        Returns:
            True if selective domain handling is enabled, otherwise False.
        """
        if self.action_endpoint is None:
            return False
        return bool(
            self.action_endpoint.kwargs.get(SELECTIVE_DOMAIN, DEFAULT_SELECTIVE_DOMAIN)
        )

    def _action_call_format(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        should_include_domain: bool = True,
    ) -> Dict[str, Any]:
        """Create the JSON payload for the action server request.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            should_include_domain: If domain context should be in the payload.

        Returns:
            A JSON payload to be sent to the action server.
        """
        from rasa.shared.core.trackers import EventVerbosity

        tracker_state = tracker.current_state(EventVerbosity.ALL)

        result = {
            "next_action": self._name,
            "sender_id": tracker.sender_id,
            "tracker": tracker_state,
            "version": rasa.__version__,
        }

        if should_include_domain and (
            not self._is_selective_domain_enabled()
            or domain.does_custom_action_explicitly_need_domain(self._name)
        ):
            result["domain"] = domain.as_dict()

        if domain_digest := self._get_domain_digest():
            result["domain_digest"] = domain_digest

        return result

    @abc.abstractmethod
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> Dict[Text, Any]:
        """Executes the custom action.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.

        Returns:
            The response from the execution of the custom action.
        """
        pass
