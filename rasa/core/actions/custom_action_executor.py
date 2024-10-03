from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Any, Dict, Text

import rasa
from rasa.core.actions.action_exceptions import DomainNotFound
from rasa.core.actions.constants import DEFAULT_SELECTIVE_DOMAIN, SELECTIVE_DOMAIN
from rasa.shared.constants import DOCS_BASE_URL
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker


logger = logging.getLogger(__name__)


class CustomActionExecutor(abc.ABC):
    """Interface for custom action executors.

    Provides an abstraction layer for executing custom actions
    regardless of the communication protocol.
    """

    @abc.abstractmethod
    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        """Executes the custom action.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            The response from the execution of the custom action.
        """
        pass


class NoEndpointCustomActionExecutor(CustomActionExecutor):
    """Implementation of a custom action executor when endpoint is not set.

    Used to handle the case where no endpoint is configured.

    Raises RasaException when executed.
    """

    def __init__(self, action_name: str) -> None:
        """Initializes the custom action executor.

        Args:
            action_name: The name of the custom action.
        """
        self.action_name = action_name

    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        """Executes the custom action.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            The response from the execution of the custom action.
        """
        raise RasaException(
            f"Failed to execute custom action '{self.action_name}' "
            f"because no endpoint is configured to run this "
            f"custom action. Please take a look at "
            f"the docs and set an endpoint configuration via the "
            f"--endpoints flag. "
            f"{DOCS_BASE_URL}/custom-actions"
        )


class CustomActionRequestWriter:
    """Writes the request payload for a custom action."""

    def __init__(self, action_name: str, action_endpoint: EndpointConfig) -> None:
        """Initializes the request writer.

        Args:
            action_name: The name of the custom action.
            action_endpoint: The endpoint configuration for the action server.
        """
        self.action_name = action_name
        self.action_endpoint = action_endpoint

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

    def create(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> Dict[str, Any]:
        """Create the JSON payload for the action server request.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            A JSON payload to be sent to the action server.
        """
        from rasa.shared.core.trackers import EventVerbosity

        tracker_state = tracker.current_state(EventVerbosity.ALL)

        result = {
            "next_action": self.action_name,
            "sender_id": tracker.sender_id,
            "tracker": tracker_state,
            "version": rasa.__version__,
        }

        if include_domain and (
            not self._is_selective_domain_enabled()
            or domain.does_custom_action_explicitly_need_domain(self.action_name)
        ):
            result["domain"] = domain.as_dict()

        result["domain_digest"] = domain.fingerprint()

        return result


class RetryCustomActionExecutor(CustomActionExecutor):
    """Retries the execution of a custom action."""

    def __init__(self, custom_action_executor: CustomActionExecutor) -> None:
        self._custom_action_executor = custom_action_executor

    async def run(
        self,
        tracker: "DialogueStateTracker",
        domain: "Domain",
        include_domain: bool = False,
    ) -> Dict[Text, Any]:
        """Runs the wrapped custom action executor.

        First request to the action server is made with/without the domain
        as specified by the `include_domain` parameter.

        If the action server responds with a `DomainNotFound` error, by running the
        custom action executor again with the domain information.

        Args:
            tracker: The current state of the dialogue.
            domain: The domain object containing domain-specific information.
            include_domain: If True, the domain is included in the request.

        Returns:
            The response from the execution of the custom action.
        """
        try:
            return await self._custom_action_executor.run(
                tracker,
                domain,
                include_domain=include_domain,
            )
        except DomainNotFound:
            return await self._custom_action_executor.run(
                tracker, domain, include_domain=True
            )
