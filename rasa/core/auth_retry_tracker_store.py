import logging
from typing import Iterable, Optional, Text

from rasa.core.brokers.broker import EventBroker
from rasa.core.tracker_store import TrackerStore, create_tracker_store
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig

from rasa.core.secrets_manager.secret_manager import EndpointResolver

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3


class AuthRetryTrackerStore(TrackerStore):
    """Tracker store wrapper which implements retry mechanism in every abstract method.

    The retry mechanism first updates credentials from the secret manager.
    """

    def __init__(
        self,
        domain: "Domain",
        endpoint_config: "EndpointConfig",
        retries: int = DEFAULT_RETRIES,
        event_broker: Optional["EventBroker"] = None,
    ) -> None:
        """Initialise the tracker store wrapper."""
        self.retries = self.validate_retries(retries)
        self.endpoint_config = endpoint_config

        self._tracker_store = self.recreate_tracker_store(domain, event_broker)

        super().__init__(domain, event_broker)

    @property
    def domain(self) -> Domain:
        """Returns the domain of the wrapped tracker store."""
        return self._tracker_store.domain

    @domain.setter
    def domain(self, domain: Optional[Domain]) -> None:
        """Sets the domain of wrapped tracker store."""
        self._tracker_store.domain = domain or Domain.empty()

    @staticmethod
    def validate_retries(retries: int) -> int:
        """Validate the number of retries."""
        if retries <= 0:
            logger.warning(
                f"Invalid number of retries: {retries}. "
                f"Using default number of retries: {DEFAULT_RETRIES}."
            )
            return DEFAULT_RETRIES
        else:
            return retries

    async def keys(self) -> Iterable[Text]:
        """Retries retrieving the keys if it fails."""
        # add + 1 to retries because the retries are additional to the first attempt
        for _ in range(self.retries + 1):
            try:
                return await self._tracker_store.keys()
            except Exception as e:
                logger.warning("Failed to retrieve keys. Retrying...", exc_info=e)
                self._tracker_store = self.recreate_tracker_store(
                    self.domain, self.event_broker
                )
        else:
            logger.error(f"Failed to retrieve keys after {self.retries} retries.")
            return []

    async def retrieve(self, sender_id: Text) -> Optional["DialogueStateTracker"]:
        """Retries retrieving the tracker if it fails."""
        # add + 1 to retries because the retries are additional to the first attempt
        for _ in range(self.retries + 1):
            try:
                return await self._tracker_store.retrieve(sender_id)
            except Exception as e:
                logger.warning(
                    f"Failed to retrieve tracker for {sender_id}. Retrying...",
                    exc_info=e,
                )
                self._tracker_store = self.recreate_tracker_store(
                    self.domain, self.event_broker
                )
        else:
            logger.error(
                f"Failed to retrieve tracker for {sender_id} "
                f"after {self.retries} retries."
            )
            return None

    async def save(self, tracker: "DialogueStateTracker") -> None:
        """Retries saving the tracker if it fails."""
        # add + 1 to retries because the retries are additional to the first attempt
        for _ in range(self.retries + 1):
            try:
                await self._tracker_store.save(tracker)
                break
            except Exception as e:
                logger.warning(
                    f"Failed to save tracker for {tracker.sender_id}. Retrying...",
                    exc_info=e,
                )
                self._tracker_store = self.recreate_tracker_store(
                    self.domain, self.event_broker
                )
        else:
            logger.error(
                f"Failed to save tracker for {tracker.sender_id} "
                f"after {self.retries} retries."
            )

    def recreate_tracker_store(
        self, domain: "Domain", event_broker: Optional["EventBroker"] = None
    ) -> TrackerStore:
        """Recreate tracker store with updated credentials."""
        endpoint_config = EndpointResolver.update_config(self.endpoint_config)
        return create_tracker_store(endpoint_config, domain, event_broker)
