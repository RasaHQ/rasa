from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Text

import structlog
from pydantic import ValidationError

import rasa.shared
from rasa.core.brokers.broker import EventBroker
from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    SupportedServiceType,
)
from rasa.core.redis_connection_factory import (
    DeploymentMode,
    RedisConfig,
    RedisConnectionFactory,
)
from rasa.core.tracker_stores.tracker_store import SerializedTrackerAsText, TrackerStore
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException

structlogger = structlog.get_logger(__name__)

# default value for key prefix in RedisTrackerStore
DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX = "tracker:"


class RedisTrackerStore(TrackerStore, SerializedTrackerAsText):
    """Stores conversation history in Redis."""

    def __init__(
        self,
        domain: Domain,
        host: Text = "localhost",
        port: int = 6379,
        db: int = 0,
        username: Optional[Text] = None,
        password: Optional[Text] = None,
        event_broker: Optional[EventBroker] = None,
        record_exp: Optional[float] = None,
        key_prefix: Optional[Text] = None,
        use_ssl: bool = False,
        ssl_keyfile: Optional[Text] = None,
        ssl_certfile: Optional[Text] = None,
        ssl_ca_certs: Optional[Text] = None,
        deployment_mode: Text = DeploymentMode.STANDARD.value,
        endpoints: Optional[list] = None,
        sentinel_service: Optional[Text] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        """Initializes the tracker store."""

        # Create Redis connection using the factory directly
        try:
            config = RedisConfig(
                host=host,
                port=port,
                service_type=SupportedServiceType.TRACKER_STORE,
                db=db,
                username=username,
                password=password,
                use_ssl=use_ssl,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                ssl_ca_certs=ssl_ca_certs,
                deployment_mode=deployment_mode,
                endpoints=endpoints,
                sentinel_service=sentinel_service,
                decode_responses=True,
            )
            self.red = RedisConnectionFactory.create_connection(config)
        except ValidationError as e:
            raise RasaException(f"Invalid Redis configuration: {e}")

        self.record_exp = record_exp

        self.key_prefix = DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX
        if key_prefix:
            structlogger.debug(
                "redis_tracker_store.init.custom_key_prefix",
                event_info=f"Setting non-default redis key prefix: '{key_prefix}'.",
            )
            self._set_key_prefix(key_prefix)

        super().__init__(domain, event_broker, **kwargs)

    def _set_key_prefix(self, key_prefix: Text) -> None:
        if isinstance(key_prefix, str) and key_prefix.isalnum():
            self.key_prefix = key_prefix + ":" + DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX
        else:
            structlogger.warning(
                "redis_tracker_store.init.invalid_key_prefix",
                event_info=(
                    f"Omitting provided non-alphanumeric "
                    f"redis key prefix: '{key_prefix}'. "
                    f"Using default '{self.key_prefix}' instead."
                ),
            )

    def _get_key_prefix(self) -> Text:
        return self.key_prefix

    async def save(
        self, tracker: DialogueStateTracker, timeout: Optional[float] = None
    ) -> None:
        """Saves the current conversation state."""
        await self.stream_events(tracker)

        if not timeout and self.record_exp:
            timeout = self.record_exp

        # if the sender_id starts with the key prefix, we remove it
        # this is used to avoid storing the prefix twice
        sender_id = tracker.sender_id
        if sender_id.startswith(self.key_prefix):
            sender_id = sender_id[len(self.key_prefix) :]

        stored = self.red.get(self.key_prefix + sender_id)

        if stored is not None:
            prior_tracker = self.deserialise_tracker(sender_id, stored)

            tracker = self._merge_trackers(prior_tracker, tracker)

        serialised_tracker = self.serialise_tracker(tracker)
        self.red.set(self.key_prefix + sender_id, serialised_tracker, ex=timeout)

    async def delete(self, sender_id: Text) -> None:
        """Delete tracker for the given sender_id.

        Args:
            sender_id: Sender id of the tracker to be deleted.
        """
        if not await self.exists(sender_id):
            structlogger.info(
                "redis_tracker_store.delete.no_tracker_for_sender_id",
                event_info=f"Could not find tracker for conversation ID '{sender_id}'.",
            )
            return None

        if sender_id.startswith(self.key_prefix):
            sender_id = sender_id[len(self.key_prefix) :]

        self.red.delete(self.key_prefix + sender_id)
        structlogger.info(
            "redis_tracker_store.delete.deleted_tracker",
            sender_id=sender_id,
        )

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for the latest conversation session.

        The Redis key is formed by appending a prefix to sender_id.

        Args:
            sender_id: Conversation ID to fetch the tracker for.

        Returns:
            Tracker containing events from the latest conversation sessions.
        """
        return await self._retrieve(sender_id, fetch_all_sessions=False)

    async def retrieve_full_tracker(
        self, sender_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for all conversation sessions.

        The Redis key is formed by appending a prefix to sender_id.

        Args:
            sender_id: Conversation ID to fetch the tracker for.

        Returns:
            Tracker containing events from all conversation sessions.
        """
        return await self._retrieve(sender_id, fetch_all_sessions=True)

    async def _retrieve(
        self, sender_id: Text, fetch_all_sessions: bool
    ) -> Optional[DialogueStateTracker]:
        """Returns tracker matching sender_id.

        Args:
            sender_id: Conversation ID to fetch the tracker for.
            fetch_all_sessions: Whether to fetch all sessions or only the last one.
        """
        if sender_id.startswith(self.key_prefix):
            sender_id = sender_id[len(self.key_prefix) :]

        stored = self.red.get(self.key_prefix + sender_id)
        if stored is None:
            structlogger.debug(
                "redis_tracker_store.retrieve.no_tracker_for_sender_id",
                event_info=f"Could not find tracker for conversation ID '{sender_id}'.",
            )
            return None

        tracker = self.deserialise_tracker(sender_id, stored)
        if fetch_all_sessions:
            return tracker

        # only return the last session
        multiple_tracker_sessions = (
            rasa.shared.core.trackers.get_trackers_for_conversation_sessions(tracker)
        )

        if len(multiple_tracker_sessions) <= 1:
            return tracker

        return multiple_tracker_sessions[-1]

    async def keys(self) -> Iterable[Text]:
        """Returns keys of the Redis Tracker Store."""
        return self.red.keys(self.key_prefix + "*")

    @staticmethod
    def _merge_trackers(
        prior_tracker: DialogueStateTracker, tracker: DialogueStateTracker
    ) -> DialogueStateTracker:
        """Merges two trackers.

        Args:
            prior_tracker: Tracker containing events from the previous conversation
                sessions.
            tracker: Tracker containing events from the current conversation session.
        """
        if not prior_tracker.events:
            return tracker

        last_event_timestamp = prior_tracker.events[-1].timestamp
        past_tracker = tracker.travel_back_in_time(target_time=last_event_timestamp)

        if past_tracker.events == prior_tracker.events:
            return tracker

        merged = tracker.init_copy()
        merged.update_with_events(list(prior_tracker.events), override_timestamp=False)

        for new_event in tracker.events:
            # Event subclasses implement `__eq__` method that make it difficult
            # to compare events. We use `as_dict` to compare events.
            if all(
                [
                    new_event.as_dict() != existing_event.as_dict()
                    for existing_event in merged.events
                ]
            ):
                merged.update(new_event)

        return merged

    async def update(self, tracker: DialogueStateTracker) -> None:
        """Overwrites the tracker for the given sender_id."""
        serialised_tracker = self.serialise_tracker(tracker)

        # if the sender_id starts with the key prefix, we remove it
        # this is used to avoid storing the prefix twice
        sender_id = tracker.sender_id
        if sender_id.startswith(self.key_prefix):
            sender_id = sender_id[len(self.key_prefix) :]

        self.red.set(
            self.key_prefix + sender_id, serialised_tracker, ex=self.record_exp
        )

        first_event_timestamp = str(datetime.fromtimestamp(tracker.events[0].timestamp))

        structlogger.info(
            "redis_tracker_store.update.updated_tracker",
            sender_id=tracker.sender_id,
            first_event_timestamp=first_event_timestamp,
        )
