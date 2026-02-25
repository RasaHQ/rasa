from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Text

import redis
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
from rasa.core.tracker_stores.tracker_store import (
    SerializedTrackerAsText,
    TrackerDeserialisationException,
    TrackerStore,
)
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

    def _get_user_trackers_key(self, user_id: str) -> str:
        """Get the Redis key for storing sender_ids for a given user_id.

        Args:
            user_id: The user ID.

        Returns:
            Redis key for the user's tracker sorted set.
        """
        # When key_prefix is in use, use same key_prefix for the index
        # so all keys can be isolated by the prefix.
        if DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX in self.key_prefix:
            namespace = self.key_prefix.split(DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX)[0]
            return f"{namespace}user_trackers:{user_id}"
        return f"user_trackers:{user_id}"

    def _get_expiration_timestamp(self, ttl: Optional[float]) -> Optional[float]:
        """Calculate expiration timestamp from TTL.

        Args:
            ttl: Time-to-live in seconds.

        Returns:
            Expiration timestamp (current time + ttl) or None if ttl is None/0.
        """
        if ttl and ttl > 0:
            return time.time() + ttl
        return None

    def _normalize_sender_id(self, sender_id: Text) -> Text:
        """Normalize sender_id by removing key prefix if present.

        Args:
            sender_id: The sender ID to normalize.

        Returns:
            Normalized sender ID without key prefix.
        """
        if sender_id.startswith(self.key_prefix):
            return sender_id[len(self.key_prefix) :]
        return sender_id

    def _add_to_sorted_set_index(
        self, user_id: str, sender_id: str, ttl: Optional[float]
    ) -> None:
        """Add sender_id to the sorted set index with appropriate expiration.

        Args:
            user_id: The user ID.
            sender_id: The sender ID to add.
            ttl: Time-to-live in seconds, or None for no expiration.
        """
        user_trackers_key = self._get_user_trackers_key(user_id)
        expiration_timestamp = self._get_expiration_timestamp(ttl)
        if expiration_timestamp:
            # Use sorted set with expiration timestamp as score for per-member TTL
            self.red.zadd(user_trackers_key, {sender_id: expiration_timestamp})
        else:
            # No TTL: use a score of +inf to keep it indefinitely
            self.red.zadd(user_trackers_key, {sender_id: float("inf")})

    def _decode_sender_ids(self, sender_ids: List[Any]) -> List[str]:
        """Convert sender_ids from bytes to strings if needed.

        Args:
            sender_ids: List of sender IDs (may be bytes or strings).

        Returns:
            List of sender IDs as strings.
        """
        return [
            sid.decode("utf-8") if isinstance(sid, bytes) else sid for sid in sender_ids
        ]

    async def save(
        self, tracker: DialogueStateTracker, timeout: Optional[float] = None
    ) -> None:
        """Saves the current conversation state."""
        await self.stream_events(tracker)

        if not timeout and self.record_exp:
            timeout = self.record_exp

        # Normalize sender_id by removing key prefix if present
        sender_id = self._normalize_sender_id(tracker.sender_id)

        stored = self.red.get(self.key_prefix + sender_id)

        if stored is not None:
            try:
                prior_tracker = self.deserialise_tracker(sender_id, stored)
            except TrackerDeserialisationException as e:
                structlogger.error(
                    "redis_tracker_store.save.deserialization_failed",
                    event_info=f"Failed to deserialize prior tracker for "
                    f"'{sender_id}': {e}. Overwriting with new tracker.",
                )
                prior_tracker = DialogueStateTracker(sender_id, self.domain.slots)

            tracker = self._merge_trackers(prior_tracker, tracker)

        # Ensure conversation_started_timestamp is set (for backward compatibility)
        tracker.ensure_conversation_started_timestamp()

        serialised_tracker = self.serialise_tracker(tracker)
        self.red.set(self.key_prefix + sender_id, serialised_tracker, ex=timeout)

        # Maintain secondary index: add sender_id to user's tracker sorted set
        # A key assumption is that user_id is immutable for a given sender_id
        # i.e. a conversation ID is always associated with the same user ID.
        # Therefore, the index does not become stale.
        if tracker.user_id:
            ttl_to_use = timeout if timeout else self.record_exp
            self._add_to_sorted_set_index(tracker.user_id, sender_id, ttl_to_use)

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

        sender_id = self._normalize_sender_id(sender_id)

        # Before deleting, get the tracker to find user_id for index cleanup
        stored = self.red.get(self.key_prefix + sender_id)
        if stored:
            try:
                tracker = self.deserialise_tracker(sender_id, stored)
                # Remove sender_id from user's tracker sorted set
                if tracker and tracker.user_id:
                    user_trackers_key = self._get_user_trackers_key(tracker.user_id)
                    self.red.zrem(user_trackers_key, sender_id)
            except TrackerDeserialisationException:
                structlogger.error(
                    "redis_tracker_store.delete.deserialization_failed",
                    event_info=(
                        f"Failed to deserialize tracker for '{sender_id}'. "
                        f"Skipping index cleanup."
                    ),
                )

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
        sender_id = self._normalize_sender_id(sender_id)

        stored = self.red.get(self.key_prefix + sender_id)
        if stored is None:
            structlogger.debug(
                "redis_tracker_store.retrieve.no_tracker_for_sender_id",
                event_info=f"Could not find tracker for conversation ID '{sender_id}'.",
            )
            return None

        try:
            tracker = self.deserialise_tracker(sender_id, stored)
        except TrackerDeserialisationException as e:
            structlogger.error(
                "redis_tracker_store.retrieve.deserialization_failed",
                event_info=f"Failed to deserialize tracker for '{sender_id}': {e}",
            )
            return None
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
        # Ensure conversation_started_timestamp is set (for backward compatibility)
        tracker.ensure_conversation_started_timestamp()

        serialised_tracker = self.serialise_tracker(tracker)

        # Normalize sender_id by removing key prefix if present
        sender_id = self._normalize_sender_id(tracker.sender_id)

        self.red.set(
            self.key_prefix + sender_id, serialised_tracker, ex=self.record_exp
        )

        # Maintain secondary index: add sender_id to user's tracker sorted set
        if tracker.user_id:
            self._add_to_sorted_set_index(tracker.user_id, sender_id, self.record_exp)

        first_event_timestamp = str(datetime.fromtimestamp(tracker.events[0].timestamp))

        structlogger.info(
            "redis_tracker_store.update.updated_tracker",
            sender_id=tracker.sender_id,
            first_event_timestamp=first_event_timestamp,
        )

    async def get_trackers_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[DialogueStateTracker]:
        """Retrieves all trackers for a given user_id using efficient secondary index.

        Uses a Redis Sorted Set (user_trackers:{user_id}) to store all sender_ids for a
        user with per-member expiration timestamps, enabling O(1) lookup instead of
        scanning all keys.

        Args:
            user_id: User ID to fetch trackers for.
            limit: Optional maximum number of trackers to return. If None, returns all
                matching trackers.
            skip: Optional number of trackers to skip before returning results. If None,
                starts from the beginning.

        Returns:
            List of trackers associated with the user_id.
        """
        # Get all non-expired sender_ids for this user from the secondary index
        user_trackers_key = self._get_user_trackers_key(user_id)
        current_time = time.time()

        # Clean up expired members (score <= current_time) first
        expired_count = self.red.zremrangebyscore(
            user_trackers_key, min="-inf", max=current_time
        )
        if expired_count > 0:
            structlogger.debug(
                "redis_tracker_store.get_trackers_by_user_id.cleaned_expired_members",
                event_info=(
                    f"Cleaned up {expired_count} expired sender_ids from index "
                    f"for user_id '{user_id}'."
                ),
            )

        # Get all members with score > current_time (not expired)
        # ZRANGEBYSCORE returns members with scores in the range (current_time, +inf]
        sender_ids = self.red.zrangebyscore(
            user_trackers_key, min=current_time, max="+inf"
        )

        if not sender_ids:
            structlogger.debug(
                "redis_tracker_store.get_trackers_by_user_id.no_senders_for_user_id",
                event_info=f"No sender_ids found for user_id '{user_id}'.",
            )
            return []

        # Convert set members to strings if needed
        conversation_ids = self._decode_sender_ids(sender_ids)

        # Build tracker keys
        keys = [self.key_prefix + sender_id for sender_id in conversation_ids]

        if not keys:
            return []

        # Fetch all trackers in batch
        if isinstance(self.red, redis.RedisCluster):
            # Background context: https://redis.readthedocs.io/en/stable/clustering.html#multi-key-commands
            values = self.red.mget_nonatomic(keys)  # type: ignore[no-untyped-call]
        else:
            values = self.red.mget(keys)

        # Deserialize trackers
        trackers = self._retrieve_trackers_by_user_id(
            conversation_ids, user_trackers_key, values, user_id
        )

        # Sort by timestamp, then sender_id
        trackers.sort(key=self._sort_key)

        return self._apply_pagination(trackers, skip, limit)

    def _retrieve_trackers_by_user_id(
        self,
        conversation_ids: List[str],
        user_trackers_key: str,
        values: List[Optional[str]],
        user_id: str,
    ) -> List[DialogueStateTracker]:
        """Helper method to retrieve trackers by user_id from given keys and values."""
        trackers = []
        for sender_id, value in zip(conversation_ids, values):
            if value is None:
                # Tracker was deleted but index wasn't cleaned up - remove from index
                self.red.zrem(user_trackers_key, sender_id)
                continue

            try:
                tracker = self.deserialise_tracker(sender_id, value)
                if tracker and tracker.user_id == user_id:
                    trackers.append(tracker)
            except TrackerDeserialisationException:
                structlogger.error(
                    "redis_tracker_store.get_trackers_by_user_id.deserialization_failed",
                    event_info=(
                        f"Failed to deserialize tracker for sender_id "
                        f"'{sender_id}'. Skipping."
                    ),
                )
                continue

        return trackers
