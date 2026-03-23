from __future__ import annotations

import itertools
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Text, Tuple

import structlog
from pymongo.synchronous.collection import Collection

from rasa.constants import USER_ID
from rasa.core.brokers.broker import EventBroker
from rasa.core.tracker_stores.tracker_store import SerializedTrackerAsText, TrackerStore
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    EventVerbosity,
    get_latest_replay_safe_session_tracker,
)

structlogger = structlog.get_logger(__name__)


class MongoTrackerStore(TrackerStore, SerializedTrackerAsText):
    """Stores conversation history in Mongo.

    Property methods:
        conversations: returns the current conversation

    Latest-session retrieval and incremental saves use the same session boundary
    as other tracker stores: ``ActionExecuted(action_session_start)``. That
    action is run by :class:`~rasa.core.processor.MessageProcessor` whenever a
    new session starts, so it is a reliable marker in stored histories. Slicing
    on ``SessionStarted`` was legacy and breaks when a custom
    ``action_session_start`` omits that event.
    """

    def __init__(
        self,
        domain: Domain,
        host: Optional[Text] = "mongodb://localhost:27017",
        db: Optional[Text] = "rasa",
        username: Optional[Text] = None,
        password: Optional[Text] = None,
        auth_source: Optional[Text] = "admin",
        collection: Text = "conversations",
        event_broker: Optional[EventBroker] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        from pymongo import MongoClient
        from pymongo.database import Database

        self.client: MongoClient = MongoClient(
            host,
            username=username,
            password=password,
            authSource=auth_source,
            # delay connect until process forking is done
            connect=False,
        )

        self.db = Database(self.client, db)
        self.collection = collection
        super().__init__(domain, event_broker, **kwargs)

        self._ensure_indices()

    @property
    def conversations(self) -> Collection:
        """Returns the current conversation."""
        return self.db[self.collection]

    def _ensure_indices(self) -> None:
        """Create indices on the sender_id and user_id."""
        self.conversations.create_index("sender_id")
        # Create index on user_id for efficient querying by user_id
        # This index is sparse (only indexes documents with user_id) to save space
        self.conversations.create_index(USER_ID, sparse=True)
        # Create compound index on conversation_started_timestamp and sender_id
        # for efficient sorting and pagination in get_trackers_by_user_id
        # This index is sparse (only indexes documents with
        # conversation_started_timestamp)
        self.conversations.create_index(
            [("conversation_started_timestamp", 1), ("sender_id", 1)], sparse=True
        )

    @staticmethod
    def _current_tracker_state_without_events(tracker: DialogueStateTracker) -> Dict:
        # get current tracker state and remove `events` key from state
        # since events are pushed separately in the `update_one()` operation
        state = tracker.current_state(EventVerbosity.ALL)
        state.pop("events", None)

        return state

    async def delete(self, sender_id: Text) -> None:
        """Delete tracker for the given sender_id.

        Args:
            sender_id: Sender id of the tracker to be deleted.
        """
        if not await self.exists(sender_id):
            structlogger.info(
                "mongo_tracker_store.delete.no_tracker_for_sender_id",
                event_info=f"Could not find tracker for conversation ID '{sender_id}'.",
            )
            return None
        self.conversations.delete_one({"sender_id": sender_id})

        structlogger.info(
            "mongo_tracker_store.delete.deleted_tracker",
            sender_id=sender_id,
        )

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Saves the current conversation state."""
        await self.stream_events(tracker)

        additional_events = self._additional_events(tracker)
        # Store conversation_started_timestamp for efficient sorting
        # This allows us to sort by timestamp at the database level
        # Ensure it's set on the tracker (for backward compatibility with old trackers)
        tracker.ensure_conversation_started_timestamp()

        # Prepare update document
        update_doc = {
            "$set": self._current_tracker_state_without_events(tracker),
            "$push": {"events": {"$each": [e.as_dict() for e in additional_events]}},
        }

        self.conversations.update_one(
            {"sender_id": tracker.sender_id},
            update_doc,
            upsert=True,
        )

    def _additional_events(self, tracker: DialogueStateTracker) -> Iterator:
        """Return events from the tracker which aren't currently stored.

        The offset matches the length of the tracker returned by
        :meth:`retrieve` (replay-safe latest session with ``action_session_start``
        boundaries), not only the suffix length from the last matching action in
        storage.

        Args:
            tracker: Tracker to inspect.

        Returns:
            List of serialised events that aren't currently stored.

        """
        stored = self.conversations.find_one({"sender_id": tracker.sender_id}) or {}
        all_events = self._events_from_serialized_tracker(stored)

        if self.domain.is_empty():
            # Fallback for domain-less operation: use a simple offset based on the
            # number of already-persisted events for this sender_id so that new
            # events are still stored even when no domain is loaded.
            offset = len(all_events)
            return itertools.islice(tracker.events, offset, len(tracker.events))

        if not all_events:
            # Nothing persisted yet — all in-memory events are new (matches legacy
            # suffix-length 0 => islice from 0).
            return itertools.islice(tracker.events, 0, len(tracker.events))

        stored_tracker = DialogueStateTracker.from_dict(
            tracker.sender_id,
            all_events,
            self.domain.slots,
            user_id=stored.get(USER_ID),
        )
        sliced = get_latest_replay_safe_session_tracker(
            stored_tracker,
            start_session_after_expiry=(
                self.domain.session_config.start_session_after_expiry
            ),
        )
        offset = len(sliced.events)

        return itertools.islice(tracker.events, offset, len(tracker.events))

    @staticmethod
    def _events_from_serialized_tracker(serialised: Dict) -> List[Dict]:
        return serialised.get("events", [])

    @staticmethod
    def _events_since_last_action_session_start(events: List[Dict]) -> List[Dict]:
        """Events from the latest ``action_session_start`` action onwards (inclusive).

        Args:
            events: All events for a conversation ID.

        Returns:
            Serialised events from the latest ``ActionExecuted(action_session_start)``
            onward. Returns all events if no such action is found.

        """
        events_after_session_start = []
        for event in reversed(events):
            events_after_session_start.append(event)
            if (
                event.get("event") == ActionExecuted.type_name
                and event.get("name") == ACTION_SESSION_START_NAME
            ):
                break

        return list(reversed(events_after_session_start))

    async def _retrieve(
        self, sender_id: Text, fetch_events_from_all_sessions: bool
    ) -> Optional[Tuple[List[Dict[Text, Any]], Optional[Text]]]:
        stored = self.conversations.find_one({"sender_id": sender_id})

        # look for conversations which have used an `int` sender_id in the past
        # and update them.
        if not stored and sender_id.isdigit():
            from pymongo import ReturnDocument

            stored = self.conversations.find_one_and_update(
                {"sender_id": int(sender_id)},
                {"$set": {"sender_id": str(sender_id)}},
                return_document=ReturnDocument.AFTER,
            )

        if not stored:
            return None

        events = self._events_from_serialized_tracker(stored)

        if not fetch_events_from_all_sessions:
            events = self._events_since_last_action_session_start(events)

        # Return both events and user_id
        return events, stored.get(USER_ID)

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for the latest conversation session.

        The latest session is the replay-safe slice from the last
        ``action_session_start`` action (see
        :func:`get_latest_replay_safe_session_tracker`).
        """
        result = await self._retrieve(sender_id, fetch_events_from_all_sessions=True)

        if result is None:
            return None

        events, user_id = result

        if not events:
            return None

        tracker = DialogueStateTracker.from_dict(
            sender_id, events, self.domain.slots, user_id=user_id
        )
        return get_latest_replay_safe_session_tracker(
            tracker,
            start_session_after_expiry=(
                self.domain.session_config.start_session_after_expiry
            ),
        )

    async def retrieve_full_tracker(
        self, conversation_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Fetching all tracker events across conversation sessions."""
        result = await self._retrieve(
            conversation_id, fetch_events_from_all_sessions=True
        )

        if result is None:
            return None

        events, user_id = result

        if not events:
            return None

        return DialogueStateTracker.from_dict(
            conversation_id,
            events,
            self.domain.slots,
            user_id=user_id,
        )

    async def keys(self) -> Iterable[Text]:
        """Returns sender_ids of the Mongo Tracker Store."""
        return [c["sender_id"] for c in self.conversations.find()]

    async def update(
        self, tracker: DialogueStateTracker, apply_deletion_only: bool = True
    ) -> None:
        """Overwrites the tracker for the given sender_id."""
        # Ensure conversation_started_timestamp is set (for backward compatibility)
        tracker.ensure_conversation_started_timestamp()

        self.conversations.replace_one(
            {"sender_id": tracker.sender_id},
            tracker.current_state(EventVerbosity.ALL),
            upsert=True,
        )

        first_event_timestamp = str(datetime.fromtimestamp(tracker.events[0].timestamp))

        structlogger.info(
            "mongo_tracker_store.update.updated_tracker",
            sender_id=tracker.sender_id,
            first_event_timestamp=first_event_timestamp,
        )

    async def get_trackers_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[DialogueStateTracker]:
        """Retrieves all trackers for a given user_id.

        Uses MongoDB query to efficiently find trackers by user_id,
        leveraging the user_id index for optimal performance.

        Note: MongoDB cursors automatically batch results, but all matching
        trackers are loaded into memory. For users with a very large number
        of conversations (thousands), use the limit parameter for pagination.

        Args:
            user_id: User ID to fetch trackers for.
            limit: Optional maximum number of trackers to return. If None, returns all
                matching trackers. Useful for pagination.
            skip: Optional number of trackers to skip before returning results. If None,
                starts from the beginning. Useful for pagination.

        Returns:
            List of trackers associated with the user_id.
        """
        trackers = []
        # Use MongoDB aggregation pipeline to efficiently sort by
        # conversation_started_timestamp. This handles both documents with
        # conversation_started_timestamp field (new) and documents without it
        # (old) by extracting from events[0].timestamp
        pipeline = [
            {"$match": {USER_ID: user_id}},
            {
                "$addFields": {
                    "sort_timestamp": {
                        "$ifNull": [
                            "$conversation_started_timestamp",
                            {"$arrayElemAt": ["$events.timestamp", 0]},
                        ]
                    }
                }
            },
            {"$sort": {"sort_timestamp": 1, "sender_id": 1}},
        ]

        # Apply skip and limit at the database level for efficiency
        if skip is not None and skip > 0:
            pipeline.append({"$skip": skip})
        if limit is not None and limit > 0:
            pipeline.append({"$limit": limit})

        # Remove the temporary sort_timestamp field before returning
        pipeline.append({"$project": {"sort_timestamp": 0}})

        # Execute aggregation pipeline
        for doc in self.conversations.aggregate(pipeline):
            sender_id = doc.get("sender_id")
            if not sender_id:
                continue

            # Reconstruct tracker from the MongoDB document
            events = self._events_from_serialized_tracker(doc)
            # Get conversation_started_timestamp from document if available
            # DialogueStateTracker.from_dict() will extract from events if not provided
            conversation_started_timestamp = doc.get("conversation_started_timestamp")
            tracker = DialogueStateTracker.from_dict(
                sender_id,
                events,
                self.domain.slots,
                # user_id should be present as we queried by it
                user_id=doc.get(USER_ID),
                conversation_started_timestamp=conversation_started_timestamp,
            )
            trackers.append(tracker)

        return trackers
