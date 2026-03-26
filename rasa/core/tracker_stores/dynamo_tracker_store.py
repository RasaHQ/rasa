from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Text

import structlog
from boto3.dynamodb.conditions import Attr, Key
from botocore.config import Config
from botocore.exceptions import ClientError

import rasa.utils
from rasa.constants import DEFAULT_SANIC_WORKERS, ENV_SANIC_WORKERS, USER_ID
from rasa.core.tracker_stores.tracker_store import (
    SerializedTrackerAsDict,
    TrackerStore,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig
from rasa.utils.json_utils import (
    replace_floats_with_decimals,
)

structlogger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    import boto3.resources.factory.dynamodb.Table


def _deduplicate_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate events that may have been appended more than once.

    Duplicate events arise when ``save()`` is called more than once for the
    same tracker turn (e.g. due to a retry or a race condition during load
    testing).  Replaying such events causes ``jsonpatch`` to attempt
    operations on fields that were already mutated by the first replay
    (e.g. removing ``frame_type`` that was already removed), which raises a
    ``JsonPatchConflict``.

    Each event is uniquely identified by its full serialised form.  Using the
    complete dict — rather than just ``(event_type, timestamp)`` — is
    necessary because multiple events can legitimately share the same
    timestamp (e.g. several slot-reset events emitted in the same batch each
    have distinct ``name`` fields).

    Args:
        events: Ordered list of event dicts as returned by DynamoDB after
            float conversion.

    Returns:
        The same list with consecutive or non-consecutive duplicate entries
        removed, preserving the original order of first occurrence.
    """
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for event in events:
        try:
            key = json.dumps(event, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Unserializable event — keep it to avoid silently dropping data.
            deduped.append(event)
            continue
        if key not in seen:
            seen.add(key)
            deduped.append(event)
    return deduped


class DynamoTrackerStore(TrackerStore, SerializedTrackerAsDict):
    """Stores conversation history in DynamoDB."""

    def __init__(
        self,
        domain: Domain,
        table_name: Text = "states",
        region: Text = "us-east-1",
        event_broker: Optional[EndpointConfig] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        """Initialize `DynamoTrackerStore`.

        Args:
            domain: Domain associated with this tracker store.
            table_name: The name of the DynamoDB table, does not need to be present a
                priori.
            region: The name of the region associated with the client.
                A client is associated with a single region.
            event_broker: An event broker used to publish events.
            kwargs: Additional kwargs.
        """
        import boto3

        self.client = boto3.client("dynamodb", region_name=region)
        max_pool_connections = kwargs.pop("max_pool_connections", 50)
        self._dynamo = boto3.resource(
            "dynamodb",
            region_name=region,
            config=Config(max_pool_connections=max_pool_connections),
        )
        self.region = region
        self.table_name = table_name
        self.db = self.get_or_create_table(table_name)
        super().__init__(domain, event_broker, **kwargs)

    def get_or_create_table(
        self, table_name: Text
    ) -> "boto3.resources.factory.dynamodb.Table":
        """Returns table or creates one if the table name is not in the table list.

        Note: This method only creates the base table with the primary key (sender_id).
        Global Secondary Indexes (GSIs) must be created manually. See
        `get_trackers_by_user_id` docstring for GSI creation instructions.
        """
        try:
            self.client.describe_table(TableName=table_name)
        except self.client.exceptions.ResourceNotFoundException:
            sanic_workers_count = int(
                os.environ.get(ENV_SANIC_WORKERS, DEFAULT_SANIC_WORKERS)
            )

            if sanic_workers_count > 1:
                structlogger.error(
                    "dynamo_tracker_store.table_creation_not_supported_in_multi_worker_mode",
                    event_info=(
                        "DynamoDB table creation is not "
                        "supported in multi-worker mode. "
                        "Table should already exist.",
                    ),
                )
                raise RasaException(
                    "DynamoDB table creation is not supported in "
                    "case of multiple sanic workers. To create the table either "
                    "run Rasa with a single worker or create the table manually."
                    "Here are the defaults which can be used to "
                    "create the table manually: "
                    f"Table name: {table_name}, Primary key: sender_id, "
                    f"key type `HASH`, attribute type `S` (String), "
                    "Provisioned throughput: Read capacity units: 5, "
                    "Write capacity units: 5"
                )

            table = self._dynamo.create_table(
                TableName=self.table_name,
                KeySchema=[{"AttributeName": "sender_id", "KeyType": "HASH"}],
                AttributeDefinitions=[
                    {"AttributeName": "sender_id", "AttributeType": "S"}
                ],
                ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            )

            # Wait until the table exists.
            table.meta.client.get_waiter("table_exists").wait(TableName=table_name)
        else:
            table = self._dynamo.Table(table_name)

        return table

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Saves the current conversation state.

        On the first save for a given ``sender_id`` (new item), the full tracker
        is written via ``put_item`` so that all events — including those before the
        latest ``UserUttered`` — are persisted.  A ``last_event_timestamp``
        attribute is included so that any duplicate first-save call is rejected by
        the idempotent ``update_item`` path below.

        On subsequent saves, only the new last-turn events (from the latest
        ``UserUttered`` timestamp onwards) are appended via a conditional
        ``update_item``.  The condition rejects duplicates atomically:
        the append is allowed only when the stored ``last_event_timestamp``
        precedes the first event being appended.

        If there are no last-turn events (empty turn), the full tracker is
        stored unconditionally via ``put_item`` — this handles the edge case of
        a tracker with no ``UserUttered`` events at all.
        """
        await self.stream_events(tracker)

        # Ensure conversation_started_timestamp is set (for backward compatibility)
        tracker.ensure_conversation_started_timestamp()

        serialized = self.serialise_tracker(tracker)
        new_events = tracker.get_last_turn_events()
        new_serialized_events = replace_floats_with_decimals(
            [event.as_dict() for event in new_events]
        )

        # No last-turn events — store the full tracker as-is
        if not new_serialized_events:
            self.db.put_item(Item=serialized)
            return None

        first_new_timestamp = new_serialized_events[0]["timestamp"]
        last_new_timestamp = new_serialized_events[-1]["timestamp"]

        # For a brand-new item, write the complete tracker (all events) so that
        # events before the first UserUttered are persisted.  We include
        # last_event_timestamp to enable idempotency on duplicate first-save calls.
        full_item = {**serialized, "last_event_timestamp": last_new_timestamp}
        try:
            self.db.put_item(
                Item=full_item,
                ConditionExpression="attribute_not_exists(sender_id)",
            )
            return None
        except ClientError as error:
            if error.response["Error"]["Code"] == "ConditionalCheckFailedException":
                structlogger.debug(
                    "rasa.core.tracker_stores.dynamo_tracker_store.save.item_already_exists",
                    sender_id=tracker.sender_id,
                    event_info=(
                        "Tracker item already exists: another process has saved "
                        "this tracker since it was last read. "
                        "Attempting to append only new last-turn events."
                    ),
                )
            else:
                raise error

        # Item exists: append only new last-turn events idempotently.
        # Guard against duplicate appends caused by retries or concurrent saves.
        # DynamoDB evaluates the condition atomically, so this is race-condition-safe
        # without a read-before-write.  The condition allows the write only when:
        #  - the item has no last_event_timestamp yet (legacy item without it), OR
        #  - the stored last_event_timestamp precedes the first new event
        #    (i.e. these events belong to a later turn than what is stored).
        # A duplicate save for the same turn is rejected because last_event_timestamp
        # would already be >= first_new_timestamp.
        update_expression = (
            "SET events = list_append(if_not_exists(events, :empty_list), :events)"
            ", last_event_timestamp = :last_event_timestamp"
        )
        expression_attribute_values: Dict[str, Any] = {
            ":events": new_serialized_events,
            ":empty_list": [],
            ":last_event_timestamp": last_new_timestamp,
            ":first_new_timestamp": first_new_timestamp,
        }

        # If user_id exists on tracker, ensure it's set in DynamoDB for GSI queries
        # Using if_not_exists to only write user_id if it's missing. This optimizes
        # performance for long-running trackers with frequent updates where user_id
        # doesn't change. This assumes old conversations are migrated offline to set
        # user_id, so missing user_id should not occur after migration.
        if tracker.user_id is not None:
            update_expression += ", user_id = if_not_exists(user_id, :user_id)"
            expression_attribute_values[":user_id"] = tracker.user_id

        # Store conversation_started_timestamp for efficient sorting
        # Using if_not_exists to only write if it's missing (for backward compatibility)
        if tracker.conversation_started_timestamp is not None:
            update_expression += (
                ", conversation_started_timestamp = "
                "if_not_exists(conversation_started_timestamp, "
                ":conversation_started_timestamp)"
            )
            # DynamoDB requires Decimal types, not float
            expression_attribute_values[":conversation_started_timestamp"] = Decimal(
                str(tracker.conversation_started_timestamp)
            )

        condition_expression = (
            "attribute_not_exists(last_event_timestamp) OR "
            "last_event_timestamp < :first_new_timestamp"
        )

        try:
            self.db.update_item(
                Key={"sender_id": tracker.sender_id},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attribute_values,
                ConditionExpression=condition_expression,
                ReturnValues="UPDATED_NEW",
            )
        except ClientError as error:
            if error.response["Error"]["Code"] == "ConditionalCheckFailedException":
                structlogger.warning(
                    "rasa.core.tracker_stores.dynamo_tracker_store.save.duplicate_events_skipped",
                    sender_id=tracker.sender_id,
                    first_new_timestamp=float(first_new_timestamp),
                    event_info=(
                        "Skipped duplicate event append: these turn events are "
                        "already stored (duplicate save call detected)."
                    ),
                )
            else:
                raise error
        return None

    async def delete(self, sender_id: Text) -> None:
        """Delete tracker for the given sender_id."""
        if not await self.exists(sender_id):
            structlogger.info(
                "dynamo_tracker_store.delete.no_tracker_for_sender_id",
                event_info=f"Could not find tracker for conversation ID '{sender_id}'.",
            )
            return None

        self.db.delete_item(
            Key={"sender_id": sender_id},
            ConditionExpression="attribute_exists(sender_id)",
        )
        structlogger.info(
            "dynamo_tracker_store.delete.deleted_tracker",
            sender_id=sender_id,
        )

    @staticmethod
    def serialise_tracker(
        tracker: "DialogueStateTracker",
    ) -> Dict:
        """Serializes the tracker, returns object with decimal types.

        DynamoDB cannot store `float`s, so we'll convert them to `Decimal`s.
        """
        return replace_floats_with_decimals(
            SerializedTrackerAsDict.serialise_tracker(tracker)
        )

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Retrieve dialogues for a sender_id in reverse-chronological order.

        Based on the session_date sort key.
        """
        return await self._retrieve(sender_id, fetch_all_sessions=False)

    async def retrieve_full_tracker(
        self, sender_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for all conversation sessions.

        Args:
            sender_id: Conversation ID to fetch the tracker for.
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
        dialogues = self.db.query(
            KeyConditionExpression=Key("sender_id").eq(sender_id),
            ScanIndexForward=False,
        )["Items"]

        if not dialogues:
            return None

        events_with_floats = []
        # Extract user_id from the first dialogue that has it
        user_id = dialogues[0].get(USER_ID) if dialogues else None

        for dialogue in dialogues:
            if dialogue.get("events"):
                events = rasa.utils.json_utils.replace_decimals_with_floats(
                    dialogue["events"]
                )
                events_with_floats.extend(events)

        events_with_floats = _deduplicate_events(events_with_floats)

        if self.domain is None:
            slots = []
        else:
            slots = self.domain.slots

        tracker = DialogueStateTracker.from_dict(
            sender_id, events_with_floats, slots, user_id=user_id
        )

        if fetch_all_sessions:
            return tracker

        return rasa.shared.core.trackers.get_latest_replay_safe_session_tracker(
            tracker,
            start_session_after_expiry=(
                self.domain.session_config.start_session_after_expiry
            ),
        )

    async def keys(self) -> Iterable[Text]:
        """Returns sender_ids of the `DynamoTrackerStore`."""
        response = self.db.scan(ProjectionExpression="sender_id")
        sender_ids = [i["sender_id"] for i in response["Items"]]

        while response.get("LastEvaluatedKey"):
            response = self.db.scan(
                ProjectionExpression="sender_id",
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            sender_ids.extend([i["sender_id"] for i in response["Items"]])

        return sender_ids

    async def update(
        self, tracker: DialogueStateTracker, apply_deletion_only: bool = True
    ) -> None:
        """Overwrites the tracker for the given sender_id."""
        # Ensure conversation_started_timestamp is set (for backward compatibility)
        tracker.ensure_conversation_started_timestamp()

        serialized = self.serialise_tracker(tracker)
        self.db.put_item(Item=serialized)

        structlogger.info(
            "dynamo_tracker_store.replace.replaced_tracker",
            sender_id=tracker.sender_id,
        )

    def _items_to_serialized(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert DynamoDB items to serialized tracker dicts without event replay.

        Applies Decimal-to-float conversion on the events list and deduplication,
        then assembles the target events-centric dict format. No
        ``DialogueStateTracker.from_dict()`` call is made.

        Args:
            items: DynamoDB item dicts as returned by a query or scan.

        Returns:
            List of serialized tracker dicts ready to be returned by the API.
        """
        result = []
        for item in items:
            sender_id = item.get("sender_id")
            if not sender_id:
                continue
            raw_events = item.get("events") or []
            events = _deduplicate_events(
                rasa.utils.json_utils.replace_decimals_with_floats(raw_events)
            )
            raw_ts = item.get("conversation_started_timestamp")
            result.append(
                {
                    "sender_id": sender_id,
                    "events": events,
                    USER_ID: item.get(USER_ID),
                    "conversation_started_timestamp": float(raw_ts)
                    if raw_ts is not None
                    else None,
                    "current_session_id": self._current_session_id_from_events(events),
                }
            )
        return result

    async def get_serialized_trackers_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves serialized trackers for a given user_id without event replay.

        Uses the same GSI path as :meth:`get_trackers_by_user_id` but returns
        raw event-centric dicts instead of reconstructed
        :class:`~rasa.shared.core.trackers.DialogueStateTracker` objects.
        Falls back to a full-table scan when the GSI is unavailable.

        Args:
            user_id: User ID to fetch trackers for.
            limit: Optional maximum number of trackers to return.
            skip: Optional number of trackers to skip before returning results.

        Returns:
            List of serialized tracker dicts sorted by
            ``(conversation_started_timestamp, sender_id)``.
        """
        try:
            gsi_name = "user_id-index"
            query_kwargs: Dict[str, Any] = {
                "IndexName": gsi_name,
                "KeyConditionExpression": Key(USER_ID).eq(user_id),
                "ScanIndexForward": True,
            }

            response = self.db.query(**query_kwargs)
            serialized: List[Dict[str, Any]] = self._items_to_serialized(
                response.get("Items", [])
            )

            while "LastEvaluatedKey" in response:
                query_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
                response = self.db.query(**query_kwargs)
                serialized.extend(self._items_to_serialized(response.get("Items", [])))

            serialized.sort(key=self._sort_key_serialized)
            return self._apply_pagination_serialized(serialized, skip, limit)

        except self.client.exceptions.ResourceNotFoundException:
            structlogger.debug(
                "dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
                event_info=(
                    "GSI 'user_id-index' not found. "
                    "Falling back to scanning all trackers."
                ),
            )
        except Exception as exc:
            structlogger.debug(
                "dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_query_failed",
                event_info=f"Failed to query GSI: {exc}. Falling back to scan.",
            )

        return await self._fallback_get_serialized_trackers_by_user_id(
            user_id, limit, skip
        )

    async def _fallback_get_serialized_trackers_by_user_id(
        self,
        user_id: str,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Scan-based fallback for :meth:`get_serialized_trackers_by_user_id`.

        Used when the GSI is unavailable. Scans all items and filters by
        ``user_id`` in Python.

        Args:
            user_id: User ID to filter by.
            limit: Optional maximum number of trackers to return.
            skip: Optional number of trackers to skip.

        Returns:
            List of serialized tracker dicts for the given user_id.
        """
        scan_kwargs: Dict[str, Any] = {
            "FilterExpression": Attr(USER_ID).eq(user_id),
        }
        response = self.db.scan(**scan_kwargs)
        all_items = response.get("Items", [])
        while "LastEvaluatedKey" in response:
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = self.db.scan(**scan_kwargs)
            all_items.extend(response.get("Items", []))

        serialized = self._items_to_serialized(all_items)
        serialized.sort(key=self._sort_key_serialized)
        return self._apply_pagination_serialized(serialized, skip, limit)
