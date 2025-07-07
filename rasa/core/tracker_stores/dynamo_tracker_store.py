from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Text

import structlog
from boto3.dynamodb.conditions import Key

import rasa.utils
from rasa.constants import DEFAULT_SANIC_WORKERS, ENV_SANIC_WORKERS
from rasa.core.tracker_stores.tracker_store import (
    SerializedTrackerAsDict,
    TrackerStore,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

structlogger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    import boto3.resources.factory.dynamodb.Table


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
        self.region = region
        self.table_name = table_name
        self.db = self.get_or_create_table(table_name)
        super().__init__(domain, event_broker, **kwargs)

    def get_or_create_table(
        self, table_name: Text
    ) -> "boto3.resources.factory.dynamodb.Table":
        """Returns table or creates one if the table name is not in the table list."""
        import boto3

        dynamo = boto3.resource("dynamodb", region_name=self.region)
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

            table = dynamo.create_table(
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
            table = dynamo.Table(table_name)

        return table

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Saves the current conversation state."""
        await self.stream_events(tracker)
        serialized = self.serialise_tracker(tracker)

        full_tracker = await self.retrieve_full_tracker(tracker.sender_id)
        if full_tracker is None:
            self.db.put_item(Item=serialized)
            return None

        # return the latest events since the last user message
        new_tracker = DialogueStateTracker.from_dict(
            serialized["sender_id"], events_as_dict=serialized["events"]
        )
        new_events = new_tracker.get_last_turn_events()
        new_serialized_events = [event.as_dict() for event in new_events]

        # we need to save the full tracker if it is a new tracker
        # without events following a user message
        if not new_serialized_events:
            self.db.put_item(Item=serialized)
            return None

        # append new events to the existing tracker
        self.db.update_item(
            Key={"sender_id": tracker.sender_id},
            UpdateExpression="SET events = list_append(if_not_exists(events, :empty_list), :events)",  # noqa: E501
            ExpressionAttributeValues={
                ":events": new_serialized_events,
                ":empty_list": [],
            },
            ReturnValues="UPDATED_NEW",
        )
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
        return rasa.utils.json_utils.replace_floats_with_decimals(
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
        for dialogue in dialogues:
            if dialogue.get("events"):
                events = rasa.utils.json_utils.replace_decimals_with_floats(
                    dialogue["events"]
                )
                events_with_floats.extend(events)

        if self.domain is None:
            slots = []
        else:
            slots = self.domain.slots

        tracker = DialogueStateTracker.from_dict(sender_id, events_with_floats, slots)

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

    async def update(self, tracker: DialogueStateTracker) -> None:
        """Overwrites the tracker for the given sender_id."""
        serialized = self.serialise_tracker(tracker)
        self.db.put_item(Item=serialized)

        structlogger.info(
            "dynamo_tracker_store.replace.replaced_tracker",
            sender_id=tracker.sender_id,
        )
