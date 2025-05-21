import os
import uuid
from typing import Any, List

import pytest

from rasa.core.tracker_stores.dynamo_tracker_store import DynamoTrackerStore
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.fixture(scope="session")
def setup_env_vars() -> None:
    os.environ["AWS_ENDPOINT_URL"] = "http://0.0.0.0:8000"
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"


@pytest.fixture(scope="session")
def table_name() -> str:
    return f"rasa-{uuid.uuid4().hex}"


@pytest.fixture(scope="session")
async def dynamo_tracker_store(
    setup_env_vars: Any, table_name: str
) -> DynamoTrackerStore:
    return DynamoTrackerStore(
        domain=Domain.empty(),
        table_name=table_name,
        region="us-east-1",
    )


def test_dynamo_tracker_store_login(
    dynamo_tracker_store: DynamoTrackerStore, table_name: str
) -> None:
    """Verify that the dynamo client can connect to the dynamo server."""
    assert dynamo_tracker_store.client is not None
    response = dynamo_tracker_store.client.list_tables()
    assert table_name in response["TableNames"]


async def test_dynamo_tracker_store_retrieve_full_tracker(
    dynamo_tracker_store: DynamoTrackerStore,
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    """Verify that the DynamoTrackerStore can retrieve a full tracker."""
    if not await dynamo_tracker_store.exists(tracker_with_restarted_event.sender_id):
        await dynamo_tracker_store.save(tracker_with_restarted_event)

    sender_id = tracker_with_restarted_event.sender_id

    tracker = await dynamo_tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_dynamo_tracker_store_retrieve(
    dynamo_tracker_store: DynamoTrackerStore,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    if not await dynamo_tracker_store.exists(tracker_with_restarted_event.sender_id):
        await dynamo_tracker_store.save(tracker_with_restarted_event)
    sender_id = tracker_with_restarted_event.sender_id
    tracker = await dynamo_tracker_store.retrieve(sender_id)

    # the retrieved tracker with the latest session would not contain
    # `action_session_start` event because the DynamoTrackerStore filters
    # only the events after `session_started` event
    assert list(tracker.events) == events_after_restart


async def test_dynamo_tracker_store_delete(
    dynamo_tracker_store: DynamoTrackerStore,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    if not await dynamo_tracker_store.exists(tracker_with_restarted_event.sender_id):
        await dynamo_tracker_store.save(tracker_with_restarted_event)

    sender_id = tracker_with_restarted_event.sender_id
    await dynamo_tracker_store.delete(sender_id)

    tracker = await dynamo_tracker_store.retrieve(sender_id)
    assert tracker is None
