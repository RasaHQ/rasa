import os
import time
import uuid
from typing import Any, List

import pytest
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.core.tracker_stores.dynamo_tracker_store import DynamoTrackerStore
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    ConversationInactive,
    DialogueStackUpdated,
    Event,
    SessionStarted,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker
from tests.integration_tests.core.conftest import (
    assert_all_trackers_have_properties,
    assert_tracker_properties,
    create_multiple_trackers_with_user_id,
    create_tracker_with_user_id,
)
from tests.utilities import filter_logs


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


def create_user_id_gsi(tracker_store: DynamoTrackerStore) -> None:
    """Create a GSI with user_id as partition key and conversation_started_timestamp
    as sort key.

    This simulates the manual GSI creation that would be done in production.
    Creates a GSI with:
    - Partition key: user_id (String)
    - Sort key: conversation_started_timestamp (Number)
    - Index name: user_id-index

    This enables efficient database-level sorting and pagination.
    """
    import boto3

    dynamodb_resource = boto3.resource("dynamodb", region_name="us-east-1")
    table = dynamodb_resource.Table(tracker_store.table_name)

    # Wait for table to exist
    table.meta.client.get_waiter("table_exists").wait(
        TableName=tracker_store.table_name
    )

    # Check if GSI already exists
    try:
        table.reload()
        existing_gsi = None
        for gsi in table.global_secondary_indexes or []:
            if gsi["IndexName"] == "user_id-index":
                existing_gsi = gsi
                break
        if existing_gsi and existing_gsi["IndexStatus"] == "ACTIVE":
            return  # GSI already exists and is active
    except Exception:
        pass

    # Create GSI with user_id as partition key and
    # conversation_started_timestamp as sort key
    try:
        table.update(
            AttributeDefinitions=[
                {"AttributeName": "user_id", "AttributeType": "S"},
                {
                    "AttributeName": "conversation_started_timestamp",
                    "AttributeType": "N",
                },
            ],
            GlobalSecondaryIndexUpdates=[
                {
                    "Create": {
                        "IndexName": "user_id-index",
                        "KeySchema": [
                            {"AttributeName": "user_id", "KeyType": "HASH"},
                            {
                                "AttributeName": "conversation_started_timestamp",
                                "KeyType": "RANGE",
                            },
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                        "ProvisionedThroughput": {
                            "ReadCapacityUnits": 5,
                            "WriteCapacityUnits": 5,
                        },
                    }
                }
            ],
        )
        # Wait for GSI to be active
        max_wait = 30
        wait_time = 0
        while wait_time < max_wait:
            table.reload()
            gsi_status = None
            for gsi in table.global_secondary_indexes or []:
                if gsi["IndexName"] == "user_id-index":
                    gsi_status = gsi["IndexStatus"]
                    break
            if gsi_status == "ACTIVE":
                break
            time.sleep(0.5)
            wait_time += 0.5
    except Exception:
        # GSI creation failed (e.g., already exists or other error)
        # This is okay for integration tests - fallback behavior will be tested
        pass


@pytest.fixture(scope="session")
async def dynamo_tracker_store_with_gsi(
    setup_env_vars: Any, table_name: str
) -> DynamoTrackerStore:
    tracker_store = DynamoTrackerStore(
        domain=Domain.empty(),
        table_name=table_name,
        region="us-east-1",
    )
    # Create GSI for efficient user_id queries with timestamp sorting
    create_user_id_gsi(tracker_store)
    return tracker_store


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


async def test_dynamo_tracker_store_retrieve_widens_prefix_for_stack_integrity_true_mode(
    dynamo_tracker_store: DynamoTrackerStore,
) -> None:
    """Replay-safe widening across action_session_start boundaries (integration)."""
    sender_id = f"it_dynamo_widen_true_{uuid.uuid4().hex}"
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/0", "value": {"frame_id": "f1", "flow_id": "foo", "step_id": "START", "frame_type": "regular", "type": "flow"}}]'
            ),
            ActionExecuted(ACTION_SESSION_START_NAME),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/0/step_id", "value": "SECOND"}]'
            ),
        ],
    )
    await dynamo_tracker_store.save(tracker)

    retrieved = await dynamo_tracker_store.retrieve(sender_id)

    assert retrieved is not None
    assert len(retrieved.events) == 4
    assert retrieved.stack.frames[0].step_id == "SECOND"


async def test_dynamo_tracker_store_retrieve_widens_prefix_for_stack_integrity_false_mode(
    dynamo_tracker_store: DynamoTrackerStore,
    monkeypatch: MonkeyPatch,
) -> None:
    """Widening when latest boundary is after ConversationInactive (integration)."""
    false_domain = Domain.from_dict(
        {"session_config": {"start_session_after_expiry": False}}
    )
    monkeypatch.setattr(dynamo_tracker_store, "domain", false_domain)
    sender_id = f"it_dynamo_widen_false_{uuid.uuid4().hex}"
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/0", "value": {"frame_id": "f1", "flow_id": "foo", "step_id": "START", "frame_type": "regular", "type": "flow"}}]'
            ),
            ConversationInactive(),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/0/step_id", "value": "AFTER_INACTIVE"}]'
            ),
        ],
    )
    await dynamo_tracker_store.save(tracker)

    retrieved = await dynamo_tracker_store.retrieve(sender_id)

    assert retrieved is not None
    assert len(retrieved.events) == 3
    assert retrieved.stack.frames[0].step_id == "AFTER_INACTIVE"


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


async def test_dynamo_tracker_store_update(
    dynamo_tracker_store: DynamoTrackerStore,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    if not await dynamo_tracker_store.exists(tracker_with_restarted_event.sender_id):
        await dynamo_tracker_store.save(tracker_with_restarted_event)

    sender_id = tracker_with_restarted_event.sender_id
    new_tracker = DialogueStateTracker.from_events(
        sender_id=sender_id,
        evts=events_after_restart,
    )
    await dynamo_tracker_store.update(new_tracker)

    tracker = await dynamo_tracker_store.retrieve(sender_id)
    assert tracker == new_tracker


async def test_dynamo_get_trackers_by_user_id(
    dynamo_tracker_store_with_gsi: DynamoTrackerStore,
) -> None:
    """Test get_trackers_by_user_id functionality."""
    user_id = uuid.uuid4().hex
    sender_id_1 = uuid.uuid4().hex
    sender_id_2 = uuid.uuid4().hex

    # Create two trackers with the same user_id
    tracker1 = await create_tracker_with_user_id(
        dynamo_tracker_store_with_gsi,
        sender_id_1,
        user_id,
        [SessionStarted(), UserUttered("Hello")],
    )
    timestamp1 = tracker1.conversation_started_timestamp
    await dynamo_tracker_store_with_gsi.save(tracker1)

    tracker2 = await create_tracker_with_user_id(
        dynamo_tracker_store_with_gsi,
        sender_id_2,
        user_id,
        [SessionStarted(), UserUttered("Hi")],
    )
    timestamp2 = tracker2.conversation_started_timestamp
    await dynamo_tracker_store_with_gsi.save(tracker2)

    # Retrieve trackers by user_id
    trackers = await dynamo_tracker_store_with_gsi.get_trackers_by_user_id(user_id)

    # Verify results - check both user_id and conversation_started_timestamp
    assert len(trackers) == 2
    sender_ids = {tracker.sender_id for tracker in trackers}
    assert sender_id_1 in sender_ids
    assert sender_id_2 in sender_ids
    assert_all_trackers_have_properties(trackers, user_id)

    assert trackers[0].conversation_started_timestamp == timestamp1
    assert trackers[1].conversation_started_timestamp == timestamp2


async def test_dynamo_get_trackers_by_user_id_with_limit(
    dynamo_tracker_store_with_gsi: DynamoTrackerStore,
) -> None:
    """Test get_trackers_by_user_id with limit parameter."""
    user_id = uuid.uuid4().hex
    saved_trackers = await create_multiple_trackers_with_user_id(
        dynamo_tracker_store_with_gsi, user_id, 10
    )

    # Retrieve with limit
    trackers = await dynamo_tracker_store_with_gsi.get_trackers_by_user_id(
        user_id, limit=5
    )

    assert len(trackers) == 5
    assert_all_trackers_have_properties(trackers, user_id)
    assert trackers == saved_trackers[:5]


async def test_dynamo_get_trackers_by_user_id_with_skip(
    dynamo_tracker_store_with_gsi: DynamoTrackerStore,
) -> None:
    """Test get_trackers_by_user_id with skip parameter."""
    user_id = uuid.uuid4().hex
    saved_trackers = await create_multiple_trackers_with_user_id(
        dynamo_tracker_store_with_gsi, user_id, 10
    )

    # Retrieve with skip
    trackers = await dynamo_tracker_store_with_gsi.get_trackers_by_user_id(
        user_id, skip=3
    )

    assert len(trackers) == 7  # 10 total - 3 skipped
    assert_all_trackers_have_properties(trackers, user_id)
    assert trackers == saved_trackers[3:]


async def test_dynamo_get_trackers_by_user_id_with_skip_and_limit(
    dynamo_tracker_store_with_gsi: DynamoTrackerStore,
) -> None:
    """Test get_trackers_by_user_id with both skip and limit parameters,
    verifying both user_id and conversation_started_timestamp."""
    user_id = uuid.uuid4().hex
    saved_trackers = await create_multiple_trackers_with_user_id(
        dynamo_tracker_store_with_gsi, user_id, 10
    )

    # Retrieve with skip and limit
    trackers = await dynamo_tracker_store_with_gsi.get_trackers_by_user_id(
        user_id, skip=2, limit=3
    )

    assert len(trackers) == 3
    assert_all_trackers_have_properties(trackers, user_id)

    assert trackers == saved_trackers[2:5]


async def test_dynamo_get_trackers_by_user_id_no_matches(
    dynamo_tracker_store_with_gsi: DynamoTrackerStore,
) -> None:
    """Test get_trackers_by_user_id returns empty list when no matches."""
    user_id = uuid.uuid4().hex

    # Query for user_id that doesn't exist
    trackers = await dynamo_tracker_store_with_gsi.get_trackers_by_user_id(user_id)

    # Verify empty result
    assert len(trackers) == 0


async def test_dynamo_conversation_started_timestamp_backward_compatibility(
    dynamo_tracker_store_with_gsi: DynamoTrackerStore,
) -> None:
    """Test that old trackers without conversation_started_timestamp."""
    sender_id = uuid.uuid4().hex
    user_id = uuid.uuid4().hex

    # Create tracker and manually clear timestamp (simulating old tracker)
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [SessionStarted(), UserUttered("Hello")],
        slots=dynamo_tracker_store_with_gsi.domain.slots,
        domain=dynamo_tracker_store_with_gsi.domain,
        user_id=user_id,
    )
    tracker.conversation_started_timestamp = None
    expected_timestamp = tracker.events[0].timestamp

    # Save should populate the timestamp
    await dynamo_tracker_store_with_gsi.save(tracker)

    # Retrieve and verify both properties
    retrieved = await dynamo_tracker_store_with_gsi.retrieve(sender_id)
    assert retrieved is not None
    assert_tracker_properties(retrieved, user_id, sender_id, expected_timestamp)

    # Also verify via get_trackers_by_user_id
    trackers = await dynamo_tracker_store_with_gsi.get_trackers_by_user_id(user_id)
    assert len(trackers) == 1
    assert_tracker_properties(trackers[0], user_id, sender_id, expected_timestamp)


async def test_dynamo_get_trackers_by_user_id_sorted_by_timestamp(
    dynamo_tracker_store_with_gsi: DynamoTrackerStore,
) -> None:
    """Test that get_trackers_by_user_id returns trackers sorted by timestamp."""
    user_id = uuid.uuid4().hex

    # Create trackers with different timestamps
    base_timestamp = time.time()
    for i in range(5):
        sender_id = uuid.uuid4().hex
        # Create events with explicit timestamps
        events = [
            SessionStarted(timestamp=base_timestamp + i),
            UserUttered("Hello", timestamp=base_timestamp + i + 1),
        ]
        await create_tracker_with_user_id(
            dynamo_tracker_store_with_gsi, sender_id, user_id, events
        )

    # Retrieve trackers
    trackers = await dynamo_tracker_store_with_gsi.get_trackers_by_user_id(user_id)

    assert len(trackers) == 5
    assert_all_trackers_have_properties(trackers, user_id)

    # Verify sorting (should be sorted by conversation_started_timestamp)
    timestamps = [
        t.conversation_started_timestamp
        for t in trackers
        if t.conversation_started_timestamp
    ]
    # Verify timestamps are in ascending order
    assert timestamps == sorted(timestamps)


async def test_dynamo_get_trackers_by_user_id_filters_by_user_id(
    dynamo_tracker_store_with_gsi: DynamoTrackerStore,
) -> None:
    user_id_1 = "integration_test_user_filter_1"
    user_id_2 = "integration_test_user_filter_2"
    sender_id_1 = uuid.uuid4().hex
    sender_id_2 = uuid.uuid4().hex

    # Create tracker with user_id_1
    tracker1 = await create_tracker_with_user_id(
        dynamo_tracker_store_with_gsi,
        sender_id_1,
        user_id_1,
        [SessionStarted(), UserUttered("Hello")],
    )
    timestamp1 = tracker1.conversation_started_timestamp

    # Create tracker with user_id_2
    tracker2 = await create_tracker_with_user_id(
        dynamo_tracker_store_with_gsi,
        sender_id_2,
        user_id_2,
        [SessionStarted(), UserUttered("Hi")],
    )
    timestamp2 = tracker2.conversation_started_timestamp

    # Query for user_id_1
    with capture_logs() as caplog:
        trackers = await dynamo_tracker_store_with_gsi.get_trackers_by_user_id(
            user_id_1
        )

        # Verify only tracker1 is returned with both properties
        assert len(trackers) == 1
        assert_tracker_properties(trackers[0], user_id_1, sender_id_1, timestamp1)

        # Verify user_id_2 returns different tracker
        trackers2 = await dynamo_tracker_store_with_gsi.get_trackers_by_user_id(
            user_id_2
        )
        assert len(trackers2) == 1
        assert_tracker_properties(trackers2[0], user_id_2, sender_id_2, timestamp2)

        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 0


@pytest.mark.parametrize("num_conversations", [100, 500, 1000, 2000])
async def test_dynamo_get_trackers_by_user_id_performance(
    num_conversations: int,
    dynamo_tracker_store_with_gsi: DynamoTrackerStore,
) -> None:
    # Create many trackers for the same user
    user_id = uuid.uuid4().hex
    await create_multiple_trackers_with_user_id(
        dynamo_tracker_store_with_gsi, user_id, num_conversations, delay=0.0
    )

    # Test retrieval without pagination
    retrieval_start = time.time()
    trackers = await dynamo_tracker_store_with_gsi.get_trackers_by_user_id(user_id)
    retrieval_time = time.time() - retrieval_start

    assert len(trackers) == num_conversations
    # Performance assertion: should retrieve within reasonable time
    # For 1000 conversations, should be < 10 seconds for in-memory and SQL
    max_time = 10.0 if num_conversations <= 1000 else 20.0
    assert (
        retrieval_time < max_time
    ), f"Retrieval took {retrieval_time:.2f}s, expected < {max_time}s"

    # Test retrieval with pagination
    page_size = 100
    paginated_start = time.time()
    page_trackers = await dynamo_tracker_store_with_gsi.get_trackers_by_user_id(
        user_id, limit=page_size
    )
    paginated_time = time.time() - paginated_start
    assert len(page_trackers) == page_size

    # Paginated queries should be faster
    assert (
        paginated_time < 5.0
    ), f"Paginated retrieval took {paginated_time:.2f}s, expected < 5.0s"
