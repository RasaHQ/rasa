import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import pytest
from dotenv import load_dotenv

from rasa.core.tracker_stores.mongo_tracker_store import MongoTrackerStore
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    ConversationInactive,
    DialogueStackUpdated,
    SessionStarted,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from tests.integration_tests.core.conftest import (
    assert_all_trackers_have_properties,
    assert_tracker_properties,
    create_multiple_trackers_with_user_id,
    create_tracker_with_user_id,
)

if TYPE_CHECKING:
    from rasa.shared.core.trackers import Event


MONGODB_PARENT_PATH_NAME = (
    "tests_deployment/integration_tests_tracker_stores/mongo_db_tracker_store"
)


@pytest.fixture
def mongodb_credentials() -> Tuple[str, str, str]:
    """Get the MongoDB credentials from the .env file.

    For local testing, you should create a .env file in the path
    tests_deployment/integration_tests_tracker_stores/mongo_db_tracker_store.
    """
    load_dotenv(Path(f"{MONGODB_PARENT_PATH_NAME}/.env"))

    db_name = os.getenv("DB_NAME")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    return db_name, username, password


def get_mongodb_tls_host_uri() -> str:
    """Prepare the MongoDB URI with TLS for the tests.

    You must generate the TLS certificates and keys for the MongoDB server
    you will run as a Docker container in the expected paths.
    """
    tls_ca_file = f"{MONGODB_PARENT_PATH_NAME}/tls/ca.crt"
    tls_certificate_key_file = f"{MONGODB_PARENT_PATH_NAME}/tls/mongodb.pem"

    return f"mongodb://localhost:27017/?tls=true&tlsCAFile={tls_ca_file}&tlsCertificateKeyFile={tls_certificate_key_file}"


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
def test_mongo_tracker_store_login(
    domain: "Domain",
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    """Verify that the mongo client can connect to the mongodb server."""
    db_name, username, password = mongodb_credentials

    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )
    assert mongo_tracker_store.client is not None

    assert db_name in mongo_tracker_store.client.list_database_names()


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_tracker_store_retrieve_full_tracker(
    domain: "Domain",
    tracker_with_restarted_event: "DialogueStateTracker",
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    """Verify that the MongoTrackerStore can retrieve a full tracker."""
    db_name, username, password = mongodb_credentials

    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )
    sender_id = tracker_with_restarted_event.sender_id

    await mongo_tracker_store.save(tracker_with_restarted_event)

    tracker = await mongo_tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_tracker_store_retrieve(
    domain: "Domain",
    tracker_with_restarted_event: "DialogueStateTracker",
    events_after_restart: List["Event"],
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    sender_id = tracker_with_restarted_event.sender_id
    db_name, username, password = mongodb_credentials

    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )

    await mongo_tracker_store.save(tracker_with_restarted_event)

    tracker = await mongo_tracker_store.retrieve(sender_id)

    # Latest session begins at the last ``action_session_start`` (inclusive).
    assert list(tracker.events) == events_after_restart


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_tracker_store_retrieve_widens_prefix_for_stack_integrity_true_mode(
    domain: "Domain",
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    """Replay-safe widening across action_session_start (start_session_after_expiry True)."""
    db_name, username, password = mongodb_credentials

    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )
    sender_id = f"mongo_it_stack_{uuid.uuid4().hex}"
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/0", "value": {"frame_id": "old", "flow_id": "foo", "step_id": "OLD", "frame_type": "regular", "type": "flow"}}]'
            ),
            ActionExecuted(ACTION_SESSION_START_NAME),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/0/step_id", "value": "SECOND"}]'
            ),
        ],
    )
    await mongo_tracker_store.save(tracker)

    retrieved = await mongo_tracker_store.retrieve(sender_id)

    assert retrieved is not None
    assert len(retrieved.events) == 4
    assert retrieved.stack.frames[0].frame_id == "old"
    assert retrieved.stack.frames[0].step_id == "SECOND"


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_tracker_store_retrieve_widens_prefix_for_stack_integrity_false_mode(
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    """Replay-safe boundary after ConversationInactive when sessions do not auto-start."""
    domain = Domain.from_dict({"session_config": {"start_session_after_expiry": False}})
    db_name, username, password = mongodb_credentials

    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )
    sender_id = f"mongo_it_widen_false_{uuid.uuid4().hex}"
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
    await mongo_tracker_store.save(tracker)

    retrieved = await mongo_tracker_store.retrieve(sender_id)

    assert retrieved is not None
    assert len(retrieved.events) == 3
    assert retrieved.stack.frames[0].step_id == "AFTER_INACTIVE"


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_tracker_store_delete(
    domain: "Domain",
    tracker_with_restarted_event: "DialogueStateTracker",
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    """Verify that the MongoTrackerStore can delete a tracker."""
    # Given
    db_name, username, password = mongodb_credentials

    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )
    sender_id = tracker_with_restarted_event.sender_id
    await mongo_tracker_store.save(tracker_with_restarted_event)

    # When
    await mongo_tracker_store.delete(sender_id)

    # Then
    tracker = await mongo_tracker_store.retrieve(sender_id)
    assert tracker is None


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_tracker_store_update(
    domain: "Domain",
    tracker_with_restarted_event: DialogueStateTracker,
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
    events_after_restart: List["Event"],
) -> None:
    """Verify that the MongoTrackerStore can update a tracker."""
    # Given
    db_name, username, password = mongodb_credentials

    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )
    await mongo_tracker_store.save(tracker_with_restarted_event)
    new_tracker = DialogueStateTracker.from_events(
        sender_id=tracker_with_restarted_event.sender_id,
        evts=events_after_restart,
        slots=domain.slots,
        domain=domain,
    )

    # When
    await mongo_tracker_store.update(new_tracker)

    # Then
    updated_tracker = await mongo_tracker_store.retrieve_full_tracker(
        tracker_with_restarted_event.sender_id
    )
    assert updated_tracker.current_state(
        EventVerbosity.ALL
    ) == new_tracker.current_state(EventVerbosity.ALL)


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_get_trackers_by_user_id(
    domain: Domain,
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    db_name, username, password = mongodb_credentials
    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )

    user_id = uuid.uuid4().hex
    sender_id_1 = uuid.uuid4().hex
    sender_id_2 = uuid.uuid4().hex

    # Create two trackers with the same user_id
    tracker1 = await create_tracker_with_user_id(
        mongo_tracker_store,
        sender_id_1,
        user_id,
        [SessionStarted(), UserUttered("Hello")],
    )
    timestamp1 = tracker1.conversation_started_timestamp

    tracker2 = await create_tracker_with_user_id(
        mongo_tracker_store, sender_id_2, user_id, [SessionStarted(), UserUttered("Hi")]
    )
    timestamp2 = tracker2.conversation_started_timestamp

    # Retrieve trackers by user_id
    trackers = await mongo_tracker_store.get_trackers_by_user_id(user_id)

    # Verify results - check both user_id and conversation_started_timestamp
    assert len(trackers) == 2
    sender_ids = {tracker["sender_id"] for tracker in trackers}
    assert sender_id_1 in sender_ids
    assert sender_id_2 in sender_ids
    assert_all_trackers_have_properties(trackers, user_id)

    # Verify timestamps match originals
    tracker_dict = {t["sender_id"]: t for t in trackers}
    assert tracker_dict[sender_id_1]["conversation_started_timestamp"] == timestamp1
    assert tracker_dict[sender_id_2]["conversation_started_timestamp"] == timestamp2


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_get_trackers_by_user_id_with_limit(
    domain: Domain,
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    db_name, username, password = mongodb_credentials
    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )

    user_id = uuid.uuid4().hex

    # Create 10 trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        mongo_tracker_store, user_id, 10
    )

    # Retrieve with limit
    trackers = await mongo_tracker_store.get_trackers_by_user_id(user_id, limit=5)

    assert len(trackers) == 5
    assert_all_trackers_have_properties(trackers, user_id)

    assert [t["sender_id"] for t in trackers] == [
        t.sender_id for t in saved_trackers[:5]
    ]


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_get_trackers_by_user_id_with_skip(
    domain: Domain,
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    db_name, username, password = mongodb_credentials
    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )

    user_id = uuid.uuid4().hex

    # Create 10 trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        mongo_tracker_store, user_id, 10
    )

    # Retrieve with skip
    trackers = await mongo_tracker_store.get_trackers_by_user_id(user_id, skip=3)

    # Verify results - check both properties
    assert len(trackers) == 7  # 10 total - 3 skipped
    assert_all_trackers_have_properties(trackers, user_id)
    assert [t["sender_id"] for t in trackers] == [
        t.sender_id for t in saved_trackers[3:]
    ]


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_get_trackers_by_user_id_with_skip_and_limit(
    domain: Domain,
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    db_name, username, password = mongodb_credentials
    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )

    user_id = uuid.uuid4().hex

    # Create 10 trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        mongo_tracker_store, user_id, 10
    )

    # Retrieve with skip and limit
    trackers = await mongo_tracker_store.get_trackers_by_user_id(
        user_id, skip=2, limit=3
    )

    # Verify results - check both properties
    assert len(trackers) == 3
    assert_all_trackers_have_properties(trackers, user_id)
    assert [t["sender_id"] for t in trackers] == [
        t.sender_id for t in saved_trackers[2:5]
    ]


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_get_trackers_by_user_id_no_matches(
    domain: Domain,
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    """Test get_trackers_by_user_id returns empty list when no matches."""
    db_name, username, password = mongodb_credentials
    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )

    user_id = "integration_test_user_no_matches"

    # Query for user_id that doesn't exist
    trackers = await mongo_tracker_store.get_trackers_by_user_id(user_id)

    # Verify empty result
    assert len(trackers) == 0


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_conversation_started_timestamp_backward_compatibility(
    domain: Domain,
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    db_name, username, password = mongodb_credentials
    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )

    sender_id = uuid.uuid4().hex
    user_id = uuid.uuid4().hex

    # Create tracker and manually clear timestamp (simulating old tracker)
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [SessionStarted(), UserUttered("Hello")],
        slots=domain.slots,
        domain=domain,
        user_id=user_id,
    )
    tracker.conversation_started_timestamp = None
    expected_timestamp = tracker.events[0].timestamp

    # Save should populate the timestamp
    await mongo_tracker_store.save(tracker)

    # Retrieve and verify both properties (retrieve() returns DialogueStateTracker)
    retrieved = await mongo_tracker_store.retrieve(sender_id)
    assert retrieved is not None
    assert retrieved.user_id == user_id
    assert retrieved.sender_id == sender_id
    assert retrieved.conversation_started_timestamp == expected_timestamp

    # Also verify via get_trackers_by_user_id (returns serialized dicts)
    trackers = await mongo_tracker_store.get_trackers_by_user_id(user_id)
    assert len(trackers) == 1
    assert_tracker_properties(trackers[0], user_id, sender_id, expected_timestamp)


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_get_trackers_by_user_id_sorted_by_timestamp(
    domain: Domain,
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    db_name, username, password = mongodb_credentials
    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )

    user_id = uuid.uuid4().hex

    # Create trackers with different timestamps
    base_timestamp = 1234567890.0
    for i in range(5):
        sender_id = uuid.uuid4().hex
        # Create events with explicit timestamps
        events = [
            SessionStarted(timestamp=base_timestamp + i),
            UserUttered("Hello", timestamp=base_timestamp + i + 1),
        ]
        await create_tracker_with_user_id(
            mongo_tracker_store, sender_id, user_id, events
        )

    # Retrieve trackers
    trackers = await mongo_tracker_store.get_trackers_by_user_id(user_id)

    # Verify both properties and sorting
    assert len(trackers) == 5
    assert_all_trackers_have_properties(trackers, user_id)

    # Verify sorting (should be sorted by conversation_started_timestamp)
    timestamps = [
        t["conversation_started_timestamp"]
        for t in trackers
        if t["conversation_started_timestamp"]
    ]
    # Verify timestamps are in ascending order
    assert timestamps == sorted(timestamps)


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_get_trackers_by_user_id_filters_by_user_id(
    domain: Domain,
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    db_name, username, password = mongodb_credentials
    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )

    user_id_1 = uuid.uuid4().hex
    user_id_2 = uuid.uuid4().hex
    sender_id_1 = uuid.uuid4().hex
    sender_id_2 = uuid.uuid4().hex

    # Create trackers with different user_ids
    tracker1 = await create_tracker_with_user_id(
        mongo_tracker_store, sender_id_1, user_id_1
    )
    timestamp1 = tracker1.conversation_started_timestamp

    tracker2 = await create_tracker_with_user_id(
        mongo_tracker_store, sender_id_2, user_id_2
    )
    timestamp2 = tracker2.conversation_started_timestamp

    # Query for user_id_1
    trackers = await mongo_tracker_store.get_trackers_by_user_id(user_id_1)

    # Verify only tracker1 is returned with both properties
    assert len(trackers) == 1
    assert_tracker_properties(trackers[0], user_id_1, sender_id_1, timestamp1)

    # Verify user_id_2 returns different tracker
    trackers2 = await mongo_tracker_store.get_trackers_by_user_id(user_id_2)
    assert len(trackers2) == 1
    assert_tracker_properties(trackers2[0], user_id_2, sender_id_2, timestamp2)


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
@pytest.mark.parametrize("num_conversations", [100, 500, 1000, 2000])
async def test_mongo_get_trackers_by_user_id_performance(
    num_conversations: int,
    domain: Domain,
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    """Performance test for querying trackers by user_id with large datasets."""
    user_id = uuid.uuid4().hex
    db_name, username, password = mongodb_credentials
    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )
    # Create many trackers for the same user
    await create_multiple_trackers_with_user_id(
        mongo_tracker_store, user_id, num_conversations, delay=0.0
    )

    # Test retrieval without pagination
    retrieval_start = time.time()
    trackers = await mongo_tracker_store.get_trackers_by_user_id(user_id)
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
    page_trackers = await mongo_tracker_store.get_trackers_by_user_id(
        user_id, limit=page_size
    )
    paginated_time = time.time() - paginated_start
    assert len(page_trackers) == page_size

    # Paginated queries should be faster
    assert (
        paginated_time < 5.0
    ), f"Paginated retrieval took {paginated_time:.2f}s, expected < 5.0s"
