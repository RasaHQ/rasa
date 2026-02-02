import time
import uuid
from typing import Generator, List
from unittest.mock import Mock

import pytest
import sqlalchemy as sa
import structlog
from pytest import MonkeyPatch

from rasa.constants import ENV_SANIC_WORKERS
from rasa.core.tracker_stores.redis_tracker_store import RedisTrackerStore
from rasa.core.tracker_stores.sql_tracker_store import SQLTrackerStore
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, SessionStarted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from tests.integration_tests.core.conftest import (
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
    assert_all_trackers_have_properties,
    assert_tracker_properties,
    create_multiple_trackers_with_user_id,
    create_tracker_with_user_id,
)
from tests.utilities import filter_logs

# NOTE about the timeouts in this file. We want to fail fast
# because SQLTrackerStore tries to connect several times
# until it works. If the timeout is hit, it probably means
# that something is wrong in the setup of the test


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
def test_sql_tracker_store_with_login_db(
    postgres_login_db_connection: sa.engine.Connection,
    postgres_db_name: str,
    postgres_login_db_name: str,
):
    tracker_store = SQLTrackerStore(
        dialect="postgresql",
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        db=postgres_db_name,
        login_db=postgres_login_db_name,
    )

    matching_rows = postgres_login_db_connection.execute(
        sa.text(
            f"SELECT 1 FROM pg_catalog.pg_database "
            f"WHERE datname = '{postgres_db_name}'"
        )
    ).rowcount
    assert matching_rows == 1
    assert tracker_store.engine.url.database == postgres_db_name
    tracker_store.engine.dispose()


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
def test_sql_tracker_store_with_login_db_db_already_exists(
    postgres_login_db_connection: sa.engine.Connection,
    postgres_db_name: str,
    postgres_login_db_name: str,
):
    postgres_login_db_connection.execute(sa.text(f"CREATE DATABASE {postgres_db_name}"))

    tracker_store = SQLTrackerStore(
        dialect="postgresql",
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        db=postgres_db_name,
        login_db=postgres_login_db_name,
    )

    matching_rows = postgres_login_db_connection.execute(
        sa.text(
            f"SELECT 1 FROM pg_catalog.pg_database "
            f"WHERE datname = '{postgres_db_name}'"
        )
    ).rowcount

    assert matching_rows == 1
    tracker_store.engine.dispose()


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
def test_sql_tracker_store_with_login_db_race_condition(
    postgres_login_db_connection: sa.engine.Connection,
    monkeypatch: MonkeyPatch,
    postgres_login_db_name: str,
    postgres_db_name: str,
):
    original_execute = sa.engine.Connection.execute

    def mock_execute(self, *args, **kwargs):
        # this simulates a race condition
        from sqlalchemy import Executable

        if isinstance(args[0], Executable):
            if (
                f"SELECT 1 FROM pg_catalog.pg_database "
                f"WHERE datname = '{postgres_db_name}'" in str(args[0])
            ):
                original_execute(
                    self.execution_options(isolation_level="AUTOCOMMIT"),
                    sa.text(f"CREATE DATABASE {postgres_db_name}"),
                )
                return Mock(rowcount=0)
            else:
                return original_execute(self, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(sa.engine.Connection, "execute", mock_execute)
        with structlog.testing.capture_logs() as caplog:
            tracker_store = SQLTrackerStore(
                dialect="postgresql",
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                username=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                db=postgres_db_name,
                login_db=postgres_login_db_name,
            )
            print(caplog)
            # IntegrityError has been caught and we log the error
            logs = filter_logs(
                caplog,
                event="sql_tracker_store.create_database_failed",
                log_level="error",
                log_message_parts=[f"Could not create database '{postgres_db_name}'"],
            )
            assert len(logs) == 1

    matching_rows = postgres_login_db_connection.execute(
        sa.text(
            f"SELECT 1 FROM pg_catalog.pg_database "
            f"WHERE datname = '{postgres_db_name}'"
        )
    ).rowcount

    assert matching_rows == 1
    tracker_store.engine.dispose()


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_postgres_tracker_store_retrieve_full_tracker(
    tracker_with_restarted_event: DialogueStateTracker,
    postgres_login_db_connection: sa.engine.Connection,
    postgres_login_db_name: str,
    postgres_db_name: str,
) -> None:
    sender_id = tracker_with_restarted_event.sender_id

    postgres_login_db_connection.execute(sa.text(f"CREATE DATABASE {postgres_db_name}"))

    tracker_store = SQLTrackerStore(
        dialect="postgresql",
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        db=postgres_db_name,
        login_db=postgres_login_db_name,
    )
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker is not None
    assert tracker == tracker_with_restarted_event

    tracker_store.engine.dispose()


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_postgres_tracker_store_retrieve(
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
    postgres_login_db_connection: sa.engine.Connection,
    postgres_login_db_name: str,
    postgres_db_name: str,
) -> None:
    sender_id = tracker_with_restarted_event.sender_id

    postgres_login_db_connection.execute(sa.text(f"CREATE DATABASE {postgres_db_name}"))

    tracker_store = SQLTrackerStore(
        dialect="postgresql",
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        db=postgres_db_name,
        login_db=postgres_login_db_name,
    )
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve(sender_id)
    assert tracker is not None

    # the retrieved tracker with the latest session would not contain
    # `action_session_start` event because the SQLTrackerStore filters
    # only the events after `session_started` event

    assert list(tracker.events) == events_after_restart[1:]

    tracker_store.engine.dispose()


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_postgres_tracker_store_delete(
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
    postgres_login_db_connection: sa.engine.Connection,
    postgres_login_db_name: str,
    postgres_db_name: str,
) -> None:
    # Given
    sender_id = tracker_with_restarted_event.sender_id
    postgres_login_db_connection.execute(sa.text(f"CREATE DATABASE {postgres_db_name}"))
    tracker_store = SQLTrackerStore(
        dialect="postgresql",
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        db=postgres_db_name,
        login_db=postgres_login_db_name,
    )
    await tracker_store.save(tracker_with_restarted_event)

    # When
    await tracker_store.delete(sender_id)

    # Then
    tracker = await tracker_store.retrieve(sender_id)
    assert tracker is None

    tracker_store.engine.dispose()


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_postgres_tracker_store_update(
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
    postgres_login_db_connection: sa.engine.Connection,
    postgres_login_db_name: str,
    postgres_db_name: str,
) -> None:
    # Given
    sender_id = uuid.uuid4().hex
    postgres_login_db_connection.execute(sa.text(f"CREATE DATABASE {postgres_db_name}"))
    empty_domain = Domain.empty()
    tracker_store = SQLTrackerStore(
        dialect="postgresql",
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        db=postgres_db_name,
        login_db=postgres_login_db_name,
        domain=empty_domain,
    )
    initial_tracker = DialogueStateTracker.from_events(
        sender_id=sender_id,
        evts=tracker_with_restarted_event.events,
    )
    await tracker_store.save(initial_tracker)

    new_tracker = DialogueStateTracker.from_events(
        sender_id=sender_id,
        evts=events_after_restart,
        slots=empty_domain.slots,
        domain=empty_domain,
    )

    # When
    await tracker_store.update(new_tracker)

    # Then
    updated_tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert updated_tracker.current_state(
        EventVerbosity.ALL
    ) == new_tracker.current_state(EventVerbosity.ALL)

    tracker_store.engine.dispose()


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_postgres_concurrent_initialization_with_advisory_lock(
    postgres_login_db_connection: sa.engine.Connection,
    postgres_login_db_name: str,
    postgres_db_name: str,
    domain: Domain,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that multiple workers can safely initialize tables concurrently.

    This test verifies that the advisory lock mechanism prevents
    race conditions when multiple Sanic workers attempt to create tables
    simultaneously in PostgreSQL.
    """
    import concurrent.futures

    monkeypatch.setenv(ENV_SANIC_WORKERS, "5")

    postgres_login_db_connection.execute(sa.text(f"CREATE DATABASE {postgres_db_name}"))

    tracker_stores = []
    errors = []
    all_logs = []

    def create_tracker_store(worker_id: int):
        """Simulate a Sanic worker creating a tracker store."""
        with structlog.testing.capture_logs() as caplog:
            try:
                tracker_store = SQLTrackerStore(
                    domain=domain,
                    dialect="postgresql",
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    username=POSTGRES_USER,
                    password=POSTGRES_PASSWORD,
                    db=postgres_db_name,
                    login_db=postgres_login_db_name,
                )
                tracker_stores.append(tracker_store)
                all_logs.extend(caplog)
                return tracker_store
            except Exception as e:
                errors.append((worker_id, e))
                raise

    try:
        # Simulate 5 workers starting simultaneously
        num_workers = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(create_tracker_store, i) for i in range(num_workers)
            ]
            concurrent.futures.wait(futures)

        # All workers should have succeeded without IntegrityError
        assert len(errors) == 0

        # Verify that exactly one worker created tables
        tables_created_logs = filter_logs(
            all_logs,
            event="sql_tracker_store.tables_created",
            log_level="debug",
            log_message_parts=["Successfully created database tables."],
        )
        assert len(tables_created_logs) == 1

        # Verify that other workers skipped table creation
        tables_exist_logs = filter_logs(
            all_logs,
            event="sql_tracker_store.tables_already_exist",
            log_level="debug",
            log_message_parts=["Tables already exist, skipping creation."],
        )
        # At least one worker should have skipped table creation
        assert len(tables_exist_logs) >= 1

        # All tracker stores should be functional
        for tracker_store in tracker_stores:
            assert tracker_store.engine.url.database == postgres_db_name
    finally:
        for tracker_store in tracker_stores:
            tracker_store.engine.dispose()


async def test_redis_tracker_store_retrieve_full_tracker(
    tracker_with_restarted_event: DialogueStateTracker,
    redis_tracker_store: RedisTrackerStore,
) -> None:
    sender_id = tracker_with_restarted_event.sender_id

    await redis_tracker_store.save(tracker_with_restarted_event)

    tracker = await redis_tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_redis_tracker_store_retrieve(
    redis_tracker_store: RedisTrackerStore,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    sender_id = tracker_with_restarted_event.sender_id

    await redis_tracker_store.save(tracker_with_restarted_event)

    tracker = await redis_tracker_store.retrieve(sender_id)
    assert list(tracker.events) == events_after_restart


async def test_redis_tracker_store_delete(
    redis_tracker_store: RedisTrackerStore,
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    # Given
    sender_id = tracker_with_restarted_event.sender_id
    await redis_tracker_store.save(tracker_with_restarted_event)

    # When
    await redis_tracker_store.delete(sender_id)

    # Then
    tracker = await redis_tracker_store.retrieve(sender_id)
    assert tracker is None


async def test_redis_tracker_store_update(
    redis_tracker_store: RedisTrackerStore,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    # Given
    sender_id = tracker_with_restarted_event.sender_id
    await redis_tracker_store.save(tracker_with_restarted_event)
    new_tracker = DialogueStateTracker.from_events(
        sender_id=sender_id,
        evts=events_after_restart,
    )

    # When
    await redis_tracker_store.update(new_tracker)

    # Then
    tracker = await redis_tracker_store.retrieve(sender_id)
    assert tracker == new_tracker


async def test_redis_get_trackers_by_user_id(
    domain: Domain,
    redis_tracker_store: RedisTrackerStore,
) -> None:
    user_id = uuid.uuid4().hex
    sender_id_1 = uuid.uuid4().hex
    sender_id_2 = uuid.uuid4().hex

    # Create two trackers with the same user_id
    tracker1 = await create_tracker_with_user_id(
        redis_tracker_store,
        sender_id_1,
        user_id,
        [SessionStarted(), UserUttered("Hello")],
    )
    timestamp1 = tracker1.conversation_started_timestamp

    tracker2 = await create_tracker_with_user_id(
        redis_tracker_store, sender_id_2, user_id, [SessionStarted(), UserUttered("Hi")]
    )
    timestamp2 = tracker2.conversation_started_timestamp

    # Retrieve trackers by user_id
    trackers = await redis_tracker_store.get_trackers_by_user_id(user_id)

    assert len(trackers) == 2
    sender_ids = {tracker.sender_id for tracker in trackers}
    assert sender_id_1 in sender_ids
    assert sender_id_2 in sender_ids
    assert_all_trackers_have_properties(trackers, user_id)

    # Verify timestamps match originals
    tracker_dict = {t.sender_id: t for t in trackers}
    assert tracker_dict[sender_id_1].conversation_started_timestamp == timestamp1
    assert tracker_dict[sender_id_2].conversation_started_timestamp == timestamp2


async def test_redis_get_trackers_by_user_id_with_limit(
    domain: Domain,
    redis_tracker_store: RedisTrackerStore,
) -> None:
    user_id = uuid.uuid4().hex

    # Create 10 trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        redis_tracker_store, user_id, 10
    )

    # Retrieve with limit
    trackers = await redis_tracker_store.get_trackers_by_user_id(user_id, limit=5)

    assert len(trackers) == 5
    assert_all_trackers_have_properties(trackers, user_id)

    assert trackers == saved_trackers[:5]


async def test_redis_get_trackers_by_user_id_with_skip(
    domain: Domain,
    redis_tracker_store: RedisTrackerStore,
) -> None:
    user_id = uuid.uuid4().hex

    # Create 10 trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        redis_tracker_store, user_id, 10
    )

    # Retrieve with skip
    trackers = await redis_tracker_store.get_trackers_by_user_id(user_id, skip=3)

    # Verify results - check both properties
    assert len(trackers) == 7  # 10 total - 3 skipped
    assert_all_trackers_have_properties(trackers, user_id)
    assert trackers == saved_trackers[3:]


async def test_redis_get_trackers_by_user_id_with_skip_and_limit(
    domain: Domain,
    redis_tracker_store: RedisTrackerStore,
) -> None:
    user_id = uuid.uuid4().hex

    # Create 10 trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        redis_tracker_store, user_id, 10
    )

    # Retrieve with skip and limit
    trackers = await redis_tracker_store.get_trackers_by_user_id(
        user_id, skip=2, limit=3
    )

    # Verify results - check both properties
    assert len(trackers) == 3
    assert_all_trackers_have_properties(trackers, user_id)
    assert trackers == saved_trackers[2:5]


async def test_redis_get_trackers_by_user_id_no_matches(
    domain: Domain,
    redis_tracker_store: RedisTrackerStore,
) -> None:
    user_id = "integration_test_user_no_matches"

    # Query for user_id that doesn't exist
    trackers = await redis_tracker_store.get_trackers_by_user_id(user_id)
    assert len(trackers) == 0


async def test_redis_conversation_started_timestamp_backward_compatibility(
    domain: Domain,
    redis_tracker_store: RedisTrackerStore,
) -> None:
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
    await redis_tracker_store.save(tracker)

    # Retrieve and verify both properties
    retrieved = await redis_tracker_store.retrieve(sender_id)
    assert retrieved is not None
    assert_tracker_properties(retrieved, user_id, sender_id, expected_timestamp)

    # Also verify via get_trackers_by_user_id
    trackers = await redis_tracker_store.get_trackers_by_user_id(user_id)
    assert len(trackers) == 1
    assert_tracker_properties(trackers[0], user_id, sender_id, expected_timestamp)


async def test_redis_get_trackers_by_user_id_sorted_by_timestamp(
    domain: Domain,
    redis_tracker_store: RedisTrackerStore,
) -> None:
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
            redis_tracker_store, sender_id, user_id, events
        )

    # Retrieve trackers
    trackers = await redis_tracker_store.get_trackers_by_user_id(user_id)

    # Verify both properties and sorting
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


async def test_redis_get_trackers_by_user_id_filters_by_user_id(
    domain: Domain,
    redis_tracker_store: RedisTrackerStore,
) -> None:
    user_id_1 = uuid.uuid4().hex
    user_id_2 = uuid.uuid4().hex
    sender_id_1 = uuid.uuid4().hex
    sender_id_2 = uuid.uuid4().hex

    # Create trackers with different user_ids
    tracker1 = await create_tracker_with_user_id(
        redis_tracker_store, sender_id_1, user_id_1
    )
    timestamp1 = tracker1.conversation_started_timestamp

    tracker2 = await create_tracker_with_user_id(
        redis_tracker_store, sender_id_2, user_id_2
    )
    timestamp2 = tracker2.conversation_started_timestamp

    # Query for user_id_1
    trackers = await redis_tracker_store.get_trackers_by_user_id(user_id_1)

    # Verify only tracker1 is returned with both properties
    assert len(trackers) == 1
    assert_tracker_properties(trackers[0], user_id_1, sender_id_1, timestamp1)

    # Verify user_id_2 returns different tracker
    trackers2 = await redis_tracker_store.get_trackers_by_user_id(user_id_2)
    assert len(trackers2) == 1
    assert_tracker_properties(trackers2[0], user_id_2, sender_id_2, timestamp2)


@pytest.mark.parametrize("num_conversations", [100, 500, 1000, 2000])
async def test_redis_get_trackers_by_user_id_performance(
    num_conversations: int,
    domain: Domain,
    redis_tracker_store: RedisTrackerStore,
) -> None:
    # Create many trackers for the same user
    user_id = uuid.uuid4().hex
    await create_multiple_trackers_with_user_id(
        redis_tracker_store, user_id, num_conversations, delay=0.0
    )

    # Test retrieval without pagination
    retrieval_start = time.time()
    trackers = await redis_tracker_store.get_trackers_by_user_id(user_id)
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
    page_trackers = await redis_tracker_store.get_trackers_by_user_id(
        user_id, limit=page_size
    )
    paginated_time = time.time() - paginated_start
    assert len(page_trackers) == page_size

    # Paginated queries should be faster
    assert (
        paginated_time < 5.0
    ), f"Paginated retrieval took {paginated_time:.2f}s, expected < 5.0s"


@pytest.fixture
def sql_tracker_store(
    postgres_login_db_connection: sa.engine.Connection,
    postgres_db_name: str,
    postgres_login_db_name: str,
) -> Generator[SQLTrackerStore, None, None]:
    postgres_login_db_connection.execute(sa.text(f"CREATE DATABASE {postgres_db_name}"))
    sql_tracker_store = SQLTrackerStore(
        dialect="postgresql",
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        db=postgres_db_name,
        login_db=postgres_login_db_name,
    )
    yield sql_tracker_store
    sql_tracker_store.engine.dispose()


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_postgres_get_trackers_by_user_id(
    domain: Domain,
    sql_tracker_store: SQLTrackerStore,
) -> None:
    user_id = uuid.uuid4().hex
    sender_id_1 = uuid.uuid4().hex
    sender_id_2 = uuid.uuid4().hex

    # Create two trackers with the same user_id
    tracker1 = await create_tracker_with_user_id(
        sql_tracker_store,
        sender_id_1,
        user_id,
        [SessionStarted(), UserUttered("Hello")],
    )
    timestamp1 = tracker1.conversation_started_timestamp

    tracker2 = await create_tracker_with_user_id(
        sql_tracker_store, sender_id_2, user_id, [SessionStarted(), UserUttered("Hi")]
    )
    timestamp2 = tracker2.conversation_started_timestamp

    # Retrieve trackers by user_id
    trackers = await sql_tracker_store.get_trackers_by_user_id(user_id)

    assert len(trackers) == 2
    sender_ids = {tracker.sender_id for tracker in trackers}
    assert sender_id_1 in sender_ids
    assert sender_id_2 in sender_ids
    assert_all_trackers_have_properties(trackers, user_id)

    # Verify timestamps match originals
    tracker_dict = {t.sender_id: t for t in trackers}
    assert tracker_dict[sender_id_1].conversation_started_timestamp == timestamp1
    assert tracker_dict[sender_id_2].conversation_started_timestamp == timestamp2


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_sql_get_trackers_by_user_id_with_limit(
    domain: Domain,
    sql_tracker_store: SQLTrackerStore,
) -> None:
    user_id = uuid.uuid4().hex

    # Create 10 trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        sql_tracker_store, user_id, 10
    )

    # Retrieve with limit
    trackers = await sql_tracker_store.get_trackers_by_user_id(user_id, limit=5)

    assert len(trackers) == 5
    assert_all_trackers_have_properties(trackers, user_id)

    assert trackers == saved_trackers[:5]


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_sql_get_trackers_by_user_id_with_skip(
    domain: Domain,
    sql_tracker_store: SQLTrackerStore,
) -> None:
    user_id = uuid.uuid4().hex

    # Create 10 trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        sql_tracker_store, user_id, 10
    )

    # Retrieve with skip
    trackers = await sql_tracker_store.get_trackers_by_user_id(user_id, skip=3)

    # Verify results - check both properties
    assert len(trackers) == 7  # 10 total - 3 skipped
    assert_all_trackers_have_properties(trackers, user_id)
    assert trackers == saved_trackers[3:]


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_sql_get_trackers_by_user_id_with_skip_and_limit(
    domain: Domain,
    sql_tracker_store: SQLTrackerStore,
) -> None:
    user_id = uuid.uuid4().hex

    # Create 10 trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        sql_tracker_store, user_id, 10
    )

    # Retrieve with skip and limit
    trackers = await sql_tracker_store.get_trackers_by_user_id(user_id, skip=2, limit=3)

    # Verify results - check both properties
    assert len(trackers) == 3
    assert_all_trackers_have_properties(trackers, user_id)
    assert trackers == saved_trackers[2:5]


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_sql_get_trackers_by_user_id_no_matches(
    domain: Domain,
    sql_tracker_store: SQLTrackerStore,
) -> None:
    user_id = "integration_test_user_no_matches"

    # Query for user_id that doesn't exist
    trackers = await sql_tracker_store.get_trackers_by_user_id(user_id)
    assert len(trackers) == 0


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_sql_conversation_started_timestamp_backward_compatibility(
    domain: Domain,
    sql_tracker_store: SQLTrackerStore,
) -> None:
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
    await sql_tracker_store.save(tracker)

    # Retrieve and verify both properties
    retrieved = await sql_tracker_store.retrieve(sender_id)
    assert retrieved is not None
    assert_tracker_properties(retrieved, user_id, sender_id, expected_timestamp)

    # Also verify via get_trackers_by_user_id
    trackers = await sql_tracker_store.get_trackers_by_user_id(user_id)
    assert len(trackers) == 1
    assert_tracker_properties(trackers[0], user_id, sender_id, expected_timestamp)


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_sql_get_trackers_by_user_id_sorted_by_timestamp(
    domain: Domain,
    sql_tracker_store: SQLTrackerStore,
) -> None:
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
        await create_tracker_with_user_id(sql_tracker_store, sender_id, user_id, events)

    # Retrieve trackers
    trackers = await sql_tracker_store.get_trackers_by_user_id(user_id)

    # Verify both properties and sorting
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


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_sql_get_trackers_by_user_id_filters_by_user_id(
    domain: Domain,
    sql_tracker_store: SQLTrackerStore,
) -> None:
    user_id_1 = uuid.uuid4().hex
    user_id_2 = uuid.uuid4().hex
    sender_id_1 = uuid.uuid4().hex
    sender_id_2 = uuid.uuid4().hex

    # Create trackers with different user_ids
    tracker1 = await create_tracker_with_user_id(
        sql_tracker_store, sender_id_1, user_id_1
    )
    timestamp1 = tracker1.conversation_started_timestamp

    tracker2 = await create_tracker_with_user_id(
        sql_tracker_store, sender_id_2, user_id_2
    )
    timestamp2 = tracker2.conversation_started_timestamp

    # Query for user_id_1
    trackers = await sql_tracker_store.get_trackers_by_user_id(user_id_1)

    # Verify only tracker1 is returned with both properties
    assert len(trackers) == 1
    assert_tracker_properties(trackers[0], user_id_1, sender_id_1, timestamp1)

    # Verify user_id_2 returns different tracker
    trackers2 = await sql_tracker_store.get_trackers_by_user_id(user_id_2)
    assert len(trackers2) == 1
    assert_tracker_properties(trackers2[0], user_id_2, sender_id_2, timestamp2)


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
@pytest.mark.parametrize("num_conversations", [100, 500, 1000, 2000])
async def test_sql_get_trackers_by_user_id_performance(
    num_conversations: int,
    domain: Domain,
    sql_tracker_store: SQLTrackerStore,
) -> None:
    # Create many trackers for the same user
    user_id = uuid.uuid4().hex
    await create_multiple_trackers_with_user_id(
        sql_tracker_store, user_id, num_conversations, delay=0.0
    )

    # Test retrieval without pagination
    retrieval_start = time.time()
    trackers = await sql_tracker_store.get_trackers_by_user_id(user_id)
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
    page_trackers = await sql_tracker_store.get_trackers_by_user_id(
        user_id, limit=page_size
    )
    paginated_time = time.time() - paginated_start
    assert len(page_trackers) == page_size

    # Paginated queries should be faster
    assert (
        paginated_time < 5.0
    ), f"Paginated retrieval took {paginated_time:.2f}s, expected < 5.0s"
