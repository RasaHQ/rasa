import os
import random
import time
import uuid
from typing import Iterator, List, Optional, Text

import pytest
import sqlalchemy as sa

from rasa.core.lock_store import RedisLockStore, RedisLockStoreConfig
from rasa.core.tracker_stores.redis_tracker_store import RedisTrackerStore
from rasa.core.tracker_stores.tracker_store import TrackerStore
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, SessionStarted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)
POSTGRES_USER = os.getenv("POSTGRES_USER", "rasa")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "rasa")
POSTGRES_DEFAULT_DB = os.getenv("POSTGRES_DEFAULT_DB", "postgres")
POSTGRES_TRACKER_STORE_DB = "tracker_store_db"
POSTGRES_LOGIN_DB = "login_db"


@pytest.fixture(
    params=[
        {"deployment_mode": "standard"},
        {
            "deployment_mode": "cluster",
            "endpoints": [
                f"{REDIS_HOST}:7000",
                f"{REDIS_HOST}:7001",
                f"{REDIS_HOST}:7002",
            ],
        },
        {
            "deployment_mode": "sentinel",
            "endpoints": [
                f"{REDIS_HOST}:26379",
                f"{REDIS_HOST}:26380",
                f"{REDIS_HOST}:26381",
            ],
            "sentinel_service": "mymaster",
        },
    ]
)
def redis_lock_store(request: pytest.FixtureRequest) -> Iterator[RedisLockStore]:
    # we need one redis database per worker, otherwise
    # tests conflicts with each others when databases are flushed
    pytest_worker_id = os.getenv("PYTEST_XDIST_WORKER", "gw0")
    redis_database = int(pytest_worker_id.replace("gw", ""))
    # Base configuration
    config = {"host": REDIS_HOST, "port": REDIS_PORT}

    # For cluster mode, don't set db (clusters only support db 0)
    if request.param["deployment_mode"] != "cluster":
        config["db"] = redis_database

    config.update(request.param)

    lock_store = RedisLockStore(RedisLockStoreConfig(**config))
    try:
        yield lock_store
    finally:
        lock_store.red.flushdb()


@pytest.fixture
def postgres_db_name() -> str:
    """Generates a random postgres database name."""
    random_str = "".join([str(random.randint(0, 9)) for i in range(8)])
    return f"{POSTGRES_TRACKER_STORE_DB}_{random_str}"


@pytest.fixture
def postgres_login_db_name() -> str:
    """Generates a random postgres database name."""
    random_str = "".join([str(random.randint(0, 9)) for i in range(8)])
    return f"{POSTGRES_LOGIN_DB}_{random_str}"


@pytest.fixture
def postgres_login_db_connection(
    postgres_db_name: str, postgres_login_db_name: str
) -> Iterator[sa.engine.Connection]:
    engine = sa.create_engine(
        sa.engine.url.URL(
            "postgresql",
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            username=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DEFAULT_DB,
            query={},
        )
    )

    conn = engine.connect()
    conn.execution_options(isolation_level="AUTOCOMMIT")
    try:
        _create_login_db(conn, postgres_login_db_name)
        yield conn
    finally:
        _drop_db(conn, postgres_login_db_name)
        _drop_db(conn, postgres_db_name)
        conn.close()
        engine.dispose()


def _create_login_db(connection: sa.engine.Connection, login_db: str) -> None:
    connection.execute(sa.text(f"CREATE DATABASE {login_db}"))


def _drop_db(connection: sa.engine.Connection, database_name: Text) -> None:
    connection.execute(sa.text(f"DROP DATABASE IF EXISTS {database_name}"))


@pytest.fixture(
    params=[
        {"deployment_mode": "standard"},
        {
            "deployment_mode": "cluster",
            "endpoints": [
                f"{REDIS_HOST}:7000",
                f"{REDIS_HOST}:7001",
                f"{REDIS_HOST}:7002",
            ],
        },
        {
            "deployment_mode": "sentinel",
            "endpoints": [
                f"{REDIS_HOST}:26379",
                f"{REDIS_HOST}:26380",
                f"{REDIS_HOST}:26381",
            ],
            "sentinel_service": "mymaster",
        },
    ]
)
def redis_tracker_store(
    domain: Domain, request: pytest.FixtureRequest
) -> Iterator[RedisTrackerStore]:
    # we need one redis database per worker, otherwise
    # tests conflicts with each others when databases are flushed
    pytest_worker_id = os.getenv("PYTEST_XDIST_WORKER", "gw0")
    redis_database = int(pytest_worker_id.replace("gw", ""))
    # Base configuration
    config = {"domain": domain}

    # For cluster mode, don't set db (clusters only support db 0)
    if request.param["deployment_mode"] != "cluster":
        config["db"] = redis_database

    config.update(request.param)

    tracker_store = RedisTrackerStore(**config)
    try:
        yield tracker_store
    finally:
        tracker_store.red.flushdb()


async def create_tracker_with_user_id(
    tracker_store: TrackerStore,
    sender_id: str,
    user_id: str,
    events: Optional[List[Event]] = None,
) -> DialogueStateTracker:
    """Create and save a tracker with user_id."""
    if events is None:
        events = [SessionStarted(), UserUttered("Hello")]
    tracker = DialogueStateTracker.from_events(
        sender_id,
        events,
        slots=tracker_store.domain.slots,
        domain=tracker_store.domain,
    )
    tracker.user_id = user_id
    await tracker_store.save(tracker)
    return tracker


async def create_multiple_trackers_with_user_id(
    tracker_store: TrackerStore,
    user_id: str,
    count: int,
    delay: float = 0.01,
) -> List[DialogueStateTracker]:
    """Create and save multiple trackers with the same user_id."""
    trackers = []
    for i in range(count):
        sender_id = uuid.uuid4().hex
        tracker = DialogueStateTracker.from_events(
            sender_id,
            [SessionStarted(), UserUttered(f"Message {i}")],
            slots=tracker_store.domain.slots,
            domain=tracker_store.domain,
        )
        tracker.user_id = user_id
        await tracker_store.save(tracker)
        trackers.append(tracker)
        if delay > 0:
            time.sleep(delay)
    return trackers


def assert_tracker_properties(
    tracker: DialogueStateTracker,
    expected_user_id: str,
    expected_sender_id: Optional[str] = None,
    expected_timestamp: Optional[float] = None,
) -> None:
    """Assert tracker has correct user_id and conversation_started_timestamp."""
    assert tracker.user_id == expected_user_id
    assert tracker.conversation_started_timestamp is not None
    if expected_sender_id is not None:
        assert tracker.sender_id == expected_sender_id
    if expected_timestamp is not None:
        assert tracker.conversation_started_timestamp == expected_timestamp


def assert_all_trackers_have_properties(
    trackers: List[DialogueStateTracker], user_id: str
) -> None:
    """Assert all trackers have correct user_id and conversation_started_timestamp."""
    for tracker in trackers:
        assert_tracker_properties(tracker, user_id)
