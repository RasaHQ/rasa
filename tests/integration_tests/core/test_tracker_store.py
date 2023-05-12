import logging
from typing import List
from unittest.mock import Mock

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
import sqlalchemy as sa

from rasa.core.tracker_store import (
    MongoTrackerStore,
    RedisTrackerStore,
    SQLTrackerStore,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.trackers import DialogueStateTracker
from .conftest import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_LOGIN_DB,
    POSTGRES_TRACKER_STORE_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
)

# NOTE about the timeouts in this file. We want to fail fast
# because SQLTrackerStore tries to connect several times
# until it works. If the timeout is hit, it probably means
# that something is wrong in the setup of the test


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
def test_sql_tracker_store_with_login_db(
    postgres_login_db_connection: sa.engine.Connection,
):
    tracker_store = SQLTrackerStore(
        dialect="postgresql",
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        db=POSTGRES_TRACKER_STORE_DB,
        login_db=POSTGRES_LOGIN_DB,
    )

    matching_rows = (
        postgres_login_db_connection.execution_options(isolation_level="AUTOCOMMIT")
        .execute(
            sa.text(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = :database_name"
            ),
            database_name=POSTGRES_TRACKER_STORE_DB,
        )
        .rowcount
    )
    assert matching_rows == 1
    assert tracker_store.engine.url.database == POSTGRES_TRACKER_STORE_DB
    tracker_store.engine.dispose()


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
def test_sql_tracker_store_with_login_db_db_already_exists(
    postgres_login_db_connection: sa.engine.Connection,
):
    postgres_login_db_connection.execution_options(
        isolation_level="AUTOCOMMIT"
    ).execute(f"CREATE DATABASE {POSTGRES_TRACKER_STORE_DB}")

    tracker_store = SQLTrackerStore(
        dialect="postgresql",
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        db=POSTGRES_TRACKER_STORE_DB,
        login_db=POSTGRES_LOGIN_DB,
    )

    matching_rows = (
        postgres_login_db_connection.execution_options(isolation_level="AUTOCOMMIT")
        .execute(
            sa.text(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = :database_name"
            ),
            database_name=POSTGRES_TRACKER_STORE_DB,
        )
        .rowcount
    )
    assert matching_rows == 1
    tracker_store.engine.dispose()


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
def test_sql_tracker_store_with_login_db_race_condition(
    postgres_login_db_connection: sa.engine.Connection,
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
):
    original_execute = sa.engine.Connection.execute

    def mock_execute(self, *args, **kwargs):
        # this simulates a race condition
        if kwargs == {"database_name": POSTGRES_TRACKER_STORE_DB}:
            original_execute(
                self.execution_options(isolation_level="AUTOCOMMIT"),
                f"CREATE DATABASE {POSTGRES_TRACKER_STORE_DB}",
            )
            return Mock(rowcount=0)
        else:
            return original_execute(self, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(sa.engine.Connection, "execute", mock_execute)
        with caplog.at_level(logging.ERROR):
            tracker_store = SQLTrackerStore(
                dialect="postgresql",
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                username=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                db=POSTGRES_TRACKER_STORE_DB,
                login_db=POSTGRES_LOGIN_DB,
            )

    # IntegrityError has been caught and we log the error
    assert any(
        [
            f"Could not create database '{POSTGRES_TRACKER_STORE_DB}'" in record.message
            for record in caplog.records
        ]
    )
    matching_rows = (
        postgres_login_db_connection.execution_options(isolation_level="AUTOCOMMIT")
        .execute(
            sa.text(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = :database_name"
            ),
            database_name=POSTGRES_TRACKER_STORE_DB,
        )
        .rowcount
    )
    assert matching_rows == 1
    tracker_store.engine.dispose()


@pytest.mark.sequential
@pytest.mark.timeout(10, func_only=True)
async def test_postgres_tracker_store_retrieve_full_tracker(
    tracker_with_restarted_event: DialogueStateTracker,
    postgres_login_db_connection: sa.engine.Connection,
) -> None:
    sender_id = tracker_with_restarted_event.sender_id

    postgres_login_db_connection.execution_options(
        isolation_level="AUTOCOMMIT"
    ).execute(f"CREATE DATABASE {POSTGRES_TRACKER_STORE_DB}")

    tracker_store = SQLTrackerStore(
        dialect="postgresql",
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        db=POSTGRES_TRACKER_STORE_DB,
        login_db=POSTGRES_LOGIN_DB,
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
) -> None:
    sender_id = tracker_with_restarted_event.sender_id

    postgres_login_db_connection.execution_options(
        isolation_level="AUTOCOMMIT"
    ).execute(f"CREATE DATABASE {POSTGRES_TRACKER_STORE_DB}")

    tracker_store = SQLTrackerStore(
        dialect="postgresql",
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        db=POSTGRES_TRACKER_STORE_DB,
        login_db=POSTGRES_LOGIN_DB,
    )
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve(sender_id)
    assert tracker is not None

    # the retrieved tracker with the latest session would not contain
    # `action_session_start` event because the SQLTrackerStore filters
    # only the events after `session_started` event

    assert list(tracker.events) == events_after_restart[1:]

    tracker_store.engine.dispose()


async def test_redis_tracker_store_retrieve_full_tracker(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    tracker_store = RedisTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_redis_tracker_store_retrieve(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    tracker_store = RedisTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve(sender_id)
    assert list(tracker.events) == events_after_restart


async def test_mongo_tracker_store_retrieve_full_tracker(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    tracker_store = MongoTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_mongo_tracker_store_retrieve(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    tracker_store = MongoTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve(sender_id)

    # the retrieved tracker with the latest session would not contain
    # `action_session_start` event because the MongoTrackerStore filters
    # only the events after `session_started` event
    assert list(tracker.events) == events_after_restart[1:]
