from typing import List
from unittest.mock import Mock

import pytest
import sqlalchemy as sa
import structlog
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.tracker_store import (
    RedisTrackerStore,
    SQLTrackerStore,
)
from rasa.shared.core.events import Event
from rasa.shared.core.trackers import DialogueStateTracker
from tests.utilities import filter_logs

from .conftest import (
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)

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
