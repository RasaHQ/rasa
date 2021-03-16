import logging
from unittest.mock import Mock

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
import sqlalchemy as sa

from rasa.core.tracker_store import SQLTrackerStore
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
@pytest.mark.timeout(10)
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
@pytest.mark.timeout(10)
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
@pytest.mark.timeout(10)
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
