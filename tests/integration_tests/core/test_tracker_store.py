import pytest
import sqlalchemy as sa

from rasa.core.tracker_store import SQLTrackerStore
from tests.integration_tests.conftest import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_LOGIN_DB,
    POSTGRES_TRACKER_STORE_DB,
)

# NOTE about the timeouts in this file. We want to fail fast
# because SQLTrackerStore tries to connect several times
# until it works. If the timeout is hit, it probably means
# that something is wrong in the setup of the test


@pytest.mark.timeout(10)
def test_sql_tracker_store_with_login_db(
    postgres_login_db_connection: sa.engine.Connection,
):
    tracker_store = SQLTrackerStore(
        dialect="postgresql",
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
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


# TODO: IntegrityError is raised?
