import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Dict, List, Optional, Union
from unittest.mock import MagicMock, Mock, patch

import pytest
import sqlalchemy
from moto import mock_aws
from moto.core import set_initial_no_auth_action_count
from pytest import CaptureFixture, LogCaptureFixture, MonkeyPatch
from sqlalchemy import URL, Engine, make_url
from sqlalchemy.dialects.oracle.base import OracleDialect
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.dialects.sqlite.base import SQLiteDialect
from structlog.testing import capture_logs

from rasa.constants import ENV_SANIC_WORKERS
from rasa.core.agent import Agent
from rasa.core.constants import (
    IAM_CLOUD_PROVIDER_ENV_VAR_NAME,
    POSTGRESQL_SCHEMA,
    RDS_SQL_DB_AWS_IAM_ENABLED_ENV_VAR_NAME,
    SQL_TRACKER_STORE_SSL_MODE_ENV_VAR_NAME,
    SQL_TRACKER_STORE_SSL_ROOT_CERTIFICATE_ENV_VAR_NAME,
)
from rasa.core.tracker_stores.auth_retry_tracker_store import AuthRetryTrackerStore
from rasa.core.tracker_stores.sql_tracker_store import (
    POSTGRESQL_DEFAULT_MAX_OVERFLOW,
    POSTGRESQL_DEFAULT_POOL_SIZE,
    SQLTrackerStore,
    create_engine_kwargs,
    ensure_schema_exists,
    get_ssl_args,
    is_postgresql_url,
)
from rasa.core.tracker_stores.tracker_store import (
    TrackerStore,
    check_if_tracker_store_async,
    create_tracker_store,
)
from rasa.shared.core.constants import ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    Event,
    SessionStarted,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from tests.conftest import with_session_ids
from tests.core.tracker_stores.conftest import (
    _saved_tracker_with_multiple_session_starts,
    assert_all_trackers_have_user_id,
    assert_tracker_has_user_id,
    create_multiple_trackers_with_user_id,
    create_tracker_with_partially_saved_events,
    create_tracker_with_user_id,
    create_trackers_with_same_timestamp,
    old_tracker_gets_timestamp_on_save,
    old_tracker_gets_timestamp_on_update,
    prepare_token_serialisation,
)
from tests.utilities import filter_logs


# SQL-specific helper function
def get_user_id_from_users_table(
    tracker_store: SQLTrackerStore, sender_id: str
) -> Optional[str]:
    """Query users table for user_id associated with sender_id.

    Args:
        tracker_store: The tracker store.
        sender_id: Sender ID to query.

    Returns:
        User ID if found, None otherwise.
    """
    with tracker_store.session_scope() as session:
        user_mapping = (
            session.query(tracker_store.SQLUser.user_id)
            .filter(tracker_store.SQLUser.sender_id == sender_id)
            .first()
        )
        return user_mapping[0] if user_mapping else None


@pytest.mark.parametrize(
    "full_url",
    [
        "postgresql://localhost",
        "postgresql://localhost:5432",
        "postgresql://user:secret@localhost",
        "sqlite:///",
    ],
)
def test_get_db_url_with_fully_specified_url(full_url: str):
    assert SQLTrackerStore.get_db_url(host=full_url) == full_url


def test_get_db_url_with_port_in_host():
    host = "localhost:1234"
    dialect = "postgresql"
    db = "mydb"

    expected = f"{dialect}://{host}/{db}"

    assert (
        str(SQLTrackerStore.get_db_url(dialect=dialect, host=host, db=db)) == expected
    )


def test_db_get_url_with_sqlite():
    expected = "sqlite:///rasa.db"
    assert str(SQLTrackerStore.get_db_url(dialect="sqlite", db="rasa.db")) == expected


def test_get_db_url_with_correct_host():
    expected = "postgresql://localhost:5005/mydb"

    assert (
        str(
            SQLTrackerStore.get_db_url(
                dialect="postgresql", host="localhost", port=5005, db="mydb"
            )
        )
        == expected
    )


def test_get_db_url_with_query():
    expected = "postgresql://localhost:5005/mydb?driver=my-driver"

    assert (
        str(
            SQLTrackerStore.get_db_url(
                dialect="postgresql",
                host="localhost",
                port=5005,
                db="mydb",
                query={"driver": "my-driver"},
            )
        )
        == expected
    )


def test_sql_tracker_store_logs_do_not_show_password(caplog: LogCaptureFixture):
    dialect = "postgresql"
    host = "localhost"
    port = 9901
    db = "some-database"
    username = "db-user"
    # deepcode ignore NoHardcodedPasswords/test: Test credential
    password = "some-password"

    with capture_logs() as caplog:
        _ = SQLTrackerStore(None, dialect, host, port, db, username, password)
        # instead the password is displayed as '***'
        logs = filter_logs(
            caplog,
            event="sql_tracker_store.connect_to_sql_database",
            log_level="debug",
            log_message_parts=[f"postgresql://{username}:***@{host}:{port}/{db}"],
        )
        assert len(logs) == 1
        # the URL in the logs does not contain the password
        logs = filter_logs(
            caplog,
            log_message_parts=[password],
        )
        assert len(logs) == 0


def test_db_url_with_query_from_endpoint_config(tmp_path: Path):
    endpoint_config = """
    tracker_store:
      dialect: postgresql
      url: localhost
      port: 5123
      username: user
      password: pw
      login_db: login-db
      query:
        driver: my-driver
        another: query
    """
    f = tmp_path / "tmp_config_file.yml"
    f.write_text(endpoint_config)
    store_config = read_endpoint_config(str(f), "tracker_store")

    url = SQLTrackerStore.get_db_url(**store_config.kwargs)

    import itertools

    # order of query dictionary in yaml is random, test against both permutations
    connection_url = "postgresql://user:***@:5123/login-db?"

    assert any(
        str(url) == connection_url + "&".join(permutation)
        for permutation in (
            itertools.permutations(("another=query", "driver=my-driver"))
        )
    )


# we cannot parametrise over this and the previous test due to the different ways of
# calling _additional_events()
async def test_sql_additional_events(domain: Domain):
    tracker_store = SQLTrackerStore(domain)
    additional_events, tracker = await create_tracker_with_partially_saved_events(
        tracker_store
    )

    # make sure only new events are returned
    with tracker_store.session_scope() as session:
        # noinspection PyProtectedMember
        assert (
            list(tracker_store._additional_events(session, tracker))
            == additional_events
        )


async def test_sql_additional_events_with_session_start(domain: Domain):
    sender = "test_sql_additional_events_with_session_start"
    tracker_store = SQLTrackerStore(domain)
    tracker = await _saved_tracker_with_multiple_session_starts(tracker_store, sender)

    tracker.update(UserUttered("hi2"), domain)

    # make sure only new events are returned
    with tracker_store.session_scope() as session:
        # noinspection PyProtectedMember
        additional_events = list(tracker_store._additional_events(session, tracker))
        assert len(additional_events) == 1
        assert isinstance(additional_events[0], UserUttered)


async def test_tracker_store_retrieve_ordered_by_id(
    domain: Domain,
):
    tracker_store_kwargs = {"host": "sqlite:///"}
    tracker_store = SQLTrackerStore(domain, **tracker_store_kwargs)
    events = [
        SessionStarted(timestamp=1),
        UserUttered("Hola", {"name": "greet"}, timestamp=2),
        BotUttered("Hi", timestamp=2),
        UserUttered("How are you?", {"name": "greet"}, timestamp=2),
        BotUttered("I am good, whats up", timestamp=2),
        UserUttered("Ciao", {"name": "greet"}, timestamp=2),
        BotUttered("Bye", timestamp=2),
    ]
    sender_id = "test_sql_tracker_store_events_order"
    tracker = DialogueStateTracker.from_events(sender_id, events)
    await tracker_store.save(tracker)

    # Save other tracker to ensure that we don't run into problems with other senders
    other_tracker = DialogueStateTracker.from_events("other-sender", [SessionStarted()])
    await tracker_store.save(other_tracker)

    # Retrieve tracker with events since latest SessionStarted
    tracker = await tracker_store.retrieve(sender_id)

    assert len(tracker.events) == 7
    # assert the order of events is same as the order in which they were added
    assert all((event == tracker.events[i] for i, event in enumerate(events)))


def test_session_scope_error(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture, domain: Domain
):
    tracker_store = SQLTrackerStore(domain)
    tracker_store.sessionmaker = Mock()

    requested_schema = uuid.uuid4().hex

    # `ensure_schema_exists()` raises `ValueError`
    mocked_ensure_schema_exists = Mock(side_effect=ValueError(requested_schema))
    monkeypatch.setattr(
        "rasa.core.tracker_stores.sql_tracker_store.ensure_schema_exists",
        mocked_ensure_schema_exists,
    )

    # `SystemExit` is triggered by failing `ensure_schema_exists()`
    with pytest.raises(SystemExit):
        with tracker_store.session_scope() as _:
            pass

    # error message is printed
    assert (
        f"Requested PostgreSQL schema '{requested_schema}' was not found in the "
        f"database." in capsys.readouterr()[0]
    )


@pytest.mark.parametrize(
    "url,is_postgres_url",
    [
        (f"{PGDialect.name}://admin:pw@localhost:5432/rasa", True),
        (f"{SQLiteDialect.name}:///", False),
        (URL(PGDialect.name, None, None, None, None, None, {}), True),
        (URL(SQLiteDialect.name, None, None, None, None, None, {}), False),
    ],
)
def test_is_postgres_url(url: Union[str, URL], is_postgres_url: bool):
    assert is_postgresql_url(url) == is_postgres_url


def set_or_delete_postgresql_schema_env_var(
    monkeypatch: MonkeyPatch, value: Optional[str]
) -> None:
    """Set `POSTGRESQL_SCHEMA` environment variable using `MonkeyPatch`.

    Args:
        monkeypatch: Instance of `MonkeyPatch` to use for patching.
        value: Value of the `POSTGRESQL_SCHEMA` environment variable to set.
    """
    if value is None:
        monkeypatch.delenv(POSTGRESQL_SCHEMA, raising=False)
    else:
        monkeypatch.setenv(POSTGRESQL_SCHEMA, value)


@pytest.mark.parametrize(
    "url,schema_env,kwargs",
    [
        # postgres without schema
        (
            f"{PGDialect.name}://admin:pw@localhost:5432/rasa",
            None,
            {
                "pool_size": POSTGRESQL_DEFAULT_POOL_SIZE,
                "max_overflow": POSTGRESQL_DEFAULT_MAX_OVERFLOW,
            },
        ),
        # postgres with schema
        (
            f"{PGDialect.name}://admin:pw@localhost:5432/rasa",
            "schema1",
            {
                "connect_args": {"options": "-csearch_path=schema1"},
                "pool_size": POSTGRESQL_DEFAULT_POOL_SIZE,
                "max_overflow": POSTGRESQL_DEFAULT_MAX_OVERFLOW,
            },
        ),
        # oracle without schema
        (f"{OracleDialect.name}://admin:pw@localhost:5432/rasa", None, {}),
        # oracle with schema
        (f"{OracleDialect.name}://admin:pw@localhost:5432/rasa", "schema1", {}),
        # sqlite
        (f"{SQLiteDialect.name}:///", None, {}),
    ],
)
def test_create_engine_kwargs(
    monkeypatch: MonkeyPatch,
    url: Union[str, URL],
    schema_env: Optional[str],
    kwargs: Dict[str, Dict[str, Union[str, int]]],
):
    set_or_delete_postgresql_schema_env_var(monkeypatch, schema_env)

    assert create_engine_kwargs(url) == kwargs


@contextmanager
def does_not_raise():
    """Contextmanager to be used when an expression is not expected to raise an
    exception.

    This contextmanager can be used in parametrized tests, where some input objects
    are expected to raise and others are not.

    Example:
        @pytest.mark.parametrize(
            "a,b,raises_context",
            [
                # 5/6 is a legal divison
                (5, 6, does_not_raise()),
                # 5/0 raises a `ZeroDivisionError`
                (5, 0, pytest.raises(ZeroDivisionError)),
            ],
        )
        def test_divide(
            a: int, b: int, raises_context: ContextManager,
        ):
            with raises_context:
                _ = a / b

    """
    yield


@pytest.mark.parametrize(
    "is_postgres,schema_env,schema_exists,raises_context",
    [
        (True, "schema1", True, does_not_raise()),
        (True, "schema1", False, pytest.raises(ValueError)),
        (False, "schema1", False, does_not_raise()),
        (True, None, False, does_not_raise()),
        (False, None, False, does_not_raise()),
    ],
)
def test_ensure_schema_exists(
    monkeypatch: MonkeyPatch,
    is_postgres: bool,
    schema_env: Optional[str],
    schema_exists: bool,
    raises_context: ContextManager,
):
    set_or_delete_postgresql_schema_env_var(monkeypatch, schema_env)
    monkeypatch.setattr(
        "rasa.core.tracker_stores.sql_tracker_store.is_postgresql_url",
        lambda _: is_postgres,
    )
    monkeypatch.setattr(sqlalchemy, "exists", Mock())

    # mock the `session.query().scalar()` query which returns whether the schema
    # exists in the db
    scalar = Mock(return_value=schema_exists)
    query = Mock(scalar=scalar)
    session = Mock()
    engine = Mock(spec=Engine)
    engine.url = Mock()
    session.get_bind = Mock(return_value=engine)
    session.query = Mock(return_value=query)

    with raises_context:
        ensure_schema_exists(session)


def test_login_db_with_no_postgresql(tmp_path: Path):
    with pytest.warns(UserWarning):
        SQLTrackerStore(db=str(tmp_path / "rasa.db"), login_db=str(tmp_path / "other"))


def test_sql_tracker_store_with_token_serialisation(
    domain: Domain, flow_policy_bot_agent: Agent
):
    tracker_store = SQLTrackerStore(domain, **{"host": "sqlite:///"})
    prepare_token_serialisation(tracker_store, flow_policy_bot_agent, "sql")


def test_sql_tracker_store_creation_with_invalid_port(domain: Domain):
    with pytest.raises(RasaException) as error:
        TrackerStore.create(
            EndpointConfig(port="$DB_PORT", type="sql"),
            domain,
        )
    assert "port '$DB_PORT' cannot be cast to integer." in str(error.value)


async def test_sql_get_or_create_full_tracker_without_action_listen() -> None:
    tracker_store = SQLTrackerStore(Domain.empty())
    sender_id = uuid.uuid4().hex
    tracker = await tracker_store.get_or_create_full_tracker(
        sender_id=sender_id, append_action_listen=False
    )
    assert tracker.sender_id == sender_id
    assert tracker.events == deque()


async def test_sql_get_or_create_full_tracker_with_action_listen(
    mock_session_id: str,
) -> None:
    tracker_store = SQLTrackerStore(Domain.empty())
    sender_id = uuid.uuid4().hex
    tracker = await tracker_store.get_or_create_full_tracker(
        sender_id=sender_id, append_action_listen=True
    )
    assert tracker.sender_id == sender_id
    expected_event = with_session_ids(
        [ActionExecuted(ACTION_LISTEN_NAME)], mock_session_id
    )

    assert list(tracker.events) == expected_event


async def test_sql_get_or_create_full_tracker_with_existing_tracker(
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    sender_id = tracker_with_restarted_event.sender_id

    tracker_store = SQLTrackerStore(Domain.empty())
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.get_or_create_full_tracker(
        sender_id=sender_id, append_action_listen=False
    )
    assert tracker.sender_id == sender_id
    assert tracker.events == deque(tracker_with_restarted_event.events)


async def test_sql_tracker_store_retrieve_full_tracker(
    domain: Domain, tracker_with_restarted_event: DialogueStateTracker
) -> None:
    tracker_store = SQLTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_sql_tracker_store_retrieve(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    tracker_store = SQLTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve(sender_id)

    # the retrieved tracker with the latest session would not contain
    # `action_session_start` event because the SQLTrackerStore filters
    # only the events after `session_started` event
    assert list(tracker.events) == events_after_restart[1:]


def test_create_tracker_store_from_endpoints_file_in_sql_tracker_store(
    domain: Domain,
) -> None:
    endpoint_config = read_endpoint_config(
        "data/test_endpoints/endpoints_sql.yml", "tracker_store"
    )
    tracker_store = create_tracker_store(endpoint_config, domain)

    assert check_if_tracker_store_async(tracker_store) is True
    assert isinstance(tracker_store, SQLTrackerStore)


async def test_tracker_store_counts_conversations() -> None:
    tracker_store = SQLTrackerStore(Domain.empty(), **{"host": "sqlite:///"})

    # Create two trackers
    tracker1 = DialogueStateTracker.from_events("1", [SessionStarted(timestamp=1)])
    tracker2 = DialogueStateTracker.from_events("2", [SessionStarted(timestamp=3)])
    await tracker_store.save(tracker1)
    await tracker_store.save(tracker2)

    # Assert that the tracker store counts the conversations correctly
    assert await tracker_store.count_conversations() == 2
    assert await tracker_store.count_conversations(after_timestamp=2) == 1
    assert await tracker_store.count_conversations(after_timestamp=4) == 0

    # Create another tracker
    tracker3 = DialogueStateTracker.from_events("3", [SessionStarted(timestamp=5)])
    await tracker_store.save(tracker3)

    # Assert that the tracker store counts the conversations correctly
    assert await tracker_store.count_conversations() == 3
    assert await tracker_store.count_conversations(after_timestamp=4) == 1


async def test_sql_tracker_store_retrieve_with_session_started_events(
    domain: Domain,
):
    tracker_store = SQLTrackerStore(domain, **{"host": "sqlite:///"})
    events = [
        UserUttered("Hola", {"name": "greet"}, timestamp=1),
        BotUttered("Hi", timestamp=2),
        SessionStarted(timestamp=3),
        UserUttered("Ciao", {"name": "greet"}, timestamp=4),
    ]
    sender_id = "test_sql_tracker_store_with_session_events"
    tracker = DialogueStateTracker.from_events(sender_id, events)
    await tracker_store.save(tracker)

    # Save other tracker to ensure that we don't run into problems with other senders
    other_tracker = DialogueStateTracker.from_events("other-sender", [SessionStarted()])
    await tracker_store.save(other_tracker)

    # Retrieve tracker with events since latest SessionStarted
    tracker = await tracker_store.retrieve(sender_id)

    assert len(tracker.events) == 2
    assert all((event == tracker.events[i] for i, event in enumerate(events[2:])))


async def test_sql_tracker_store_retrieve_without_session_started_events(
    domain,
) -> None:
    tracker_store = SQLTrackerStore(domain, **{"host": "sqlite:///"})

    # Create tracker with a SessionStarted event
    events = [
        UserUttered("Hola", {"name": "greet"}),
        BotUttered("Hi"),
        UserUttered("Ciao", {"name": "greet"}),
        BotUttered("Hi2"),
    ]

    sender_id = "test_sql_tracker_store_retrieve_without_session_started_events"
    tracker = DialogueStateTracker.from_events(sender_id, events)
    await tracker_store.save(tracker)

    # Save other tracker to ensure that we don't run into problems with other senders
    other_tracker = DialogueStateTracker.from_events("other-sender", [SessionStarted()])
    await tracker_store.save(other_tracker)

    tracker = await tracker_store.retrieve(sender_id)

    assert len(tracker.events) == 4
    assert all(event == tracker.events[i] for i, event in enumerate(events))


async def test_sql_tracker_store_retrieve_with_events_from_previous_sessions() -> None:
    tracker_store = SQLTrackerStore(Domain.empty(), **{"host": "sqlite:///"})

    conversation_id = uuid.uuid4().hex
    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("hi"),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
        ],
    )
    await tracker_store.save(tracker)

    actual = await tracker_store.retrieve_full_tracker(conversation_id)

    assert len(actual.events) == len(tracker.events)


async def test_sql_tracker_store_delete_tracker() -> None:
    # Given
    tracker_store = SQLTrackerStore(Domain.empty(), **{"host": "sqlite:///"})

    conversation_id = uuid.uuid4().hex
    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("hi"),
        ],
    )
    await tracker_store.save(tracker)

    # When
    with capture_logs() as caplog:
        await tracker_store.delete(conversation_id)
        logs = filter_logs(
            caplog,
            event="sql_tracker_store.delete.deleted_tracker",
            log_level="info",
        )

        assert len(logs) == 1

    # Then
    retrieved_tracker = await tracker_store.retrieve(conversation_id)
    assert retrieved_tracker is None


async def test_sql_tracker_store_delete_no_tracker() -> None:
    with capture_logs() as caplog:
        tracker_store = SQLTrackerStore(Domain.empty(), **{"host": "sqlite:///"})
        conversation_id = uuid.uuid4().hex
        await tracker_store.delete(conversation_id)
        logs = filter_logs(
            caplog,
            event="sql_tracker_store.delete.no_tracker_for_sender_id",
            log_level="info",
            log_message_parts=[
                f"Could not find tracker for conversation ID '{conversation_id}'."
            ],
        )

        assert len(logs) == 1


async def test_sql_tracker_store_update_tracker() -> None:
    # Given
    sender_id = uuid.uuid4().hex
    empty_domain = Domain.empty()
    tracker_store = SQLTrackerStore(empty_domain, **{"host": "sqlite:///"})
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("hi"),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("What's the weather like today?"),
        ],
    )
    await tracker_store.save(tracker)
    new_events = list(tracker.events)[3:]
    new_tracker = DialogueStateTracker.from_events(
        sender_id,
        new_events,
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


@pytest.mark.asyncio
async def test_sql_tracker_store_update_in_place_when_no_rows_deleted_and_count_matches() -> (  # noqa: E501
    None
):
    """When delete removes 0 rows and count matches, events are updated in place."""
    sender_id = uuid.uuid4().hex
    empty_domain = Domain.empty()
    tracker_store = SQLTrackerStore(empty_domain, **{"host": "sqlite:///"})
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("secret message"),
            BotUttered("ok"),
        ],
        domain=empty_domain,
    )
    await tracker_store.save(tracker)

    # Build tracker with same events but one content change (e.g. anonymization).
    stored = await tracker_store.retrieve_full_tracker(sender_id)
    assert stored is not None
    modified_events: List[Event] = []
    for evt in stored.events:
        if isinstance(evt, UserUttered) and evt.text == "secret message":
            modified_events.append(UserUttered("REDACTED", timestamp=evt.timestamp))
        else:
            modified_events.append(evt)
    modified_tracker = DialogueStateTracker.from_events(
        sender_id,
        modified_events,
        slots=empty_domain.slots,
        domain=empty_domain,
    )

    await tracker_store.update(modified_tracker, apply_deletion_only=False)

    updated = await tracker_store.retrieve_full_tracker(sender_id)
    assert updated is not None
    user_events = [e for e in updated.events if isinstance(e, UserUttered)]
    assert len(user_events) == 1
    assert user_events[0].text == "REDACTED"


@pytest.mark.asyncio
async def test_sql_tracker_store_update_full_replace_when_no_rows_deleted_and_count_differs() -> (  # noqa: E501
    None
):
    """When delete removes 0 rows and count differs, full replace."""
    sender_id = uuid.uuid4().hex
    empty_domain = Domain.empty()
    tracker_store = SQLTrackerStore(empty_domain, **{"host": "sqlite:///"})
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("one"),
            UserUttered("two"),
        ],
        domain=empty_domain,
    )
    await tracker_store.save(tracker)

    # Fewer events (count differs); same first-event timestamp so delete removes 0.
    stored = await tracker_store.retrieve_full_tracker(sender_id)
    assert stored is not None
    first_ts = stored.events[0].timestamp
    replacement_events = [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=first_ts),
        SessionStarted(timestamp=first_ts + 0.1),
        UserUttered("only one", timestamp=first_ts + 0.2),
    ]
    replacement_tracker = DialogueStateTracker.from_events(
        sender_id,
        replacement_events,
        slots=empty_domain.slots,
        domain=empty_domain,
    )

    await tracker_store.update(replacement_tracker, apply_deletion_only=False)

    updated = await tracker_store.retrieve_full_tracker(sender_id)
    assert updated is not None
    assert len(updated.events) == 3
    user_events = [e for e in updated.events if isinstance(e, UserUttered)]
    assert len(user_events) == 1
    assert user_events[0].text == "only one"


@pytest.mark.asyncio
async def test_sql_tracker_store_update_empty_events_no_crash() -> None:
    """Update with no events does not crash; existing events are retained."""
    sender_id = uuid.uuid4().hex
    empty_domain = Domain.empty()
    tracker_store = SQLTrackerStore(empty_domain, **{"host": "sqlite:///"})
    tracker_with_events = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("hello"),
        ],
        domain=empty_domain,
    )
    await tracker_store.save(tracker_with_events)

    empty_tracker = DialogueStateTracker.from_events(
        sender_id,
        [],
        domain=empty_domain,
    )
    await tracker_store.update(empty_tracker)

    # No exception; log does not dereference events[0]; existing events are unchanged.
    stored = await tracker_store.retrieve_full_tracker(sender_id)
    assert stored is not None
    assert len(stored.events) == 3
    user_events = [e for e in stored.events if isinstance(e, UserUttered)]
    assert len(user_events) == 1
    assert user_events[0].text == "hello"


@pytest.mark.asyncio
async def test_sql_tracker_store_update_deletion_rowcount_zero_does_not_run_content_only() -> (  # noqa: E501
    None
):
    """With apply_deletion_only=True (default), rowcount==0 must not run content-only.

    When deletion cron calls update(tracker_subset) and no rows have timestamp <
    first_ts (e.g. all events share the threshold timestamp), delete removes 0 rows.
    We must not run in-place update or full replace, or we would corrupt the tracker.
    """
    sender_id = uuid.uuid4().hex
    empty_domain = Domain.empty()
    tracker_store = SQLTrackerStore(empty_domain, **{"host": "sqlite:///"})
    # All events with same timestamp so delete WHERE timestamp < first_ts removes 0.
    ts = 1000.0
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME, timestamp=ts),
            SessionStarted(timestamp=ts + 0.1),
            UserUttered("one", timestamp=ts + 0.2),
            UserUttered("two", timestamp=ts + 0.3),
        ],
        domain=empty_domain,
    )
    await tracker_store.save(tracker)

    # Subset with same first-event timestamp; delete will remove 0 rows.
    subset_tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME, timestamp=ts),
            SessionStarted(timestamp=ts + 0.2),
            UserUttered("three", timestamp=ts + 0.3),
        ],
        slots=empty_domain.slots,
        domain=empty_domain,
    )
    await tracker_store.update(subset_tracker)  # default apply_deletion_only=True

    # Content-only path must not have run: we must still have 4 events (no full replace)
    stored = await tracker_store.retrieve_full_tracker(sender_id)
    assert stored is not None
    assert len(stored.events) == 4
    user_texts = [e.text for e in stored.events if isinstance(e, UserUttered)]
    assert "one" in user_texts and "two" in user_texts


@set_initial_no_auth_action_count(1)
@mock_aws
def test_sql_tracker_store_creation_with_iam_enabled(
    monkeypatch: MonkeyPatch,
    domain: Domain,
    capsys: CaptureFixture,
):
    monkeypatch.setenv(IAM_CLOUD_PROVIDER_ENV_VAR_NAME, "aws")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv(RDS_SQL_DB_AWS_IAM_ENABLED_ENV_VAR_NAME, "true")
    # intentionally do not pass password input
    tracker_store = TrackerStore.create(
        EndpointConfig(
            url="localhost",
            username="test_user",
            port=5432,
            type="sql",
            dialect="postgresql",
        ),
        domain,
    )
    assert isinstance(tracker_store, AuthRetryTrackerStore)
    assert isinstance(tracker_store._tracker_store, SQLTrackerStore)
    assert tracker_store._tracker_store.engine.url.password is not None

    captured = capsys.readouterr()
    assert "rasa.core.aws_rds_iam_credentials_provider.get_credentials" in captured.out
    assert (
        "rasa.core.aws_rds_iam_credentials_provider.generated_credentials"
        in captured.out
    )
    assert "sql_tracker_store.iam_credentials_provider " in captured.out
    assert (
        "event_info='Using temporary auth token from "
        "IAM credentials provider.'" in captured.out
    )


def test_get_ssl_args(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(SQL_TRACKER_STORE_SSL_MODE_ENV_VAR_NAME, "verify-full")
    monkeypatch.setenv(
        SQL_TRACKER_STORE_SSL_ROOT_CERTIFICATE_ENV_VAR_NAME, "/path/to/cert"
    )

    ssl_args = get_ssl_args()
    assert ssl_args == {"sslmode": "verify-full", "sslrootcert": "/path/to/cert"}


@set_initial_no_auth_action_count(1)
@mock_aws
def test_sql_tracker_store_creation_with_iam_enabled_and_ssl_args(
    monkeypatch: MonkeyPatch,
    domain: Domain,
    capsys: CaptureFixture,
):
    monkeypatch.setenv(IAM_CLOUD_PROVIDER_ENV_VAR_NAME, "aws")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv(SQL_TRACKER_STORE_SSL_MODE_ENV_VAR_NAME, "verify-full")
    monkeypatch.setenv(
        SQL_TRACKER_STORE_SSL_ROOT_CERTIFICATE_ENV_VAR_NAME, "/path/to/cert"
    )

    mock_engine = Mock(spec=Engine)
    mock_engine.url = make_url("postgresql://test_user:***@localhost:5432/rasa.db")

    mock_conn = Mock()
    mock_engine.begin = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )

    mock_create_engine = MagicMock(return_value=mock_engine)
    monkeypatch.setattr(sqlalchemy, "create_engine", mock_create_engine)

    mock_inspector = Mock()
    mock_inspector.has_table.return_value = False
    monkeypatch.setattr(
        "sqlalchemy.engine.Inspector.from_engine", Mock(return_value=mock_inspector)
    )

    # intentionally do not pass password input
    tracker_store = TrackerStore.create(
        EndpointConfig(
            url="localhost",
            username="test_user",
            port=5432,
            type="sql",
            dialect="postgresql",
        ),
        domain,
    )
    assert isinstance(tracker_store, AuthRetryTrackerStore)
    assert isinstance(tracker_store._tracker_store, SQLTrackerStore)

    assert mock_create_engine.call_count == 1

    ssl_args = {"sslmode": "verify-full", "sslrootcert": "/path/to/cert"}
    assert ssl_args == mock_create_engine.call_args[1]["connect_args"]


async def test_sql_tracker_store_concurrent_initialization_with_advisory_lock(
    domain: Domain,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that PostgreSQL uses advisory locks during table creation.

    This verifies that when using PostgreSQL, the tracker store
    uses advisory locks to prevent race conditions during concurrent
    initialization by multiple Sanic workers.
    """
    monkeypatch.setenv(ENV_SANIC_WORKERS, "2")

    mock_engine = Mock(spec=Engine)
    mock_engine.url = make_url("postgresql://test_user:***@localhost:5432/test_db")

    mock_create_engine = MagicMock(return_value=mock_engine)
    monkeypatch.setattr(sqlalchemy, "create_engine", mock_create_engine)

    mock_advisory_lock = Mock()
    monkeypatch.setattr(
        SQLTrackerStore, "_create_tables_with_advisory_lock", mock_advisory_lock
    )

    SQLTrackerStore(domain=domain, dialect="postgresql")

    mock_advisory_lock.assert_called_once()


def test_sql_tracker_store_non_postgresql_skips_advisory_lock(
    domain: Domain,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that non-PostgreSQL databases don't use advisory locks."""
    monkeypatch.setenv(ENV_SANIC_WORKERS, "2")

    with patch.object(
        SQLTrackerStore,
        "_create_tables_with_advisory_lock",
    ) as mock_advisory_lock:
        with capture_logs() as caplog:
            tracker_store = SQLTrackerStore(domain, **{"host": "sqlite:///"})

            # Verify tracker store was created successfully
            assert isinstance(tracker_store, SQLTrackerStore)

            # Verify advisory lock method was NOT called for SQLite
            mock_advisory_lock.assert_not_called()

            # Verify warning was logged
            logs = filter_logs(
                caplog,
                event="sql_tracker_store.multiple_workers_without_locking",
                log_level="warning",
                log_message_parts=[
                    "Advisory lock mechanism is not supported for non-PostgreSQL "
                    "databases when using multiple Sanic workers."
                ],
            )
            assert len(logs) == 1


def test_sql_tracker_store_default_single_worker_skips_advisory_lock(
    domain: Domain,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that PostgreSQL skips advisory locks with single Sanic worker."""
    mock_engine = Mock(spec=Engine)
    mock_engine.url = make_url("postgresql://test_user:***@localhost:5432/test_db")

    mock_create_engine = MagicMock(return_value=mock_engine)
    monkeypatch.setattr(sqlalchemy, "create_engine", mock_create_engine)

    mock_advisory_lock = Mock()
    monkeypatch.setattr(
        SQLTrackerStore, "_create_tables_with_advisory_lock", mock_advisory_lock
    )

    SQLTrackerStore(domain=domain, dialect="postgresql")

    # Verify advisory lock method was NOT called with single worker
    mock_advisory_lock.assert_not_called()


async def test_sql_tracker_store_advisory_lock_released_on_error(
    domain: Domain,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that advisory lock is released even when table creation fails."""
    monkeypatch.setenv(ENV_SANIC_WORKERS, "2")

    lock_calls = []
    mock_conn = Mock()

    def mock_execute(statement):
        """Track advisory lock and unlock SQL statements to verify execution order."""
        statement_str = str(statement)
        if "pg_advisory_lock" in statement_str:
            lock_calls.append("lock")
        elif "pg_advisory_unlock" in statement_str:
            lock_calls.append("unlock")
        return Mock()

    mock_conn.execute = mock_execute

    mock_engine = Mock(spec=Engine)
    mock_engine.url = make_url("postgresql://test_user:***@localhost:5432/test_db")
    mock_engine.begin = Mock(
        return_value=Mock(
            __enter__=Mock(return_value=mock_conn),
            __exit__=Mock(return_value=False),
        )
    )

    mock_create_engine = MagicMock(return_value=mock_engine)
    monkeypatch.setattr(sqlalchemy, "create_engine", mock_create_engine)

    mock_inspector = Mock()
    mock_inspector.has_table.return_value = False
    monkeypatch.setattr(
        "sqlalchemy.engine.Inspector.from_engine", Mock(return_value=mock_inspector)
    )

    monkeypatch.setattr(
        SQLTrackerStore.Base.metadata,
        "create_all",
        Mock(side_effect=Exception("Table creation failed")),
    )

    with pytest.raises(Exception):
        SQLTrackerStore(domain=domain, dialect="postgresql")

    # Verify unlock happened after lock (lock is always released)
    assert "lock" in lock_calls
    assert "unlock" in lock_calls
    assert lock_calls.index("unlock") > lock_calls.index("lock")


async def test_sql_tracker_store_get_trackers_by_user_id(
    domain: Domain, tmp_path: Path
) -> None:
    """Test SQLTrackerStore.get_trackers_by_user_id returns correct trackers."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"

    # Create trackers with user_id
    await create_tracker_with_user_id(tracker_store, "sender1", user_id)
    await create_tracker_with_user_id(
        tracker_store, "sender2", user_id, [SessionStarted(), UserUttered("hi")]
    )

    # Create tracker with different user_id
    await create_tracker_with_user_id(
        tracker_store, "sender3", "user_456", [SessionStarted(), UserUttered("hey")]
    )

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 2
    assert {t.sender_id for t in trackers} == {"sender1", "sender2"}
    for tracker in trackers:
        assert tracker.user_id == user_id


async def test_sql_tracker_store_get_trackers_by_user_id_no_matches(
    domain: Domain,
) -> None:
    """Test SQLTrackerStore returns empty list when no matches exist."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"

    # Create tracker with different user_id
    await create_tracker_with_user_id(tracker_store, "sender1", "user_456")

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 0


async def test_sql_tracker_store_get_trackers_by_user_id_filters_no_user_id(
    domain: Domain,
) -> None:
    """Test SQLTrackerStore filters out trackers without user_id."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"

    # Create tracker with user_id
    await create_tracker_with_user_id(tracker_store, "sender1", user_id)

    # Create tracker without user_id (anonymous)
    await create_tracker_with_user_id(
        tracker_store, "sender2", None, [SessionStarted(), UserUttered("hi")]
    )

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 1
    assert trackers[0].sender_id == "sender1"
    assert trackers[0].user_id == user_id


async def test_sql_tracker_store_get_trackers_by_user_id_save_sets_user_id(
    domain: Domain,
) -> None:
    """Test that save method maintains users table when user_id is set."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"
    sender_id = "sender1"

    tracker = await create_tracker_with_user_id(tracker_store, sender_id, user_id)
    await tracker_store.save(tracker)

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 1
    assert_tracker_has_user_id(trackers[0], sender_id, user_id)


async def test_sql_tracker_store_get_trackers_by_user_id_update_sets_user_id(
    domain: Domain,
) -> None:
    """Test that update method maintains users table when user_id is set."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"
    sender_id = "sender1"

    tracker = await create_tracker_with_user_id(tracker_store, sender_id, user_id)
    await tracker_store.save(tracker)

    tracker.update(UserUttered("hello again"))
    await tracker_store.update(tracker)

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 1
    assert_tracker_has_user_id(trackers[0], sender_id, user_id)


async def test_sql_tracker_store_get_trackers_by_user_id_with_limit(
    domain: Domain,
) -> None:
    """Test SQLTrackerStore.get_trackers_by_user_id respects limit parameter."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"

    # Create multiple trackers with user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 10
    )

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id, limit=5)

    # Then
    assert len(trackers) == 5
    assert_all_trackers_have_user_id(trackers, user_id)
    assert trackers == saved_trackers[:5]


async def test_sql_tracker_store_get_trackers_by_user_id_with_skip(
    domain: Domain,
) -> None:
    """Test SQLTrackerStore.get_trackers_by_user_id respects skip parameter."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"

    # Create multiple trackers with user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 10
    )

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=3)

    # Then
    assert len(trackers) == 7  # 10 total - 3 skipped
    assert_all_trackers_have_user_id(trackers, user_id)
    assert trackers == saved_trackers[3:]


async def test_sql_tracker_store_get_trackers_by_user_id_with_skip_and_limit(
    domain: Domain,
) -> None:
    """Test SQLTrackerStore.get_trackers_by_user_id respects both skip and limit."""
    # Given
    import time

    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"

    # Create trackers with small delays to ensure different timestamps
    saved_trackers = []
    for i in range(10):
        time.sleep(0.1)  # Small delay to ensure different timestamps
        tracker = await create_tracker_with_user_id(
            tracker_store,
            f"sender{i}",
            user_id,
            [SessionStarted(), UserUttered(f"hello{i}")],
        )
        saved_trackers.append(tracker)

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=2, limit=3)

    # Then
    assert len(trackers) == 3
    assert_all_trackers_have_user_id(trackers, user_id)
    assert trackers == saved_trackers[2:5]


async def test_sql_tracker_store_get_trackers_by_user_id_no_users_table(
    domain: Domain,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test SQLTrackerStore.get_trackers_by_user_id handles missing users table."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"

    # Mock inspector to return False for users table
    mock_inspector = Mock()
    mock_inspector.has_table.return_value = False
    monkeypatch.setattr(
        "sqlalchemy.engine.Inspector.from_engine",
        Mock(return_value=mock_inspector),
    )

    # When
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(trackers) == 0

        # Verify warning was logged
        logs = filter_logs(
            caplog,
            event="sql_tracker_store.get_trackers_by_user_id.no_users_table",
            log_level="warning",
        )
        assert len(logs) == 1


async def test_sql_tracker_store_delete_cleans_up_users_table(
    domain: Domain,
) -> None:
    """Test that delete method removes entry from users table."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"
    sender_id = "sender1"

    # Create tracker with user_id
    await create_tracker_with_user_id(tracker_store, sender_id, user_id)

    # Verify it's in users table
    trackers = await tracker_store.get_trackers_by_user_id(user_id)
    assert len(trackers) == 1

    # When
    await tracker_store.delete(sender_id)

    # Then
    # Verify tracker is deleted
    retrieved = await tracker_store.retrieve(sender_id)
    assert retrieved is None

    # Verify entry is removed from users table
    trackers_after_delete = await tracker_store.get_trackers_by_user_id(user_id)
    assert len(trackers_after_delete) == 0


async def test_sql_tracker_store_anonymous_user_not_in_users_table(
    domain: Domain,
) -> None:
    """Test that anonymous users (user_id=None) are not stored in users table."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    sender_id = "sender1"

    # Create tracker without user_id (anonymous)
    tracker = await create_tracker_with_user_id(tracker_store, sender_id, None)
    assert tracker.user_id is None

    # When - query users table directly
    with tracker_store.session_scope() as session:
        user_mappings = session.query(tracker_store.SQLUser).all()

    # Then - no entries for anonymous users
    assert len(user_mappings) == 0

    # Verify tracker still exists in events
    retrieved = await tracker_store.retrieve(sender_id)
    assert_tracker_has_user_id(retrieved, sender_id, None)


async def test_sql_tracker_store_retrieve_gets_user_id_from_users_table(
    domain: Domain,
) -> None:
    """Test that retrieve method fetches user_id from users table."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"
    sender_id = "sender1"

    # Create tracker with user_id and save it
    await create_tracker_with_user_id(tracker_store, sender_id, user_id)

    # Verify user_id is in users table
    assert get_user_id_from_users_table(tracker_store, sender_id) == user_id

    # When - retrieve the tracker
    retrieved_tracker = await tracker_store.retrieve(sender_id)

    # Then - tracker should have user_id set from users table
    assert_tracker_has_user_id(retrieved_tracker, sender_id, user_id)


async def test_sql_tracker_store_retrieve_full_tracker_gets_user_id_from_users_table(
    domain: Domain,
) -> None:
    """Test that retrieve_full_tracker method fetches user_id from users table."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_456"
    sender_id = "sender2"

    # Create tracker with user_id and save it
    await create_tracker_with_user_id(
        tracker_store,
        sender_id,
        user_id,
        [
            SessionStarted(),
            UserUttered("hello"),
            BotUttered("hi"),
            UserUttered("how are you"),
        ],
    )

    # When - retrieve full tracker
    retrieved_tracker = await tracker_store.retrieve_full_tracker(sender_id)

    # Then - tracker should have user_id set from users table
    assert_tracker_has_user_id(retrieved_tracker, sender_id, user_id)


async def test_sql_tracker_store_retrieve_handles_missing_user_id_in_users_table(
    domain: Domain,
) -> None:
    """Test that retrieve works correctly when user_id is missing (anonymous user)."""
    # Given
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    sender_id = "sender3"

    # Create tracker without user_id (anonymous) and save it
    tracker = await create_tracker_with_user_id(tracker_store, sender_id, None)
    assert tracker.user_id is None

    # Verify no entry in users table
    assert get_user_id_from_users_table(tracker_store, sender_id) is None

    # When - retrieve the tracker
    retrieved_tracker = await tracker_store.retrieve(sender_id)

    # Then - tracker should be retrieved successfully but user_id should be None
    assert_tracker_has_user_id(retrieved_tracker, sender_id, None)


# Backward compatibility tests for conversation_started_timestamp
@pytest.mark.asyncio
async def test_sql_old_tracker_gets_timestamp_on_save(domain: Domain) -> None:
    """Test that old tracker without conversation_started_timestamp gets it on save."""
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    await old_tracker_gets_timestamp_on_save(tracker_store, domain=domain)


@pytest.mark.asyncio
async def test_sql_old_tracker_gets_timestamp_on_update(domain: Domain) -> None:
    """Test that old tracker without conversation_started_timestamp gets it
    on update."""
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    await old_tracker_gets_timestamp_on_update(tracker_store, domain=domain)


# Sorting consistency tests
@pytest.mark.asyncio
async def test_sql_sorting_by_sender_id_when_timestamps_identical(
    domain: Domain,
) -> None:
    """Test that trackers with identical timestamps are sorted by sender_id."""
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"
    timestamp = 1234567890.0
    sender_ids = ["sender_c", "sender_a", "sender_b"]

    # Create trackers with same timestamp
    await create_trackers_with_same_timestamp(
        tracker_store, user_id, timestamp, sender_ids, domain=domain
    )

    # Retrieve and verify sorting
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Should be sorted by sender_id when timestamps are identical
    assert len(trackers) == 3
    assert trackers[0].sender_id == "sender_a"
    assert trackers[1].sender_id == "sender_b"
    assert trackers[2].sender_id == "sender_c"

    # All should have same timestamp
    for tracker in trackers:
        assert tracker.conversation_started_timestamp == timestamp


@pytest.mark.asyncio
async def test_sql_pagination_very_large_skip(domain: Domain) -> None:
    """Test that very large skip values are handled gracefully."""
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"

    # Create some trackers
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 5, domain=domain
    )

    # Very large skip should return empty list
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=1000000)

    assert len(trackers) == 0


@pytest.mark.asyncio
async def test_sql_pagination_very_large_limit(domain: Domain) -> None:
    """Test that very large limit values are handled gracefully."""
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"

    # Create some trackers
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 5, domain=domain
    )

    # Very large limit should return all items (up to available)
    trackers = await tracker_store.get_trackers_by_user_id(user_id, limit=1000000)

    assert len(trackers) == 5


@pytest.mark.asyncio
async def test_sql_negative_skip_and_limit_ignored(domain: Domain) -> None:
    """Test that both negative skip and limit values are ignored."""
    tracker_store = SQLTrackerStore(domain, host="sqlite:///")
    user_id = "user_123"

    # Create some trackers
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 5, domain=domain
    )

    # When: Retrieve with both negative skip and limit
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=-3, limit=-2)

    # Then: Should return all trackers (both negative values ignored)
    assert len(trackers) == 5
    assert_all_trackers_have_user_id(trackers, user_id)
