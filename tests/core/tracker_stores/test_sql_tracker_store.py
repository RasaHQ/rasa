import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager, Dict, List, Optional, Union
from unittest.mock import Mock

import pytest
import sqlalchemy
from pytest import CaptureFixture, LogCaptureFixture, MonkeyPatch
from sqlalchemy import URL
from sqlalchemy.dialects.oracle.base import OracleDialect
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.dialects.sqlite.base import SQLiteDialect
from structlog.testing import capture_logs

from rasa.core.agent import Agent
from rasa.core.constants import POSTGRESQL_SCHEMA
from rasa.core.tracker_stores.sql_tracker_store import (
    POSTGRESQL_DEFAULT_MAX_OVERFLOW,
    POSTGRESQL_DEFAULT_POOL_SIZE,
    SQLTrackerStore,
    create_engine_kwargs,
    ensure_schema_exists,
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
from tests.core.tracker_stores.conftest import (
    _saved_tracker_with_multiple_session_starts,
    create_tracker_with_partially_saved_events,
    prepare_token_serialisation,
)
from tests.utilities import filter_logs


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
    from sqlalchemy.engine.base import Engine

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
    domain: Domain, response_selector_agent: Agent
):
    tracker_store = SQLTrackerStore(domain, **{"host": "sqlite:///"})
    prepare_token_serialisation(tracker_store, response_selector_agent, "sql")


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


async def test_sql_get_or_create_full_tracker_with_action_listen() -> None:
    tracker_store = SQLTrackerStore(Domain.empty())
    sender_id = uuid.uuid4().hex
    tracker = await tracker_store.get_or_create_full_tracker(
        sender_id=sender_id, append_action_listen=True
    )
    assert tracker.sender_id == sender_id
    assert tracker.events == deque([ActionExecuted(ACTION_LISTEN_NAME)])


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
