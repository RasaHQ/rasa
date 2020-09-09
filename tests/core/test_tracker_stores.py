import logging
from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy
import uuid

from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from moto import mock_dynamodb2
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.dialects.sqlite.base import SQLiteDialect
from sqlalchemy.dialects.oracle.base import OracleDialect
from sqlalchemy.engine.url import URL
from typing import Tuple, Text, Type, Dict, List, Union, Optional, ContextManager
from unittest.mock import Mock

import rasa.core.tracker_store
from rasa.core.actions.action import ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME
from rasa.core.channels.channel import UserMessage
from rasa.core.constants import POSTGRESQL_SCHEMA
from rasa.core.domain import Domain
from rasa.core.events import (
    SlotSet,
    ActionExecuted,
    Restarted,
    UserUttered,
    SessionStarted,
    BotUttered,
    Event,
)
from rasa.core.tracker_store import (
    TrackerStore,
    InMemoryTrackerStore,
    RedisTrackerStore,
    SQLTrackerStore,
    DynamoTrackerStore,
    FailSafeTrackerStore,
)
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from tests.core.conftest import DEFAULT_ENDPOINTS_FILE, MockedMongoTrackerStore

domain = Domain.load("data/test_domains/default.yml")


def get_or_create_tracker_store(store: TrackerStore) -> None:
    slot_key = "location"
    slot_val = "Easter Island"

    tracker = store.get_or_create_tracker(UserMessage.DEFAULT_SENDER_ID)
    ev = SlotSet(slot_key, slot_val)
    tracker.update(ev)
    assert tracker.get_slot(slot_key) == slot_val

    store.save(tracker)

    again = store.get_or_create_tracker(UserMessage.DEFAULT_SENDER_ID)
    assert again.get_slot(slot_key) == slot_val


def test_get_or_create():
    get_or_create_tracker_store(InMemoryTrackerStore(domain))


# noinspection PyPep8Naming
@mock_dynamodb2
def test_dynamo_get_or_create():
    get_or_create_tracker_store(DynamoTrackerStore(domain))


@mock_dynamodb2
def test_dynamo_tracker_floats():
    conversation_id = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(domain)
    tracker = tracker_store.get_or_create_tracker(
        conversation_id, append_action_listen=False
    )

    # save `slot` event with known `float`-type timestamp
    timestamp = 13423.23434623
    tracker.update(SlotSet("key", "val", timestamp=timestamp))
    tracker_store.save(tracker)

    # retrieve tracker and the event timestamp is retrieved as a `float`
    tracker = tracker_store.get_or_create_tracker(conversation_id)
    retrieved_timestamp = tracker.events[0].timestamp
    assert isinstance(retrieved_timestamp, float)
    assert retrieved_timestamp == timestamp


def test_restart_after_retrieval_from_tracker_store(default_domain: Domain):
    store = InMemoryTrackerStore(default_domain)
    tr = store.get_or_create_tracker("myuser")
    synth = [ActionExecuted("action_listen") for _ in range(4)]

    for e in synth:
        tr.update(e)

    tr.update(Restarted())
    latest_restart = tr.idx_after_latest_restart()

    store.save(tr)
    tr2 = store.retrieve("myuser")
    latest_restart_after_loading = tr2.idx_after_latest_restart()
    assert latest_restart == latest_restart_after_loading


def test_tracker_store_remembers_max_history(default_domain: Domain):
    store = InMemoryTrackerStore(default_domain)
    tr = store.get_or_create_tracker("myuser", max_event_history=42)
    tr.update(Restarted())

    store.save(tr)
    tr2 = store.retrieve("myuser")
    assert tr._max_event_history == tr2._max_event_history == 42


def test_tracker_store_endpoint_config_loading():
    cfg = read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "tracker_store")

    assert cfg == EndpointConfig.from_dict(
        {
            "type": "redis",
            "url": "localhost",
            "port": 6379,
            "db": 0,
            "password": "password",
            "timeout": 30000,
        }
    )


def test_create_tracker_store_from_endpoint_config(default_domain: Domain):
    store = read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "tracker_store")
    tracker_store = RedisTrackerStore(
        domain=default_domain,
        host="localhost",
        port=6379,
        db=0,
        password="password",
        record_exp=3000,
    )

    assert isinstance(tracker_store, type(TrackerStore.create(store, default_domain)))


def test_exception_tracker_store_from_endpoint_config(
    default_domain: Domain, monkeypatch: MonkeyPatch
):
    """Check if tracker store properly handles exceptions.

    If we can not create a tracker store by instantiating the
    expected type (e.g. due to an exception) we should fallback to
    the default `InMemoryTrackerStore`."""

    store = read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "tracker_store")
    mock = Mock(side_effect=Exception("test exception"))
    monkeypatch.setattr(rasa.core.tracker_store, "RedisTrackerStore", mock)

    with pytest.raises(Exception) as e:
        TrackerStore.create(store, default_domain)

    assert "test exception" in str(e.value)


class URLExampleTrackerStore(RedisTrackerStore):
    def __init__(self, domain, url, port, db, password, record_exp, event_broker=None):
        super().__init__(
            domain,
            event_broker=event_broker,
            host=url,
            port=port,
            db=db,
            password=password,
            record_exp=record_exp,
        )


class HostExampleTrackerStore(RedisTrackerStore):
    pass


def test_tracker_store_deprecated_url_argument_from_string(default_domain: Domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")
    store_config.type = "tests.core.test_tracker_stores.URLExampleTrackerStore"

    with pytest.raises(Exception):
        TrackerStore.create(store_config, default_domain)


def test_tracker_store_with_host_argument_from_string(default_domain: Domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")
    store_config.type = "tests.core.test_tracker_stores.HostExampleTrackerStore"

    with pytest.warns(None) as record:
        tracker_store = TrackerStore.create(store_config, default_domain)

    assert len(record) == 0

    assert isinstance(tracker_store, HostExampleTrackerStore)


def test_tracker_store_from_invalid_module(default_domain: Domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")
    store_config.type = "a.module.which.cannot.be.found"

    with pytest.warns(UserWarning):
        tracker_store = TrackerStore.create(store_config, default_domain)

    assert isinstance(tracker_store, InMemoryTrackerStore)


def test_tracker_store_from_invalid_string(default_domain: Domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")
    store_config.type = "any string"

    with pytest.warns(UserWarning):
        tracker_store = TrackerStore.create(store_config, default_domain)

    assert isinstance(tracker_store, InMemoryTrackerStore)


def _tracker_store_and_tracker_with_slot_set() -> Tuple[
    InMemoryTrackerStore, DialogueStateTracker
]:
    # returns an InMemoryTrackerStore containing a tracker with a slot set

    slot_key = "cuisine"
    slot_val = "French"

    store = InMemoryTrackerStore(domain)
    tracker = store.get_or_create_tracker(UserMessage.DEFAULT_SENDER_ID)
    ev = SlotSet(slot_key, slot_val)
    tracker.update(ev)

    return store, tracker


def test_tracker_serialisation():
    store, tracker = _tracker_store_and_tracker_with_slot_set()
    serialised = store.serialise_tracker(tracker)

    assert tracker == store.deserialise_tracker(
        UserMessage.DEFAULT_SENDER_ID, serialised
    )


def test_deprecated_pickle_deserialisation():
    def pickle_serialise_tracker(_tracker):
        # mocked version of TrackerStore.serialise_tracker() that uses
        # the deprecated pickle serialisation
        import pickle

        dialogue = _tracker.as_dialogue()

        return pickle.dumps(dialogue)

    store, tracker = _tracker_store_and_tracker_with_slot_set()

    serialised = pickle_serialise_tracker(tracker)

    # deprecation warning should be emitted

    with pytest.warns(FutureWarning) as record:
        assert tracker == store.deserialise_tracker(
            UserMessage.DEFAULT_SENDER_ID, serialised
        )
    assert len(record) == 1
    assert (
        "Deserialisation of pickled trackers is deprecated" in record[0].message.args[0]
    )


@pytest.mark.parametrize(
    "full_url",
    [
        "postgresql://localhost",
        "postgresql://localhost:5432",
        "postgresql://user:secret@localhost",
        "sqlite:///",
    ],
)
def test_get_db_url_with_fully_specified_url(full_url: Text):
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
    password = "some-password"

    with caplog.at_level(logging.DEBUG):
        _ = SQLTrackerStore(None, dialect, host, port, db, username, password)

    # the URL in the logs does not contain the password
    assert password not in caplog.text

    # instead the password is displayed as '***'
    assert f"postgresql://{username}:***@{host}:{port}/{db}" in caplog.text


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
    connection_url = "postgresql://user:pw@:5123/login-db?"
    assert any(
        str(url) == connection_url + "&".join(permutation)
        for permutation in (
            itertools.permutations(("another=query", "driver=my-driver"))
        )
    )


def test_fail_safe_tracker_store_if_no_errors():
    mocked_tracker_store = Mock()

    tracker_store = FailSafeTrackerStore(mocked_tracker_store, None)

    # test save
    mocked_tracker_store.save = Mock()
    tracker_store.save(None)
    mocked_tracker_store.save.assert_called_once()

    # test retrieve
    expected = [1]
    mocked_tracker_store.retrieve = Mock(return_value=expected)
    sender_id = "10"
    assert tracker_store.retrieve(sender_id) == expected
    mocked_tracker_store.retrieve.assert_called_once_with(sender_id)

    # test keys
    expected = ["sender 1", "sender 2"]
    mocked_tracker_store.keys = Mock(return_value=expected)
    assert tracker_store.keys() == expected
    mocked_tracker_store.keys.assert_called_once()


def test_fail_safe_tracker_store_with_save_error():
    mocked_tracker_store = Mock()
    mocked_tracker_store.save = Mock(side_effect=Exception())

    fallback_tracker_store = Mock()
    fallback_tracker_store.save = Mock()

    on_error_callback = Mock()

    tracker_store = FailSafeTrackerStore(
        mocked_tracker_store, on_error_callback, fallback_tracker_store
    )
    tracker_store.save(None)

    fallback_tracker_store.save.assert_called_once()
    on_error_callback.assert_called_once()


def test_fail_safe_tracker_store_with_keys_error():
    mocked_tracker_store = Mock()
    mocked_tracker_store.keys = Mock(side_effect=Exception())

    on_error_callback = Mock()

    tracker_store = FailSafeTrackerStore(mocked_tracker_store, on_error_callback)
    assert tracker_store.keys() == []
    on_error_callback.assert_called_once()


def test_fail_safe_tracker_store_with_retrieve_error():
    mocked_tracker_store = Mock()
    mocked_tracker_store.retrieve = Mock(side_effect=Exception())

    fallback_tracker_store = Mock()
    on_error_callback = Mock()

    tracker_store = FailSafeTrackerStore(
        mocked_tracker_store, on_error_callback, fallback_tracker_store
    )

    assert tracker_store.retrieve("sender_id") is None
    on_error_callback.assert_called_once()


def test_set_fail_safe_tracker_store_domain(default_domain: Domain):
    tracker_store = InMemoryTrackerStore(domain)
    fallback_tracker_store = InMemoryTrackerStore(None)
    failsafe_store = FailSafeTrackerStore(tracker_store, None, fallback_tracker_store)

    failsafe_store.domain = default_domain
    assert failsafe_store.domain is default_domain
    assert tracker_store.domain is failsafe_store.domain
    assert fallback_tracker_store.domain is failsafe_store.domain


def create_tracker_with_partially_saved_events(
    tracker_store: TrackerStore,
) -> Tuple[List[Event], DialogueStateTracker]:
    # creates a tracker with two events and saved it to the tracker store
    # following that, it adds three more events that are not saved to the tracker store
    sender_id = uuid.uuid4().hex

    # create tracker with two events and save it
    events = [UserUttered("hello"), BotUttered("what")]
    tracker = DialogueStateTracker.from_events(sender_id, events)
    tracker_store.save(tracker)

    # add more events to the tracker, do not yet save it
    events = [ActionExecuted(ACTION_LISTEN_NAME), UserUttered("123"), BotUttered("yes")]
    for event in events:
        tracker.update(event)

    return events, tracker


def _saved_tracker_with_multiple_session_starts(
    tracker_store: TrackerStore, sender_id: Text
) -> DialogueStateTracker:
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("hi"),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
        ],
    )

    tracker_store.save(tracker)
    return tracker_store.retrieve(sender_id)


def test_mongo_additional_events(default_domain: Domain):
    tracker_store = MockedMongoTrackerStore(default_domain)
    events, tracker = create_tracker_with_partially_saved_events(tracker_store)

    # make sure only new events are returned
    # noinspection PyProtectedMember
    assert list(tracker_store._additional_events(tracker)) == events


def test_mongo_additional_events_with_session_start(default_domain: Domain):
    sender = "test_mongo_additional_events_with_session_start"
    tracker_store = MockedMongoTrackerStore(default_domain)
    tracker = _saved_tracker_with_multiple_session_starts(tracker_store, sender)

    tracker.update(UserUttered("hi2"))

    # noinspection PyProtectedMember
    additional_events = list(tracker_store._additional_events(tracker))

    assert len(additional_events) == 1
    assert isinstance(additional_events[0], UserUttered)


# we cannot parametrise over this and the previous test due to the different ways of
# calling _additional_events()
def test_sql_additional_events(default_domain: Domain):
    tracker_store = SQLTrackerStore(default_domain)
    additional_events, tracker = create_tracker_with_partially_saved_events(
        tracker_store
    )

    # make sure only new events are returned
    with tracker_store.session_scope() as session:
        # noinspection PyProtectedMember
        assert (
            list(tracker_store._additional_events(session, tracker))
            == additional_events
        )


def test_sql_additional_events_with_session_start(default_domain: Domain):
    sender = "test_sql_additional_events_with_session_start"
    tracker_store = SQLTrackerStore(default_domain)
    tracker = _saved_tracker_with_multiple_session_starts(tracker_store, sender)

    tracker.update(UserUttered("hi2"), default_domain)

    # make sure only new events are returned
    with tracker_store.session_scope() as session:
        # noinspection PyProtectedMember
        additional_events = list(tracker_store._additional_events(session, tracker))
        assert len(additional_events) == 1
        assert isinstance(additional_events[0], UserUttered)


@pytest.mark.parametrize(
    "tracker_store_type,tracker_store_kwargs",
    [(MockedMongoTrackerStore, {}), (SQLTrackerStore, {"host": "sqlite:///"})],
)
def test_tracker_store_retrieve_with_session_started_events(
    tracker_store_type: Type[TrackerStore],
    tracker_store_kwargs: Dict,
    default_domain: Domain,
):
    tracker_store = tracker_store_type(default_domain, **tracker_store_kwargs)
    events = [
        UserUttered("Hola", {"name": "greet"}, timestamp=1),
        BotUttered("Hi", timestamp=2),
        SessionStarted(timestamp=3),
        UserUttered("Ciao", {"name": "greet"}, timestamp=4),
    ]
    sender_id = "test_sql_tracker_store_with_session_events"
    tracker = DialogueStateTracker.from_events(sender_id, events)
    tracker_store.save(tracker)

    # Save other tracker to ensure that we don't run into problems with other senders
    other_tracker = DialogueStateTracker.from_events("other-sender", [SessionStarted()])
    tracker_store.save(other_tracker)

    # Retrieve tracker with events since latest SessionStarted
    tracker = tracker_store.retrieve(sender_id)

    assert len(tracker.events) == 2
    assert all((event == tracker.events[i] for i, event in enumerate(events[2:])))


@pytest.mark.parametrize(
    "tracker_store_type,tracker_store_kwargs",
    [(MockedMongoTrackerStore, {}), (SQLTrackerStore, {"host": "sqlite:///"})],
)
def test_tracker_store_retrieve_without_session_started_events(
    tracker_store_type: Type[TrackerStore],
    tracker_store_kwargs: Dict,
    default_domain: Domain,
):
    tracker_store = tracker_store_type(default_domain, **tracker_store_kwargs)

    # Create tracker with a SessionStarted event
    events = [
        UserUttered("Hola", {"name": "greet"}),
        BotUttered("Hi"),
        UserUttered("Ciao", {"name": "greet"}),
        BotUttered("Hi2"),
    ]

    sender_id = "test_sql_tracker_store_retrieve_without_session_started_events"
    tracker = DialogueStateTracker.from_events(sender_id, events)
    tracker_store.save(tracker)

    # Save other tracker to ensure that we don't run into problems with other senders
    other_tracker = DialogueStateTracker.from_events("other-sender", [SessionStarted()])
    tracker_store.save(other_tracker)

    tracker = tracker_store.retrieve(sender_id)

    assert len(tracker.events) == 4
    assert all(event == tracker.events[i] for i, event in enumerate(events))


@pytest.mark.parametrize(
    "tracker_store_type,tracker_store_kwargs",
    [
        (MockedMongoTrackerStore, {}),
        (SQLTrackerStore, {"host": "sqlite:///"}),
        (InMemoryTrackerStore, {}),
    ],
)
def test_tracker_store_retrieve_with_events_from_previous_sessions(
    tracker_store_type: Type[TrackerStore], tracker_store_kwargs: Dict
):
    tracker_store = tracker_store_type(Domain.empty(), **tracker_store_kwargs)
    tracker_store.load_events_from_previous_conversation_sessions = True

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
    tracker_store.save(tracker)

    actual = tracker_store.retrieve(conversation_id)

    assert len(actual.events) == len(tracker.events)


def test_session_scope_error(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture, default_domain: Domain
):
    tracker_store = SQLTrackerStore(default_domain)
    tracker_store.sessionmaker = Mock()

    requested_schema = uuid.uuid4().hex

    # `ensure_schema_exists()` raises `ValueError`
    mocked_ensure_schema_exists = Mock(side_effect=ValueError(requested_schema))
    monkeypatch.setattr(
        rasa.core.tracker_store, "ensure_schema_exists", mocked_ensure_schema_exists
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
        (URL(PGDialect.name), True),
        (URL(SQLiteDialect.name), False),
    ],
)
def test_is_postgres_url(url: Union[Text, URL], is_postgres_url: bool):
    assert rasa.core.tracker_store.is_postgresql_url(url) == is_postgres_url


def set_or_delete_postgresql_schema_env_var(
    monkeypatch: MonkeyPatch, value: Optional[Text]
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
                "pool_size": rasa.core.tracker_store.POSTGRESQL_DEFAULT_POOL_SIZE,
                "max_overflow": rasa.core.tracker_store.POSTGRESQL_DEFAULT_MAX_OVERFLOW,
            },
        ),
        # postgres with schema
        (
            f"{PGDialect.name}://admin:pw@localhost:5432/rasa",
            "schema1",
            {
                "connect_args": {"options": "-csearch_path=schema1"},
                "pool_size": rasa.core.tracker_store.POSTGRESQL_DEFAULT_POOL_SIZE,
                "max_overflow": rasa.core.tracker_store.POSTGRESQL_DEFAULT_MAX_OVERFLOW,
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
    url: Union[Text, URL],
    schema_env: Optional[Text],
    kwargs: Dict[Text, Dict[Text, Union[Text, int]]],
):
    set_or_delete_postgresql_schema_env_var(monkeypatch, schema_env)

    assert rasa.core.tracker_store.create_engine_kwargs(url) == kwargs


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
    schema_env: Optional[Text],
    schema_exists: bool,
    raises_context: ContextManager,
):
    set_or_delete_postgresql_schema_env_var(monkeypatch, schema_env)
    monkeypatch.setattr(
        rasa.core.tracker_store, "is_postgresql_url", lambda _: is_postgres
    )
    monkeypatch.setattr(sqlalchemy, "exists", Mock())

    # mock the `session.query().scalar()` query which returns whether the schema
    # exists in the db
    scalar = Mock(return_value=schema_exists)
    query = Mock(scalar=scalar)
    session = Mock()
    session.query = Mock(return_value=query)

    with raises_context:
        rasa.core.tracker_store.ensure_schema_exists(session)


def test_current_state_without_events(default_domain: Domain):
    tracker_store = MockedMongoTrackerStore(default_domain)

    # insert some events
    events = [
        UserUttered("Hola", {"name": "greet"}),
        BotUttered("Hi"),
        UserUttered("Ciao", {"name": "greet"}),
        BotUttered("Hi2"),
    ]

    sender_id = "test_mongo_tracker_store_current_state_without_events"
    tracker = DialogueStateTracker.from_events(sender_id, events)

    # get current state without events
    # noinspection PyProtectedMember
    state = tracker_store._current_tracker_state_without_events(tracker)

    # `events` key should not be in there
    assert state and "events" not in state
