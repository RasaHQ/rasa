import logging
import tempfile
from typing import Tuple

import pytest
from _pytest.logging import LogCaptureFixture
from moto import mock_dynamodb2

from rasa.core.channels.channel import UserMessage
from rasa.core.domain import Domain
from rasa.core.events import SlotSet, ActionExecuted, Restarted
from rasa.core.tracker_store import (
    TrackerStore,
    InMemoryTrackerStore,
    RedisTrackerStore,
    SQLTrackerStore,
    DynamoTrackerStore,
)
import rasa.core.tracker_store
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from tests.conftest import assert_log_emitted
from tests.core.conftest import DEFAULT_ENDPOINTS_FILE

domain = Domain.load("data/test_domains/default.yml")


def get_or_create_tracker_store(store: "TrackerStore"):
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


def test_restart_after_retrieval_from_tracker_store(default_domain):
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


def test_tracker_store_remembers_max_history(default_domain):
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


def test_find_tracker_store(default_domain):
    store = read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "tracker_store")
    tracker_store = RedisTrackerStore(
        domain=default_domain,
        host="localhost",
        port=6379,
        db=0,
        password="password",
        record_exp=3000,
    )

    assert isinstance(
        tracker_store, type(TrackerStore.find_tracker_store(default_domain, store))
    )


class ExampleTrackerStore(RedisTrackerStore):
    def __init__(self, domain, url, port, db, password, record_exp, event_broker=None):
        super(ExampleTrackerStore, self).__init__(
            domain,
            event_broker=event_broker,
            host=url,
            port=port,
            db=db,
            password=password,
            record_exp=record_exp,
        )


def test_tracker_store_from_string(default_domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")

    tracker_store = TrackerStore.find_tracker_store(default_domain, store_config)

    assert isinstance(tracker_store, ExampleTrackerStore)


def test_tracker_store_from_invalid_module(default_domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")
    store_config.type = "a.module.which.cannot.be.found"

    tracker_store = TrackerStore.find_tracker_store(default_domain, store_config)

    assert isinstance(tracker_store, InMemoryTrackerStore)


def test_tracker_store_from_invalid_string(default_domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")
    store_config.type = "any string"

    tracker_store = TrackerStore.find_tracker_store(default_domain, store_config)

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


def test_deprecated_pickle_deserialisation(caplog: LogCaptureFixture):
    def pickle_serialise_tracker(_tracker):
        # mocked version of TrackerStore.serialise_tracker() that uses
        # the deprecated pickle serialisation
        import pickle

        dialogue = _tracker.as_dialogue()

        return pickle.dumps(dialogue)

    store, tracker = _tracker_store_and_tracker_with_slot_set()

    serialised = pickle_serialise_tracker(tracker)

    # deprecation warning should be emitted
    with assert_log_emitted(
        caplog, rasa.core.tracker_store.logger.name, logging.WARNING, "DEPRECATION"
    ):
        assert tracker == store.deserialise_tracker(
            UserMessage.DEFAULT_SENDER_ID, serialised
        )


@pytest.mark.parametrize(
    "full_url",
    [
        "postgresql://localhost",
        "postgresql://localhost:5432",
        "postgresql://user:secret@localhost",
    ],
)
def test_get_db_url_with_fully_specified_url(full_url):
    assert SQLTrackerStore.get_db_url(host=full_url) == full_url


def test_get_db_url_with_port_in_host():
    host = "localhost:1234"
    dialect = "postgresql"
    db = "mydb"

    expected = "{}://{}/{}".format(dialect, host, db)

    assert (
        str(SQLTrackerStore.get_db_url(dialect="postgresql", host=host, db=db))
        == expected
    )


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


def test_db_url_with_query_from_endpoint_config():
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

    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.yml") as f:
        f.write(endpoint_config)
        f.flush()
        store_config = read_endpoint_config(f.name, "tracker_store")

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
