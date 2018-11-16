from rasa_core import utils
from rasa_core.channels import UserMessage
from rasa_core.domain import Domain
from rasa_core.events import SlotSet, ActionExecuted, Restarted
from rasa_core.tracker_store import (
    TrackerStore,
    InMemoryTrackerStore,
    RedisTrackerStore)
from rasa_core.utils import EndpointConfig
from tests.conftest import DEFAULT_ENDPOINTS_FILE

domain = Domain.load("data/test_domains/default.yml")


def test_get_or_create():
    slot_key = 'location'
    slot_val = 'Easter Island'
    store = InMemoryTrackerStore(domain)

    tracker = store.get_or_create_tracker(UserMessage.DEFAULT_SENDER_ID)
    ev = SlotSet(slot_key, slot_val)
    tracker.update(ev)
    assert tracker.get_slot(slot_key) == slot_val

    store.save(tracker)

    again = store.get_or_create_tracker(UserMessage.DEFAULT_SENDER_ID)
    assert again.get_slot(slot_key) == slot_val


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


def test_tracker_store_endpoint_config_loading():
    cfg = utils.read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "tracker_store")

    assert cfg == EndpointConfig.from_dict({
        "store_type": "redis",
        "url": "localhost",
        "port": 6379,
        "db": 0,
        "password": "password",
        "timeout": 30000
    })


def test_find_tracker_store(default_domain):
    store = utils.read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "tracker_store")
    tracker_store = RedisTrackerStore(domain=default_domain,
                                      host="localhost",
                                      port=6379,
                                      db=0,
                                      password="password",
                                      record_exp=3000)

    assert isinstance(tracker_store,
                      type(
                          TrackerStore.find_tracker_store(default_domain, store)
                      ))


class ExampleTrackerStore(RedisTrackerStore):
    def __init__(self, domain, url, port, db, password, record_exp):
        super(ExampleTrackerStore, self).__init__(domain,
                                                  host=url,
                                                  port=port,
                                                  db=db,
                                                  password=password,
                                                  record_exp=record_exp)


def test_tracker_store_from_string(default_domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = utils.read_endpoint_config(endpoints_path, "tracker_store")

    tracker_store = TrackerStore.find_tracker_store(default_domain,
                                                    store_config)

    assert isinstance(tracker_store, ExampleTrackerStore)


def test_tracker_store_from_invalid_module(default_domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = utils.read_endpoint_config(endpoints_path, "tracker_store")
    store_config.store_type = "a.module.which.cannot.be.found"

    tracker_store = TrackerStore.find_tracker_store(default_domain,
                                                    store_config)

    assert isinstance(tracker_store, InMemoryTrackerStore)


def test_tracker_store_from_invalid_string(default_domain):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = utils.read_endpoint_config(endpoints_path, "tracker_store")
    store_config.store_type = "any string"

    tracker_store = TrackerStore.find_tracker_store(default_domain,
                                                    store_config)

    assert isinstance(tracker_store, InMemoryTrackerStore)
