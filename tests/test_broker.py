import json

from rasa_core import utils, broker
from rasa_core.broker import PikaProducer, EventChannel, FileProducer
from rasa_core.events import UserUttered, SlotSet, Restarted, Event
from rasa_core.utils import EndpointConfig
from tests.conftest import DEFAULT_ENDPOINTS_FILE

EVENT_BROKER_ENDPOINT_FILE = 'data/test_endpoints/event_brokers/' \
                             'pika_endpoint.yml'

TEST_EVENTS =  [
    UserUttered("/greet", {"name": "greet", "confidence": 1.0}, []),
    SlotSet("name", "rasa"),
    Restarted()]


def test_pika_broker_from_config():
    cfg = utils.read_endpoint_config(EVENT_BROKER_ENDPOINT_FILE,
                                     "event_broker")
    actual = broker.from_endpoint_config(cfg)

    assert isinstance(actual, PikaProducer)
    assert actual.host == "localhost"
    assert actual.credentials.username == "username"
    assert actual.queue == "queue"


def test_no_broker_in_config():
    cfg = utils.read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "event_broker")

    actual = broker.from_endpoint_config(cfg)

    assert actual is None


def test_file_broker_from_config():
    cfg = utils.read_endpoint_config('data/test_endpoints/event_brokers/'
                                     'file_endpoint.yml',
                                     "event_broker")
    actual = broker.from_endpoint_config(cfg)

    assert isinstance(actual, FileProducer)
    assert actual.path == "rasa_event.log"


def test_file_broker_logs_to_file(tmpdir):
    fname = tmpdir.join("events.log").strpath

    actual = broker.from_endpoint_config(EndpointConfig(**{"type": "file",
                                                           "path": fname}))

    assert isinstance(actual, FileProducer)

    for e in TEST_EVENTS:
        actual.publish(e.as_dict())

    # reading the events from the file one event per line
    recovered = []
    with open(fname, "r") as f:
        for l in f:
            recovered.append(Event.from_parameters(json.loads(l)))

    assert recovered == TEST_EVENTS


def test_file_broker_properly_logs_newlines(tmpdir):
    fname = tmpdir.join("events.log").strpath

    actual = broker.from_endpoint_config(EndpointConfig(**{"type": "file",
                                                           "path": fname}))

    assert isinstance(actual, FileProducer)

    event_with_newline = UserUttered("hello \n there")

    actual.publish(event_with_newline.as_dict())

    # reading the events from the file one event per line
    recovered = []
    with open(fname, "r") as f:
        for l in f:
            recovered.append(Event.from_parameters(json.loads(l)))

    assert recovered == [event_with_newline]
