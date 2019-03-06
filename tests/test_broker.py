import json

from rasa_core import broker, utils
from rasa_core.broker import FileProducer, PikaProducer
from rasa_core.events import Event, Restarted, SlotSet, UserUttered
from rasa_core.utils import EndpointConfig
from tests.conftest import DEFAULT_ENDPOINTS_FILE

TEST_EVENTS = [
    UserUttered("/greet", {"name": "greet", "confidence": 1.0}, []),
    SlotSet("name", "rasa"),
    Restarted()]


def test_pika_broker_from_config():
    cfg = utils.read_endpoint_config('data/test_endpoints/event_brokers/'
                                     'pika_endpoint.yml',
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
    cfg = utils.read_endpoint_config("data/test_endpoints/event_brokers/"
                                     "file_endpoint.yml",
                                     "event_broker")
    actual = broker.from_endpoint_config(cfg)

    assert isinstance(actual, FileProducer)
    assert actual.path == "rasa_event.log"


def test_file_broker_logs_to_file(tmpdir):
    fname = tmpdir.join("events.log").strpath

    actual = broker.from_endpoint_config(EndpointConfig(**{"type": "file",
                                                           "path": fname}))

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

    event_with_newline = UserUttered("hello \n there")

    actual.publish(event_with_newline.as_dict())

    # reading the events from the file one event per line
    recovered = []
    with open(fname, "r") as f:
        for l in f:
            recovered.append(Event.from_parameters(json.loads(l)))

    assert recovered == [event_with_newline]


def test_load_custom_broker_name():
    config = EndpointConfig(**{"type": "rasa_core.broker.FileProducer"})
    assert broker.from_endpoint_config(config)


def test_load_non_existent_custom_broker_name():
    config = EndpointConfig(**{"type": "rasa_core.broker.MyProducer"})
    assert broker.from_endpoint_config(config) is None
