from rasa_core import utils
from rasa_core.broker import PikaProducer
from tests.conftest import DEFAULT_ENDPOINTS_FILE

EVENT_BROKER_ENDPOINT_FILE = 'data/test_endpoints/event_broker_endpoint.yml'


def test_broker_from_config():
    cfg = utils.read_endpoint_config(EVENT_BROKER_ENDPOINT_FILE,
                                     "event_broker")
    actual = PikaProducer.from_endpoint_config(cfg)

    expected = PikaProducer("localhost", "username", "password", "queue")

    assert actual.host == expected.host
    assert actual.credentials.username == expected.credentials.username
    assert actual.queue == expected.queue


def test_broker_not_in_config():
    cfg = utils.read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "event_broker")

    actual = PikaProducer.from_endpoint_config(cfg)

    assert actual is None
