from rasa_core import utils
from rasa_core.event_brokers.pika_producer import PikaProducer
from rasa_core.event_brokers.kafka_producer import KafkaProducer
from tests.conftest import DEFAULT_ENDPOINTS_FILE

PIKA_EVENT_BROKER_ENDPOINT_FILE = 'data/test_endpoints/event_brokers/pika_event_broker_endpoint.yml'
KAFKA_EVENT_BROKER_ENDPOINT_FILE = 'data/test_endpoints/event_brokers/kafka_event_broker_endpoint.yml'


def test_pika_broker_from_config():
    cfg = utils.read_endpoint_config(PIKA_EVENT_BROKER_ENDPOINT_FILE,
                                     "event_broker")
    actual = PikaProducer.from_endpoint_config(cfg)

    expected = PikaProducer("localhost", "username", "password", "queue")

    assert actual.host == expected.host
    assert actual.credentials.username == expected.credentials.username
    assert actual.queue == expected.queue


def test_pika_broker_not_in_config():
    cfg = utils.read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "event_broker")

    actual = PikaProducer.from_endpoint_config(cfg)

    assert actual is None


def test_kafka_broker_from_config():
    cfg = utils.read_endpoint_config(KAFKA_EVENT_BROKER_ENDPOINT_FILE,
                                     "event_broker")

    actual = KafkaProducer.from_endpoint_config(cfg)

    expected = KafkaProducer("localhost", "username", "password",
                              topic="topic", security_protocol="SASL_PLAINTEXT")

    assert actual.host == expected.host
    assert actual.sasl_plain_username == expected.sasl_plain_username
    assert actual.sasl_plain_password == expected.sasl_plain_password
    assert actual.topic == expected.topic


def test_kafka_broker_not_in_config():
    cfg = utils.read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "event_broker")

    actual = KafkaProducer.from_endpoint_config(cfg)

    assert actual is None
