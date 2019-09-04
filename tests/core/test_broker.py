import json

import rasa.core.brokers.utils as broker_utils
from rasa.core.brokers.file_producer import FileProducer
from rasa.core.brokers.kafka import KafkaProducer
from rasa.core.brokers.pika import PikaProducer
from rasa.core.brokers.sql import SQLProducer
from rasa.core.events import Event, Restarted, SlotSet, UserUttered
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from tests.core.conftest import DEFAULT_ENDPOINTS_FILE

TEST_EVENTS = [
    UserUttered("/greet", {"name": "greet", "confidence": 1.0}, []),
    SlotSet("name", "rasa"),
    Restarted(),
]


def test_pika_broker_from_config():
    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/pika_endpoint.yml", "event_broker"
    )
    actual = broker_utils.from_endpoint_config(cfg)

    assert isinstance(actual, PikaProducer)
    assert actual.host == "localhost"
    assert actual.username == "username"
    assert actual.queue == "queue"


def test_no_broker_in_config():
    cfg = read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "event_broker")

    actual = broker_utils.from_endpoint_config(cfg)

    assert actual is None


def test_sql_broker_from_config():
    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/sql_endpoint.yml", "event_broker"
    )
    actual = broker_utils.from_endpoint_config(cfg)

    assert isinstance(actual, SQLProducer)
    assert actual.engine.name == "sqlite"


def test_sql_broker_logs_to_sql_db():
    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/sql_endpoint.yml", "event_broker"
    )
    actual = broker_utils.from_endpoint_config(cfg)

    for e in TEST_EVENTS:
        actual.publish(e.as_dict())

    with actual.session_scope() as session:
        events_types = [
            json.loads(event.data)["event"]
            for event in session.query(actual.SQLBrokerEvent).all()
        ]

    assert events_types == ["user", "slot", "restart"]


def test_file_broker_from_config():
    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/file_endpoint.yml", "event_broker"
    )
    actual = broker_utils.from_endpoint_config(cfg)

    assert isinstance(actual, FileProducer)
    assert actual.path == "rasa_event.log"


def test_file_broker_logs_to_file(tmpdir):
    fname = tmpdir.join("events.log").strpath

    actual = broker_utils.from_endpoint_config(
        EndpointConfig(**{"type": "file", "path": fname})
    )

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

    actual = broker_utils.from_endpoint_config(
        EndpointConfig(**{"type": "file", "path": fname})
    )

    event_with_newline = UserUttered("hello \n there")

    actual.publish(event_with_newline.as_dict())

    # reading the events from the file one event per line
    recovered = []
    with open(fname, "r") as f:
        for l in f:
            recovered.append(Event.from_parameters(json.loads(l)))

    assert recovered == [event_with_newline]


def test_load_custom_broker_name():
    config = EndpointConfig(**{"type": "rasa.core.brokers.file_producer.FileProducer"})
    assert broker_utils.from_endpoint_config(config)


def test_load_non_existent_custom_broker_name():
    config = EndpointConfig(**{"type": "rasa.core.brokers.my.MyProducer"})
    assert broker_utils.from_endpoint_config(config) is None


def test_kafka_broker_from_config():
    endpoints_path = "data/test_endpoints/event_brokers/kafka_plaintext_endpoint.yml"
    cfg = read_endpoint_config(endpoints_path, "event_broker")

    actual = KafkaProducer.from_endpoint_config(cfg)

    expected = KafkaProducer(
        "localhost",
        "username",
        "password",
        topic="topic",
        security_protocol="SASL_PLAINTEXT",
    )

    assert actual.host == expected.host
    assert actual.sasl_username == expected.sasl_username
    assert actual.sasl_password == expected.sasl_password
    assert actual.topic == expected.topic
