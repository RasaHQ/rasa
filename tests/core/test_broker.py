import json
import logging
from pathlib import Path
import textwrap

from typing import Union, Text, List, Optional, Type

import pytest
from _pytest.logging import LogCaptureFixture

from _pytest.monkeypatch import MonkeyPatch

import rasa.utils.io
from rasa.core.brokers.broker import EventBroker
from rasa.core.brokers.file import FileEventBroker
from rasa.core.brokers.kafka import KafkaEventBroker
from rasa.core.brokers.pika import PikaEventBroker, DEFAULT_QUEUE_NAME
from rasa.core.brokers.sql import SQLEventBroker
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
    actual = EventBroker.create(cfg)

    assert isinstance(actual, PikaEventBroker)
    assert actual.host == "localhost"
    assert actual.username == "username"
    assert actual.queues == ["queue-1"]


# noinspection PyProtectedMember
def test_pika_message_property_app_id(monkeypatch: MonkeyPatch):
    # patch PikaEventBroker so it doesn't try to connect to RabbitMQ on init
    monkeypatch.setattr(PikaEventBroker, "_run_pika", lambda _: None)
    pika_producer = PikaEventBroker("", "", "")

    # unset RASA_ENVIRONMENT env var results in empty App ID
    monkeypatch.delenv("RASA_ENVIRONMENT", raising=False)
    assert not pika_producer._get_message_properties().app_id

    # setting it to some value results in that value as the App ID
    rasa_environment = "some-test-environment"
    monkeypatch.setenv("RASA_ENVIRONMENT", rasa_environment)
    assert pika_producer._get_message_properties().app_id == rasa_environment


@pytest.mark.parametrize(
    "queues_arg,expected,warning",
    [
        # default case
        (["q1", "q2"], ["q1", "q2"], None),
        # `queues` arg supplied, as string
        ("q1", ["q1"], None),
        # no queues provided. Use default queue and print warning.
        (None, [DEFAULT_QUEUE_NAME], UserWarning),
    ],
)
def test_pika_queues_from_args(
    queues_arg: Union[Text, List[Text], None],
    expected: List[Text],
    warning: Optional[Type[Warning]],
    monkeypatch: MonkeyPatch,
):
    # patch PikaEventBroker so it doesn't try to connect to RabbitMQ on init
    monkeypatch.setattr(PikaEventBroker, "_run_pika", lambda _: None)

    with pytest.warns(warning):
        pika_producer = PikaEventBroker("", "", "", queues=queues_arg)

    assert pika_producer.queues == expected


def test_no_broker_in_config():
    cfg = read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "event_broker")

    actual = EventBroker.create(cfg)

    assert actual is None


def test_sql_broker_from_config():
    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/sql_endpoint.yml", "event_broker"
    )
    actual = EventBroker.create(cfg)

    assert isinstance(actual, SQLEventBroker)
    assert actual.engine.name == "sqlite"


def test_sql_broker_logs_to_sql_db():
    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/sql_endpoint.yml", "event_broker"
    )
    actual = EventBroker.create(cfg)

    assert isinstance(actual, SQLEventBroker)

    for e in TEST_EVENTS:
        actual.publish(e.as_dict())

    with actual.session_scope() as session:
        events_types = [
            json.loads(event.data)["event"]
            for event in session.query(actual.SQLBrokerEvent).all()
        ]

    assert events_types == ["user", "slot", "restart"]


def test_file_broker_from_config(tmp_path: Path):
    # backslashes need to be encoded (windows...) otherwise we run into unicode issues
    path = str(tmp_path / "rasa_test_event.log").replace("\\", "\\\\")
    endpoint_config = textwrap.dedent(
        f"""
        event_broker:
          path: "{path}"
          type: "file"
    """
    )
    rasa.utils.io.write_text_file(endpoint_config, tmp_path / "endpoint.yml")

    cfg = read_endpoint_config(str(tmp_path / "endpoint.yml"), "event_broker")
    actual = EventBroker.create(cfg)

    assert isinstance(actual, FileEventBroker)
    assert actual.path.endswith("rasa_test_event.log")


def test_file_broker_logs_to_file(tmp_path: Path):
    log_file_path = str(tmp_path / "events.log")

    actual = EventBroker.create(
        EndpointConfig(**{"type": "file", "path": log_file_path})
    )

    for e in TEST_EVENTS:
        actual.publish(e.as_dict())

    # reading the events from the file one event per line
    recovered = []
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            recovered.append(Event.from_parameters(json.loads(line)))

    assert recovered == TEST_EVENTS


def test_file_broker_properly_logs_newlines(tmp_path):
    log_file_path = str(tmp_path / "events.log")

    actual = EventBroker.create(
        EndpointConfig(**{"type": "file", "path": log_file_path})
    )

    event_with_newline = UserUttered("hello \n there")

    actual.publish(event_with_newline.as_dict())

    # reading the events from the file one event per line
    recovered = []
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            recovered.append(Event.from_parameters(json.loads(line)))

    assert recovered == [event_with_newline]


def test_load_custom_broker_name(tmp_path: Path):
    config = EndpointConfig(
        **{
            "type": "rasa.core.brokers.file.FileEventBroker",
            "path": str(tmp_path / "rasa_event.log"),
        }
    )
    assert EventBroker.create(config)


def test_load_non_existent_custom_broker_name():
    config = EndpointConfig(**{"type": "rasa.core.brokers.my.MyProducer"})
    assert EventBroker.create(config) is None


def test_kafka_broker_from_config():
    endpoints_path = "data/test_endpoints/event_brokers/kafka_plaintext_endpoint.yml"
    cfg = read_endpoint_config(endpoints_path, "event_broker")

    actual = KafkaEventBroker.from_endpoint_config(cfg)

    expected = KafkaEventBroker(
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


def test_no_pika_logs_if_no_debug_mode(caplog: LogCaptureFixture):
    from rasa.core.brokers import pika

    with caplog.at_level(logging.INFO):
        with pytest.raises(Exception):
            pika.initialise_pika_connection(
                "localhost", "user", "password", connection_attempts=1
            )

    assert len(caplog.records) == 0


def test_pika_logs_in_debug_mode(caplog: LogCaptureFixture, monkeypatch: MonkeyPatch):
    from rasa.core.brokers.pika import _pika_log_level

    pika_level = logging.getLogger("pika").level

    with caplog.at_level(logging.INFO):
        with _pika_log_level(logging.CRITICAL):
            assert logging.getLogger("pika").level == logging.CRITICAL

    with caplog.at_level(logging.DEBUG):
        with _pika_log_level(logging.CRITICAL):
            # level should not change
            assert logging.getLogger("pika").level == pika_level
