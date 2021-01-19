import json
import logging
from pathlib import Path
from asyncio.events import AbstractEventLoop
from typing import Union, Text, List, Optional, Type, Dict, Any

import pytest
from _pytest.logging import LogCaptureFixture

from _pytest.monkeypatch import MonkeyPatch

from rasa.core.brokers import pika
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

TEST_CONNECTION_PARAMETERS = pika.connection.ConnectionParameters(
    "amqp://username:password@host:port"
)


async def test_pika_broker_from_config(monkeypatch: MonkeyPatch):

    # patch PikaEventBroker so it doesn't try to connect to RabbitMQ on init
    async def connect(self) -> None:
        pass

    monkeypatch.setattr(PikaEventBroker, "connect", connect)

    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/pika_endpoint.yml", "event_broker"
    )
    actual = await EventBroker.create(cfg)

    assert isinstance(actual, PikaEventBroker)
    assert actual.host == "localhost"
    assert actual.username == "username"
    assert actual.queues == ["queue-1"]


# noinspection PyProtectedMember
def test_pika_message_property_app_id_without_env_set(monkeypatch: MonkeyPatch):
    # unset RASA_ENVIRONMENT env var results in empty App ID
    monkeypatch.delenv("RASA_ENVIRONMENT", raising=False)
    pika_broker = PikaEventBroker("some host", "username", "password")
    assert not pika_broker._message({}, None).app_id


def test_pika_message_property_app_id(monkeypatch: MonkeyPatch):
    # setting it to some value results in that value as the App ID
    rasa_environment = "some-test-environment"
    monkeypatch.setenv("RASA_ENVIRONMENT", rasa_environment)
    pika_broker = PikaEventBroker("some host", "username", "password")

    assert pika_broker._message({}, None).app_id == rasa_environment


@pytest.mark.parametrize(
    "queue_arg,queues_arg,expected,warning",
    [
        # default case
        (None, ["q1"], ["q1"], None),
        # only provide `queue`
        ("q1", None, ["q1"], FutureWarning),
        # supplying a list for `queue` works too
        (["q1", "q2"], None, ["q1", "q2"], FutureWarning),
        # `queues` arg supplied, takes precedence
        ("q1", "q2", ["q2"], FutureWarning),
        # same, but with a list
        ("q1", ["q2", "q3"], ["q2", "q3"], FutureWarning),
        # only supplying `queues` works, and queues is a string
        (None, "q1", ["q1"], None),
        # no queues provided. Use default queue and print warning.
        (None, None, [DEFAULT_QUEUE_NAME], UserWarning),
    ],
)
def test_pika_queues_from_args(
    queue_arg: Union[Text, List[Text], None],
    queues_arg: Union[Text, List[Text], None],
    expected: List[Text],
    warning: Optional[Type[Warning]],
):
    with pytest.warns(warning):
        pika_processor = PikaEventBroker(
            "host",
            "username",
            "password",
            queues=queues_arg,
            get_message=lambda: ("", None),

        )

    assert pika_processor.queues == expected


async def test_no_broker_in_config():
    cfg = read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "event_broker")

    actual = await EventBroker.create(cfg)

    assert actual is None


async def test_sql_broker_from_config():
    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/sql_endpoint.yml", "event_broker"
    )
    actual = await EventBroker.create(cfg)

    assert isinstance(actual, SQLEventBroker)
    assert actual.engine.name == "sqlite"


async def test_sql_broker_logs_to_sql_db():
    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/sql_endpoint.yml", "event_broker"
    )
    actual = await EventBroker.create(cfg)

    assert isinstance(actual, SQLEventBroker)

    for e in TEST_EVENTS:
        actual.publish(e.as_dict())

    with actual.session_scope() as session:
        events_types = [
            json.loads(event.data)["event"]
            for event in session.query(actual.SQLBrokerEvent).all()
        ]

    assert events_types == ["user", "slot", "restart"]


async def test_file_broker_from_config():
    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/file_endpoint.yml", "event_broker"
    )
    actual = await EventBroker.create(cfg)

    assert isinstance(actual, FileEventBroker)
    assert actual.path == "rasa_event.log"


async def test_file_broker_logs_to_file(tmpdir):
    log_file_path = tmpdir.join("events.log").strpath

    actual = await EventBroker.create(
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


async def test_file_broker_properly_logs_newlines(tmpdir):
    log_file_path = tmpdir.join("events.log").strpath

    actual = await EventBroker.create(
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


def test_load_custom_broker_name():
    config = EndpointConfig(**{"type": "rasa.core.brokers.file.FileEventBroker"})
    assert EventBroker.create(config)


class CustomEventBrokerWithoutAsync(EventBroker):
    @classmethod
    def from_endpoint_config(
        cls,
        broker_config: EndpointConfig,
        event_loop: Optional[AbstractEventLoop] = None,
    ) -> "EventBroker":
        return FileEventBroker()

    def publish(self, event: Dict[Text, Any]) -> None:
        pass


async def test_load_custom_broker_without_async_support(tmpdir):
    log_file_path = tmpdir.join("events.log").strpath

    config = EndpointConfig(
        **{
            "type": f"{CustomEventBrokerWithoutAsync.__module__}."
            f"{CustomEventBrokerWithoutAsync.__name__}",
            "path": str(log_file_path)
        }
    )

    with pytest.warns(UserWarning):
        assert isinstance(await EventBroker.create(config), FileEventBroker)


async def test_load_non_existent_custom_broker_name():
    config = EndpointConfig(**{"type": "rasa.core.brokers.my.MyProducer"})
    assert await EventBroker.create(config) is None


async def test_kafka_broker_from_config():
    endpoints_path = "data/test_endpoints/event_brokers/kafka_plaintext_endpoint.yml"
    cfg = read_endpoint_config(endpoints_path, "event_broker")

    actual = await KafkaEventBroker.from_endpoint_config(cfg)

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


async def test_no_pika_logs_if_no_debug_mode(caplog: LogCaptureFixture):
    broker = PikaEventBroker(
        "host", "username", "password", retry_delay_in_seconds=1, connection_attempts=1
    )

    with pytest.raises(Exception):
        await broker.connect()

    assert (
        len(caplog.records) == 1 and caplog.records[0].name == 'rasa.core.brokers.pika'
    )


def test_warning_if_unsupported_ssl_env_variables(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RABBITMQ_SSL_KEY_PASSWORD", "test")
    monkeypatch.setenv("RABBITMQ_SSL_CA_FILE", "test")

    with pytest.warns(UserWarning):
        pika._create_rabbitmq_ssl_options()
