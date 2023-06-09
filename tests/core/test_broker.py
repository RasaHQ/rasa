import json
import logging
import textwrap
from pathlib import Path
from typing import Union, Text, List, Optional, Type

import aio_pika.exceptions
import aiormq.exceptions
import pamqp.exceptions
import confluent_kafka
import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from aiormq import ChannelNotFoundEntity

from rasa.core.brokers import pika
from tests.conftest import AsyncMock

import rasa.shared.utils.io
import rasa.utils.io
from rasa.core.brokers.broker import EventBroker
from rasa.core.brokers.file import FileEventBroker
from rasa.core.brokers.kafka import KafkaEventBroker, KafkaProducerInitializationError
from rasa.core.brokers.pika import PikaEventBroker, DEFAULT_QUEUE_NAME
from rasa.core.brokers.sql import SQLEventBroker
from rasa.shared.core.events import Event, Restarted, SlotSet, UserUttered
from rasa.shared.exceptions import ConnectionException, RasaException
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config

TEST_EVENTS = [
    UserUttered("/greet", {"name": "greet", "confidence": 1.0}, []),
    SlotSet("name", "rasa"),
    Restarted(),
]


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
    assert actual.exchange_name == "exchange"


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


async def test_pika_raise_connection_exception(monkeypatch: MonkeyPatch):

    monkeypatch.setattr(
        PikaEventBroker, "connect", AsyncMock(side_effect=ChannelNotFoundEntity())
    )

    with pytest.raises(ConnectionException):
        await EventBroker.create(
            EndpointConfig(username="username", password="password", type="pika")
        )


@pytest.mark.parametrize(
    "exception",
    (
        RuntimeError,
        ConnectionError,
        OSError,
        aiormq.exceptions.AMQPError,
        pamqp.exceptions.PAMQPException,
        pamqp.exceptions.AMQPConnectionForced,
        pamqp.exceptions.AMQPNotFound,
        pamqp.exceptions.AMQPInternalError,
    ),
)
async def test_aio_pika_exceptions_caught(
    exception: Exception, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(PikaEventBroker, "connect", AsyncMock(side_effect=exception))

    with pytest.raises(ConnectionException):
        await EventBroker.create(
            EndpointConfig(username="username", password="password", type="pika")
        )


async def test_no_broker_in_config(endpoints_path: Text):
    cfg = read_endpoint_config(endpoints_path, "event_broker")

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


async def test_file_broker_from_config(tmp_path: Path):
    # backslashes need to be encoded (windows...) otherwise we run into unicode issues
    path = str(tmp_path / "rasa_test_event.log").replace("\\", "\\\\")
    endpoint_config = textwrap.dedent(
        f"""
        event_broker:
          path: "{path}"
          type: "file"
    """
    )
    rasa.shared.utils.io.write_text_file(endpoint_config, tmp_path / "endpoint.yml")

    cfg = read_endpoint_config(str(tmp_path / "endpoint.yml"), "event_broker")
    actual = await EventBroker.create(cfg)

    assert isinstance(actual, FileEventBroker)
    assert actual.path.endswith("rasa_test_event.log")


async def test_file_broker_logs_to_file(tmp_path: Path):
    log_file_path = str(tmp_path / "events.log")

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


async def test_file_broker_properly_logs_newlines(tmp_path: Path):
    log_file_path = str(tmp_path / "events.log")

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


async def test_load_custom_broker_name(tmp_path: Path):
    config = EndpointConfig(
        **{
            "type": "rasa.core.brokers.file.FileEventBroker",
            "path": str(tmp_path / "rasa_event.log"),
        }
    )
    broker = await EventBroker.create(config)
    assert broker


async def test_load_non_existent_custom_broker_name():
    config = EndpointConfig(**{"type": "rasa.core.brokers.my.MyProducer"})
    assert await EventBroker.create(config) is None


async def test_kafka_broker_from_config():
    endpoints_path = (
        "data/test_endpoints/event_brokers/kafka_sasl_plaintext_endpoint.yml"
    )
    cfg = read_endpoint_config(endpoints_path, "event_broker")

    actual = await KafkaEventBroker.from_endpoint_config(cfg)

    expected = KafkaEventBroker(
        "localhost",
        sasl_username="username",
        sasl_password="password",
        sasl_mechanism="PLAIN",
        topic="topic",
        partition_by_sender=True,
        security_protocol="SASL_PLAINTEXT",
    )

    assert actual.url == expected.url
    assert actual.sasl_username == expected.sasl_username
    assert actual.sasl_password == expected.sasl_password
    assert actual.sasl_mechanism == expected.sasl_mechanism
    assert actual.topic == expected.topic
    assert actual.partition_by_sender == expected.partition_by_sender


@pytest.mark.parametrize(
    "file,exception",
    [
        ("kafka_sasl_plaintext_endpoint.yml", confluent_kafka.KafkaException),
        ("kafka_plaintext_endpoint.yml", confluent_kafka.KafkaException),
        ("kafka_sasl_ssl_endpoint.yml", KafkaProducerInitializationError),
        ("kafka_ssl_endpoint.yml", KafkaProducerInitializationError),
        # `ValueError` exception is raised when the `security_protocol` is incorrect
        ("kafka_invalid_security_protocol.yml", ValueError),
        # `confluent_kafka.KafkaException` exception is raised when there is no
        # `url` specified
        ("kafka_plaintext_endpoint_no_url.yml", confluent_kafka.KafkaException),
        # `KafkaProducerInitializationError` is raised when an invalid
        # `sasl_mechanism` is provided
        ("kafka_invalid_sasl_mechanism.yml", KafkaProducerInitializationError),
    ],
)
async def test_kafka_broker_security_protocols(file: Text, exception: Exception):
    endpoints_path = f"data/test_endpoints/event_brokers/{file}"
    cfg = read_endpoint_config(endpoints_path, "event_broker")

    actual = await KafkaEventBroker.from_endpoint_config(cfg)
    with pytest.raises(exception):
        # noinspection PyProtectedMember
        producer = actual._create_producer()

        # required action to trigger expected exception because the configuration
        # seems correct and the producer gets instantiated but a connection to the
        # broker cannot be established
        producer.list_topics("topic", timeout=1)


@pytest.mark.flaky
async def test_no_pika_logs_if_no_debug_mode(caplog: LogCaptureFixture):
    """
    tests that when you run rasa with logging set at INFO,
    the debugs from pika dependency are not going to be shown
    """
    broker = PikaEventBroker(
        "host", "username", "password", retry_delay_in_seconds=1, connection_attempts=1
    )

    with caplog.at_level(logging.INFO):
        with pytest.raises(Exception):
            await broker.connect()

    # Only Rasa Open Source logs, but logs from the library itself.
    assert all(
        record.name
        in ["rasa.core.brokers.pika", "asyncio", "ddtrace.internal.writer.writer"]
        for record in caplog.records
    )


async def test_create_pika_invalid_port():

    cfg = EndpointConfig(
        username="username", password="password", type="pika", port="PORT"
    )
    with pytest.raises(RasaException) as e:
        await EventBroker.create(cfg)
        assert "Port could not be converted to integer." in str(e.value)


def test_warning_if_unsupported_ssl_env_variables(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RABBITMQ_SSL_KEY_PASSWORD", "test")
    monkeypatch.setenv("RABBITMQ_SSL_CA_FILE", "test")

    with pytest.warns(UserWarning):
        pika._create_rabbitmq_ssl_options()


async def test_pika_connection_error(monkeypatch: MonkeyPatch):
    # patch PikaEventBroker to raise an AMQP connection error
    async def connect(self) -> None:
        raise aio_pika.exceptions.ProbableAuthenticationError("Oups")

    monkeypatch.setattr(PikaEventBroker, "connect", connect)
    cfg = EndpointConfig.from_dict(
        {
            "type": "pika",
            "url": "localhost",
            "username": "username",
            "password": "password",
            "queues": ["queue-1"],
            "connection_attempts": 1,
            "retry_delay_in_seconds": 0,
        }
    )
    with pytest.raises(ConnectionException):
        await EventBroker.create(cfg)


async def test_sql_connection_error(monkeypatch: MonkeyPatch):
    cfg = EndpointConfig.from_dict(
        {
            "type": "sql",
            "dialect": "postgresql",
            "url": "0.0.0.0",
            "port": 42,
            "db": "boom",
            "username": "user",
            "password": "pw",
        }
    )
    with pytest.raises(ConnectionException):
        await EventBroker.create(cfg)


@pytest.mark.parametrize(
    "host,expected_url",
    [
        ("localhost", None),
        ("amqp://localhost", "amqp://test_user:test_pass@localhost:5672"),
        (
            "amqp://test_user:test_pass@localhost",
            "amqp://test_user:test_pass@localhost:5672",
        ),
        (
            "amqp://test_user:test_pass@localhost/myvhost?connection_timeout=10",
            "amqp://test_user:test_pass@localhost:5672/myvhost?connection_timeout=10",
        ),
        ("amqp://localhost:5672", "amqp://test_user:test_pass@localhost:5672"),
        (
            "amqp://test_user:test_pass@localhost:5672/myvhost?connection_timeout=10",
            "amqp://test_user:test_pass@localhost:5672/myvhost?connection_timeout=10",
        ),
    ],
)
def test_pika_event_broker_configure_url(
    host: Text, expected_url: Optional[Text]
) -> None:
    username = "test_user"
    password = "test_pass"
    broker = PikaEventBroker(host=host, username=username, password=password)
    url = broker._configure_url()
    assert url == expected_url
