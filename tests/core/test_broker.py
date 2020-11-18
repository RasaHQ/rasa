import json
import logging
import textwrap
from asyncio.events import AbstractEventLoop

import kafka
import pytest

from pathlib import Path
from typing import Union, Text, List, Optional, Type, Dict, Any
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.brokers import pika
from tests.core.conftest import DEFAULT_ENDPOINTS_FILE

import rasa.shared.utils.io
import rasa.utils.io
from rasa.core.brokers.broker import EventBroker
from rasa.core.brokers.file import FileEventBroker
from rasa.core.brokers.kafka import KafkaEventBroker
from rasa.core.brokers.pika import PikaEventBroker, DEFAULT_QUEUE_NAME
from rasa.core.brokers.sql import SQLEventBroker
from rasa.shared.core.events import Event, Restarted, SlotSet, UserUttered
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


def test_load_custom_broker_name(tmp_path: Path):
    config = EndpointConfig(
        **{
            "type": "rasa.core.brokers.file.FileEventBroker",
            "path": str(tmp_path / "rasa_event.log"),
        }
    )
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


async def test_load_custom_broker_without_async_support(tmp_path: Path):
    config = EndpointConfig(
        **{
            "type": f"{CustomEventBrokerWithoutAsync.__module__}."
            f"{CustomEventBrokerWithoutAsync.__name__}",
            "path": str(tmp_path / "rasa_event.log"),
        }
    )

    with pytest.warns(FutureWarning):
        assert isinstance(await EventBroker.create(config), FileEventBroker)


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
        topic="topic",
        security_protocol="SASL_PLAINTEXT",
    )

    assert actual.url == expected.url
    assert actual.sasl_username == expected.sasl_username
    assert actual.sasl_password == expected.sasl_password
    assert actual.topic == expected.topic


@pytest.mark.parametrize(
    "file,exception",
    [
        # `_create_producer()` raises `kafka.errors.NoBrokersAvailable` exception
        # which means that the configuration seems correct but a connection to
        # the broker cannot be established
        ("kafka_sasl_plaintext_endpoint.yml", kafka.errors.NoBrokersAvailable),
        ("kafka_plaintext_endpoint.yml", kafka.errors.NoBrokersAvailable),
        ("kafka_sasl_ssl_endpoint.yml", kafka.errors.NoBrokersAvailable),
        ("kafka_ssl_endpoint.yml", kafka.errors.NoBrokersAvailable),
        # `ValueError` exception is raised when the `security_protocol` is incorrect
        ("kafka_invalid_security_protocol.yml", ValueError),
        # `TypeError` exception is raised when there is no `url` specified
        ("kafka_plaintext_endpoint_no_url.yml", TypeError),
    ],
)
async def test_kafka_broker_security_protocols(file: Text, exception: Exception):
    endpoints_path = f"data/test_endpoints/event_brokers/{file}"
    cfg = read_endpoint_config(endpoints_path, "event_broker")

    actual = await KafkaEventBroker.from_endpoint_config(cfg)
    with pytest.raises(exception):
        # noinspection PyProtectedMember
        actual._create_producer()


async def test_no_pika_logs_if_no_debug_mode(caplog: LogCaptureFixture):
    broker = PikaEventBroker(
        "host", "username", "password", retry_delay_in_seconds=1, connection_attempts=1
    )

    with caplog.at_level(logging.INFO):
        with pytest.raises(Exception):
            await broker.connect()

    # Only Rasa Open Source logs, but logs from the library itself.
    assert all(
        record.name in ["rasa.core.brokers.pika", "asyncio"]
        for record in caplog.records
    )


def test_warning_if_unsupported_ssl_env_variables(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RABBITMQ_SSL_KEY_PASSWORD", "test")
    monkeypatch.setenv("RABBITMQ_SSL_CA_FILE", "test")

    with pytest.warns(UserWarning):
        pika._create_rabbitmq_ssl_options()
