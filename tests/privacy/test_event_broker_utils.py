from asyncio import AbstractEventLoop
from typing import Dict
from unittest.mock import AsyncMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from structlog.testing import capture_logs

from rasa.core.brokers.kafka import KafkaEventBroker
from rasa.core.brokers.pika import PikaEventBroker
from rasa.privacy.event_broker_utils import create_event_brokers
from rasa.utils.endpoints import EndpointConfig
from tests.utilities import filter_logs


@pytest.fixture
def kafka_endpoint_config():
    endpoint_config = EndpointConfig(url="localhost:9092", type="kafka")
    endpoint_config.kwargs["anonymization_topics"] = [
        "anonymization_topic",
        "another_anonymization_topic",
    ]
    return endpoint_config


@pytest.fixture
def pika_endpoint_config():
    return EndpointConfig(
        type="pika",
        url="localhost:5672",
        queues=["some_queue"],
        anonymization_queues=["anonymization_queue", "another_anonymization_queue"],
        username="test_user",
        password="test_password",
    )


async def test_create_event_brokers_kafka(kafka_endpoint_config: EndpointConfig):
    with capture_logs() as caplog:
        brokers = await create_event_brokers(kafka_endpoint_config)

        logs = filter_logs(
            caplog,
            "rasa.privacy_filtering.create_event_broker",
            "debug",
        )
        # the log repeats twice because of the two topics
        assert len(logs) == 2

    assert len(brokers) == 2
    assert all([isinstance(broker, KafkaEventBroker) for broker in brokers])

    assert brokers[0].topic == "anonymization_topic"
    assert brokers[1].topic == "another_anonymization_topic"


async def test_create_event_brokers_pika(
    pika_endpoint_config: EndpointConfig,
    event_loop: AbstractEventLoop,
    monkeypatch: MonkeyPatch,
):
    # patch PikaEventBroker so it doesn't try to connect to RabbitMQ on init
    monkeypatch.setattr(PikaEventBroker, "connect", AsyncMock())
    with capture_logs() as caplog:
        brokers = await create_event_brokers(pika_endpoint_config, event_loop)

        logs = filter_logs(
            caplog,
            "rasa.privacy_filtering.create_event_broker",
            "debug",
        )
        # the log repeats twice because of the two queues
        assert len(logs) == 2

    assert len(brokers) == 2
    assert all([isinstance(broker, PikaEventBroker) for broker in brokers])

    assert brokers[0].queues == ["anonymization_queue"]
    assert brokers[1].queues == ["another_anonymization_queue"]


async def test_create_event_brokers_no_event_broker_type(
    event_loop: AbstractEventLoop,
):
    endpoint_config = EndpointConfig()
    with capture_logs() as caplog:
        brokers = await create_event_brokers(endpoint_config, event_loop)

        logs = filter_logs(
            caplog,
            "rasa.privacy_filtering.create_event_broker.no_event_broker_type",
            "debug",
        )
        assert len(logs) == 1

    assert brokers == []


async def test_create_event_brokers_unsupported_event_broker():
    endpoint_config = EndpointConfig(type="sql")
    with capture_logs() as caplog:
        brokers = await create_event_brokers(endpoint_config)

        logs = filter_logs(
            caplog,
            "rasa.privacy_filtering.create_event_broker.unsupported_event_broker_type",
            "debug",
        )
        assert len(logs) == 1
        assert logs[0].get("event_broker_type") == "sql"

    assert brokers == []


@pytest.mark.parametrize(
    "broker_type, kwargs, log_event",
    [
        (
            "kafka",
            {},
            "rasa.privacy_filtering.create_event_broker.no_anonymization_topic",
        ),
        (
            "pika",
            {"username": "test_user", "password": "test_password"},
            "rasa.privacy_filtering.create_event_broker.no_anonymization_queues",
        ),
    ],
)
async def test_create_event_brokers_no_anonymization_topics_or_queues(
    broker_type: str,
    kwargs: Dict,
    log_event: str,
    monkeypatch: MonkeyPatch,
) -> None:
    # patch PikaEventBroker so it doesn't try to connect to RabbitMQ on init
    if broker_type == "pika":
        monkeypatch.setattr(PikaEventBroker, "connect", AsyncMock())

    endpoint_config = EndpointConfig(type=broker_type, kwargs=kwargs)
    with capture_logs() as caplog:
        brokers = await create_event_brokers(endpoint_config)

        logs = filter_logs(
            caplog,
            log_event,
            "debug",
        )
        assert len(logs) == 1

    assert brokers == []
