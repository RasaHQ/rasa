import logging

from pytest import LogCaptureFixture

from rasa.core.brokers.pika import PikaEventBroker
from .conftest import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_USER,
    RABBITMQ_PASSWORD,
    RABBITMQ_DEFAULT_QUEUE,
)


async def test_pika_event_broker_connect():
    broker = PikaEventBroker(
        host=RABBITMQ_HOST,
        username=RABBITMQ_USER,
        password=RABBITMQ_PASSWORD,
        port=RABBITMQ_PORT,
        queues=[RABBITMQ_DEFAULT_QUEUE],
    )
    try:
        await broker.connect()
        assert broker.is_ready()
    finally:
        await broker.close()


async def test_pika_event_broker_publish_after_restart(
    caplog: LogCaptureFixture,
):
    broker = PikaEventBroker(
        host=RABBITMQ_HOST,
        username=RABBITMQ_USER,
        password=RABBITMQ_PASSWORD,
        port=RABBITMQ_PORT,
        queues=[RABBITMQ_DEFAULT_QUEUE],
    )
    await broker.connect()
    assert broker.is_ready()
    broker.publish({"event": "test"})

    with caplog.at_level(logging.DEBUG):
        await broker.close()
        assert "Closing RabbitMQ connection." in caplog.text

    caplog.clear()

    # reconnect with the same broker
    with caplog.at_level(logging.WARNING):
        await broker.connect()
        assert broker.is_ready()

        broker.publish({"event": "test_after_restart"})
        assert caplog.text == ""

    await broker.close()
