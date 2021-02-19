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
