from typing import Text
import pytest
import asyncio
from aio_pika import ExchangeType, connect
from aio_pika.abc import AbstractIncomingMessage, AbstractConnection
import docker

from pytest import LogCaptureFixture

from rasa.core.brokers.pika import PikaEventBroker
from tests.integration_tests.core.brokers.conftest import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_DEFAULT_QUEUE,
)


@pytest.fixture
def rabbitmq_username() -> Text:
    return "test_user"


@pytest.fixture
def rabbitmq_password() -> Text:
    return "test_password"


async def on_message(message: AbstractIncomingMessage) -> None:
    async with message.process():
        print(f"[x] {message.body!r}")


async def start_consuming() -> None:
    # Perform connection
    connection = await connect("amqp://test_user:test_password@127.0.0.1/")

    async with connection:
        # Creating a channel
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        # Declaring queue
        queue = await channel.declare_queue(
            RABBITMQ_DEFAULT_QUEUE, passive=True
        )

        # Start listening the queue
        await queue.consume(on_message)


@pytest.mark.xdist_group("rabbitmq")
async def test_consume_after_pika_restart(
        docker_client: docker.DockerClient,
        caplog: LogCaptureFixture,
        rabbitmq_username: Text,
        rabbitmq_password: Text,
) -> None:
    environment = {
        "RABBITMQ_DEFAULT_USER": rabbitmq_username,
        "RABBITMQ_DEFAULT_PASS": rabbitmq_password,
    }

    rabbitmq_container_broker = docker_client.containers.run(
        image="rabbitmq:3-management",
        detach=True,
        environment=environment,
        name="rabbitmq_broker",
        domainname="rabbitmq.com",
        ports={f"{RABBITMQ_PORT}/tcp": RABBITMQ_PORT},
    )
    rabbitmq_container_broker.reload()
    assert rabbitmq_container_broker.status == "running"

    broker = PikaEventBroker(
        host=RABBITMQ_HOST,
        username=rabbitmq_username,
        password=rabbitmq_password,
        port=RABBITMQ_PORT,
        queues=[RABBITMQ_DEFAULT_QUEUE],
        should_keep_unpublished_messages=True,
    )

    await broker.connect()
    assert broker.is_ready()
    broker.publish({"event": "test"})
    await start_consuming()

    rabbitmq_container_broker.stop()
    rabbitmq_container_broker.reload()

    await broker.close()

    rabbitmq_container_broker.stop()
    rabbitmq_container_broker.remove()
