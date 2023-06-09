from typing import Text

import docker
import pytest
import pika
from pytest import LogCaptureFixture

from rasa.core.brokers.pika import PikaEventBroker
from tests.integration_tests.core.brokers.conftest import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_DEFAULT_QUEUE,
)


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
    credentials = pika.PlainCredentials(rabbitmq_username, rabbitmq_password)

    rabbitmq_container = docker_client.containers.run(
        image="healthcheck/rabbitmq",
        detach=True,
        environment=environment,
        name="rabbitmq",
        ports={f"{RABBITMQ_PORT}/tcp": RABBITMQ_PORT},
    )
    rabbitmq_container.reload()
    assert rabbitmq_container.status == "running"

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

    connection = pika.BlockingConnection(
        pika.ConnectionParameters('rabbitmq', credentials=credentials))
    channel = connection.channel()
    channel.basic_consume(queue=RABBITMQ_DEFAULT_QUEUE, auto_ack=True)
    channel.start_consuming()
    assert channel is not None

    rabbitmq_container.stop()
    assert channel is None
    rabbitmq_container.reload()
    assert channel is not None

    assert rabbitmq_container.status == "exited"
    await broker.close()

    rabbitmq_container.stop()
    rabbitmq_container.remove()
