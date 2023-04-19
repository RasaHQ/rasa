import logging
from typing import Text

import docker
from pytest import LogCaptureFixture

from rasa.core.brokers.pika import PikaEventBroker, RABBITMQ_EXCHANGE
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
    docker_client: docker.DockerClient,
    caplog: LogCaptureFixture,
    rabbitmq_username: Text,
    rabbitmq_password: Text,
) -> None:
    environment = {
        "RABBITMQ_DEFAULT_USER": rabbitmq_username,
        "RABBITMQ_DEFAULT_PASS": rabbitmq_password,
    }

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

    rabbitmq_container.stop()
    rabbitmq_container.reload()

    assert rabbitmq_container.status == "exited"

    with caplog.at_level(logging.ERROR):
        event = {"event": "test_while_closed"}
        await broker._publish(event)
        assert "Failed to publish Pika event" in caplog.text
        assert f"The message was: \n{event}" in caplog.text

    caplog.clear()

    # reconnect with the same broker
    rabbitmq_container.restart()
    rabbitmq_container.reload()
    assert rabbitmq_container.status == "running"

    with caplog.at_level(logging.DEBUG):
        await broker.connect()
        assert broker.is_ready()

        after_restart_event = {"event": "test_after_restart"}
        await broker._publish(after_restart_event)

        assert (
            f"Published Pika events to exchange '{RABBITMQ_EXCHANGE}' on host "
            f"'localhost':\n{after_restart_event}" in caplog.text
        )

    await broker.close()

    rabbitmq_container.stop()
    rabbitmq_container.remove()
