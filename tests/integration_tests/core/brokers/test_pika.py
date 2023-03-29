import asyncio
import logging
import time
from typing import AsyncGenerator, Text

import aio_pika
import docker
import pytest
from aio_pika import ExchangeType
from aio_pika.abc import AbstractIncomingMessage
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


@pytest.fixture(autouse=True)
def docker_client() -> AsyncGenerator[docker.DockerClient, None]:
    docker_client = docker.from_env()
    prev_containers = docker_client.containers.list(all=True)

    for container in prev_containers:
        container.stop()

    docker_client.containers.prune()

    yield docker_client


@pytest.fixture
def rabbitmq_username() -> Text:
    return "test_user"


@pytest.fixture
def rabbitmq_password() -> Text:
    return "test_password"


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
        user="root",
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

        assert f"Published Pika events to exchange '{RABBITMQ_EXCHANGE}' on host " \
               f"'localhost':\n{event}" in caplog.text

    await broker.close()
