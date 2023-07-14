from typing import Text
import pytest
import pika
import docker
import json

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


def _callback(ch=, method, properties, body):
    # Do something useful with your incoming message body here, e.g.
    # saving it to a database
    print('Received event {}'.format(json.loads(body)))


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
    credentials = pika.credentials.PlainCredentials(rabbitmq_username, rabbitmq_password)

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

    connection = pika.adapters.BlockingConnection(
        pika.connection.ConnectionParameters(host='0.0.0.0', port=5672,
                                             credentials=credentials, blocked_connection_timeout=1200))
    channel = connection.channel()
    channel.basic_consume(on_message_callback=_callback, queue='queue1', auto_ack=True)
    channel.start_consuming()
    assert channel is not None

    rabbitmq_container_broker.stop()
    assert channel is None
    rabbitmq_container_broker.reload()
    assert channel is not None

    assert rabbitmq_container_broker.status == "exited"
    await broker.close()

    rabbitmq_container_broker.stop()
    rabbitmq_container_broker.remove()

