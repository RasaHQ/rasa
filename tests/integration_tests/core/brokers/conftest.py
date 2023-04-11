import os
from typing import Text

import docker
import pytest

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", 5672)
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "")
RABBITMQ_DEFAULT_QUEUE = "queue1"


@pytest.fixture
def docker_client() -> docker.DockerClient:
    docker_client = docker.from_env()
    prev_containers = docker_client.containers.list(all=True)

    for container in prev_containers:
        container.stop()

    docker_client.containers.prune()

    return docker_client


@pytest.fixture
def rabbitmq_username() -> Text:
    return "test_user"


@pytest.fixture
def rabbitmq_password() -> Text:
    return "test_password"
