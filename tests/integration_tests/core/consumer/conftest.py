from typing import Text

import docker
import pytest


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
