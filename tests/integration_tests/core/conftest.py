import os
from docker import DockerClient
from docker.models.containers import Container
from typing import Iterator, Text, Optional, Callable, List

import pytest
import sqlalchemy as sa

from rasa.core.lock_store import RedisLockStore


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "1")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DEFAULT_DB = os.getenv("POSTGRES_DEFAULT_DB", "postgres")
POSTGRES_TRACKER_STORE_DB = "tracker_store_db"
POSTGRES_LOGIN_DB = "login_db"


@pytest.fixture
def redis_lock_store() -> Iterator[RedisLockStore]:
    # we need one redis database per worker, otherwise
    # tests conflicts with each others when databases are flushed
    pytest_worker_id = os.getenv("PYTEST_XDIST_WORKER", "gw0")
    redis_database = int(pytest_worker_id.replace("gw", ""))
    lock_store = RedisLockStore(REDIS_HOST, REDIS_PORT, redis_database)
    try:
        yield lock_store
    finally:
        lock_store.red.flushdb()


@pytest.fixture
def postgres_login_db_connection() -> Iterator[sa.engine.Connection]:
    engine = sa.create_engine(
        sa.engine.url.URL(
            "postgresql",
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            username=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DEFAULT_DB,
        )
    )

    conn = engine.connect()
    try:
        _create_login_db(conn)
        yield conn
    finally:
        _drop_db(conn, POSTGRES_LOGIN_DB)
        _drop_db(conn, POSTGRES_TRACKER_STORE_DB)
        conn.close()
        engine.dispose()


@pytest.fixture(scope="session")
def docker_client() -> DockerClient:
    docker_client = DockerClient.from_env()
    return docker_client


@pytest.fixture(scope="session")
def redis_image(docker_client: DockerClient) -> None:
    docker_client.images.pull("redis:latest")


@pytest.fixture(scope="session")
def _clean_test_containers() -> None:
    docker_client = DockerClient.from_env()
    containers: List[Container] = docker_client.containers.list(
        all=True, filters={"label": "rasa"}
    )
    print("Cleaning up test containers")
    for container in containers:
        print(f"Removing container {container.name}")
        if container.status == "running":
            container.stop()
        container.remove(force=True)


CreateRedisContainer = Callable[[Optional[Text]], Container]


@pytest.mark.usefixtures("_clean_test_containers", "redis_image")
@pytest.fixture(scope="session")
def create_docker_redis_container(
    docker_client,
) -> CreateRedisContainer:
    created_containers = []

    def _create_redis_container(redis_password: Optional[Text] = None) -> Container:
        entrypoint = "redis-server"

        if redis_password is not None:
            entrypoint += f" --requirepass {redis_password}"

        redis_container = docker_client.containers.create(
            "redis:latest",
            name="rasa_test_redis",
            ports={"6379/tcp": REDIS_PORT},
            detach=True,
            labels=["rasa", "redis"],
            entrypoint=entrypoint,
        )

        created_containers.append(redis_container)

        return redis_container

    yield _create_redis_container

    for container in created_containers:
        print(f"Removing container {container.name}")
        container.remove(force=True)


@pytest.fixture(scope="session")
def docker_redis_with_password(
    create_docker_redis_container: CreateRedisContainer,
) -> None:
    redis_container = create_docker_redis_container(REDIS_PASSWORD)
    redis_container.start()
    try:
        yield redis_container
    finally:
        print(f"Stopping container {redis_container.name}")
        redis_container.stop()


def _create_login_db(connection: sa.engine.Connection) -> None:
    connection.execution_options(isolation_level="AUTOCOMMIT").execute(
        f"CREATE DATABASE {POSTGRES_LOGIN_DB}"
    )


def _drop_db(connection: sa.engine.Connection, database_name: Text) -> None:
    connection.execution_options(isolation_level="AUTOCOMMIT").execute(
        f"DROP DATABASE IF EXISTS {database_name}"
    )
