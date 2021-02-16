import os
from typing import Iterator, Text, Type

import pytest
from _pytest.tmpdir import TempdirFactory
import sqlalchemy as sa

from rasa.core.agent import Agent
from rasa.core.lock_store import LockStore, RedisLockStore
from rasa.core.policies.memoization import AugmentedMemoizationPolicy
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.shared.core.domain import SessionConfig


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DEFAULT_DB = os.getenv("POSTGRES_DEFAULT_DB", "postgres")
POSTGRES_TRACKER_STORE_DB = "tracker_store_db"
POSTGRES_LOGIN_DB = "login_db"

DEFAULT_STORIES_FILE = "data/test_yaml_stories/stories_defaultdomain.yml"


@pytest.fixture
def redis_lock_store() -> RedisLockStore:
    return RedisLockStore(REDIS_HOST, REDIS_PORT)


@pytest.fixture(scope="session")
async def _trained_default_agent(tmpdir_factory: TempdirFactory) -> Agent:
    model_path = tmpdir_factory.mktemp("model").strpath

    agent = Agent(
        "data/test_domains/default_with_slots.yml",
        policies=[AugmentedMemoizationPolicy(max_history=3), RulePolicy()],
    )

    training_data = await agent.load_data(DEFAULT_STORIES_FILE)
    agent.train(training_data)
    agent.persist(model_path)
    return agent


@pytest.fixture
async def default_agent(_trained_default_agent: Agent) -> Agent:
    agent = _trained_default_agent
    agent.tracker_store = InMemoryTrackerStore(agent.domain)
    agent.domain.session_config = SessionConfig.default()
    return agent


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


def _create_login_db(connection: sa.engine.Connection) -> None:
    connection.execution_options(isolation_level="AUTOCOMMIT").execute(
        f"CREATE DATABASE {POSTGRES_LOGIN_DB}"
    )


def _drop_db(connection: sa.engine.Connection, database_name: Text) -> None:
    connection.execution_options(isolation_level="AUTOCOMMIT").execute(
        f"DROP DATABASE IF EXISTS {database_name}"
    )
