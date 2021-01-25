import os
from typing import Type

import pytest
from _pytest.tmpdir import TempdirFactory

from rasa.core.agent import Agent
from rasa.core.lock_store import LockStore, RedisLockStore
from rasa.core.policies.memoization import AugmentedMemoizationPolicy
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.shared.core.domain import SessionConfig


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

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
