import asyncio
import os

from rasa.utils.endpoints import EndpointConfig
from sanic.request import Request
import uuid
from datetime import datetime

from typing import Text, Generator, Callable

import pytest

import rasa.utils.io
from rasa.core.agent import Agent
from rasa.core.channels.channel import CollectingOutputChannel, OutputChannel
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ReminderScheduled, UserUttered, ActionExecuted
from rasa.core.nlg import TemplatedNaturalLanguageGenerator, NaturalLanguageGenerator
from rasa.core.policies.memoization import Policy
from rasa.core.processor import MessageProcessor
from rasa.shared.core.slots import Slot
from rasa.core.tracker_store import InMemoryTrackerStore, MongoTrackerStore
from rasa.core.lock_store import InMemoryLockStore
from rasa.shared.core.trackers import DialogueStateTracker


class CustomSlot(Slot):
    def as_feature(self):
        return [0.5]


# noinspection PyAbstractClass,PyUnusedLocal,PyMissingConstructor
class ExamplePolicy(Policy):
    def __init__(self, *args, **kwargs):
        super(ExamplePolicy, self).__init__(*args, **kwargs)


class MockedMongoTrackerStore(MongoTrackerStore):
    """In-memory mocked version of `MongoTrackerStore`."""

    def __init__(self, _domain: Domain):
        from mongomock import MongoClient

        self.db = MongoClient().rasa
        self.collection = "conversations"

        # skipcq: PYL-E1003
        # Skip `MongoTrackerStore` constructor to avoid that actual Mongo connection
        # is created.
        super(MongoTrackerStore, self).__init__(_domain, None)


# https://github.com/pytest-dev/pytest-asyncio/issues/68
# this event_loop is used by pytest-asyncio, and redefining it
# is currently the only way of changing the scope of this fixture
@pytest.fixture(scope="session")
def event_loop(request: Request) -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = rasa.utils.io.enable_async_loop_debugging(loop)
    yield loop
    loop.close()


@pytest.fixture
def default_channel() -> OutputChannel:
    return CollectingOutputChannel()


@pytest.fixture
async def default_processor(default_agent: Agent) -> MessageProcessor:
    tracker_store = InMemoryTrackerStore(default_agent.domain)
    lock_store = InMemoryLockStore()
    return MessageProcessor(
        default_agent.interpreter,
        default_agent.policy_ensemble,
        default_agent.domain,
        tracker_store,
        lock_store,
        TemplatedNaturalLanguageGenerator(default_agent.domain.responses),
    )


@pytest.fixture
def tracker_with_six_scheduled_reminders(
    default_processor: MessageProcessor,
) -> DialogueStateTracker:
    reminders = [
        ReminderScheduled("greet", datetime.now(), kill_on_user_message=False),
        ReminderScheduled(
            intent="greet",
            entities=[{"entity": "name", "value": "Jane Doe"}],
            trigger_date_time=datetime.now(),
            kill_on_user_message=False,
        ),
        ReminderScheduled(
            intent="default",
            entities=[{"entity": "name", "value": "Jane Doe"}],
            trigger_date_time=datetime.now(),
            kill_on_user_message=False,
        ),
        ReminderScheduled(
            intent="greet",
            entities=[{"entity": "name", "value": "Bruce Wayne"}],
            trigger_date_time=datetime.now(),
            kill_on_user_message=False,
        ),
        ReminderScheduled("default", datetime.now(), kill_on_user_message=False),
        ReminderScheduled(
            "default", datetime.now(), kill_on_user_message=False, name="special"
        ),
    ]
    sender_id = uuid.uuid4().hex
    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)
    for reminder in reminders:
        tracker.update(UserUttered("test"))
        tracker.update(ActionExecuted("action_reminder_reminder"))
        tracker.update(reminder)

    default_processor.tracker_store.save(tracker)

    return tracker


@pytest.fixture
def default_nlg(domain: Domain) -> NaturalLanguageGenerator:
    return TemplatedNaturalLanguageGenerator(domain.responses)


@pytest.fixture
def default_tracker(domain: Domain) -> DialogueStateTracker:
    return DialogueStateTracker("my-sender", domain.slots)


@pytest.fixture(scope="session")
async def form_bot_agent(trained_async: Callable) -> Agent:
    endpoint = EndpointConfig("https://example.com/webhooks/actions")

    zipped_model = await trained_async(
        domain="examples/formbot/domain.yml",
        config="examples/formbot/config.yml",
        training_files=[
            "examples/formbot/data/rules.yml",
            "examples/formbot/data/stories.yml",
        ],
    )

    return Agent.load_local_model(zipped_model, action_endpoint=endpoint)
