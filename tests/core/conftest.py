import asyncio
import os

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
from rasa.core.policies.ensemble import PolicyEnsemble
from rasa.core.policies.memoization import Policy
from rasa.core.processor import MessageProcessor
from rasa.shared.core.slots import Slot
from rasa.core.tracker_store import InMemoryTrackerStore, MongoTrackerStore
from rasa.shared.core.trackers import DialogueStateTracker

DEFAULT_DOMAIN_PATH_WITH_SLOTS = "data/test_domains/default_with_slots.yml"

DEFAULT_DOMAIN_PATH_WITH_SLOTS_AND_NO_ACTIONS = (
    "data/test_domains/default_with_slots_and_no_actions.yml"
)

DEFAULT_DOMAIN_PATH_WITH_MAPPING = "data/test_domains/default_with_mapping.yml"

DEFAULT_STORIES_FILE = "data/test_stories/stories_defaultdomain.md"

DEFAULT_STACK_CONFIG = "data/test_config/stack_config.yml"

INCORRECT_NLU_DATA = "data/test/markdown_single_sections/incorrect_nlu_format.md"

END_TO_END_STORY_FILE = "data/test_evaluations/end_to_end_story.md"

E2E_STORY_FILE_UNKNOWN_ENTITY = "data/test_evaluations/story_unknown_entity.md"

STORY_FILE_TRIPS_CIRCUIT_BREAKER = (
    "data/test_evaluations/stories_trip_circuit_breaker.md"
)

E2E_STORY_FILE_TRIPS_CIRCUIT_BREAKER = (
    "data/test_evaluations/end_to_end_trips_circuit_breaker.md"
)

DEFAULT_ENDPOINTS_FILE = "data/test_endpoints/example_endpoints.yml"

TEST_DIALOGUES = [
    "data/test_dialogues/default.json",
    "data/test_dialogues/formbot.json",
    "data/test_dialogues/moodbot.json",
]

EXAMPLE_DOMAINS = [
    DEFAULT_DOMAIN_PATH_WITH_SLOTS,
    DEFAULT_DOMAIN_PATH_WITH_SLOTS_AND_NO_ACTIONS,
    DEFAULT_DOMAIN_PATH_WITH_MAPPING,
    "examples/formbot/domain.yml",
    "examples/moodbot/domain.yml",
]


class CustomSlot(Slot):
    def as_feature(self):
        return [0.5]


# noinspection PyAbstractClass,PyUnusedLocal,PyMissingConstructor
class ExamplePolicy(Policy):
    def __init__(self, example_arg):
        super(ExamplePolicy, self).__init__()


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
@pytest.yield_fixture(scope="session")
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


@pytest.fixture(scope="session")
def default_domain_path() -> Text:
    return DEFAULT_DOMAIN_PATH_WITH_SLOTS


@pytest.fixture(scope="session")
def default_stories_file() -> Text:
    return DEFAULT_STORIES_FILE


@pytest.fixture(scope="session")
def default_stack_config() -> Text:
    return DEFAULT_STACK_CONFIG


@pytest.fixture(scope="session")
def default_nlu_data():
    from tests.conftest import DEFAULT_NLU_DATA

    return DEFAULT_NLU_DATA


@pytest.fixture
def default_channel() -> OutputChannel:
    return CollectingOutputChannel()


@pytest.fixture
async def default_processor(default_agent: Agent) -> MessageProcessor:
    tracker_store = InMemoryTrackerStore(default_agent.domain)
    return MessageProcessor(
        default_agent.interpreter,
        default_agent.policy_ensemble,
        default_agent.domain,
        tracker_store,
        TemplatedNaturalLanguageGenerator(default_agent.domain.templates),
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


@pytest.fixture(scope="session")
def moodbot_metadata(unpacked_trained_moodbot_path: Text) -> PolicyEnsemble:
    return PolicyEnsemble.load_metadata(
        os.path.join(unpacked_trained_moodbot_path, "core")
    )


@pytest.fixture
def default_nlg(default_domain: Domain) -> NaturalLanguageGenerator:
    return TemplatedNaturalLanguageGenerator(default_domain.templates)


@pytest.fixture
def default_tracker(default_domain: Domain) -> DialogueStateTracker:
    return DialogueStateTracker("my-sender", default_domain.slots)


@pytest.fixture
async def form_bot_agent(trained_async) -> Agent:
    zipped_model = await trained_async(
        domain="examples/formbot/domain.yml",
        config="examples/formbot/config.yml",
        training_files=[
            "examples/formbot/data/rules.yml",
            "examples/formbot/data/stories.yml",
        ],
    )

    return Agent.load_local_model(zipped_model)


@pytest.fixture(scope="session")
async def response_selector_agent(trained_async: Callable) -> Agent:
    zipped_model = await trained_async(
        domain="examples/responseselectorbot/domain.yml",
        config="examples/responseselectorbot/config.yml",
        training_files=[
            "examples/responseselectorbot/data/rules.yml",
            "examples/responseselectorbot/data/stories.yml",
            "examples/responseselectorbot/data/nlu.yml",
        ],
    )

    return Agent.load_local_model(zipped_model)
