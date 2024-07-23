import asyncio
import os
import uuid
from datetime import datetime
from typing import Generator, Callable, Dict, Text, List, Any
from unittest.mock import patch, Mock

import pytest
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sanic.request import Request
from scipy import sparse

from rasa.core.agent import Agent
from rasa.core.channels.channel import CollectingOutputChannel, OutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator, NaturalLanguageGenerator
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_store import MongoTrackerStore
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ReminderScheduled, UserUttered, ActionExecuted
from rasa.shared.core.slots import Slot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    INTENT,
    ACTION_NAME,
    FEATURE_TYPE_SENTENCE,
)
from rasa.shared.nlu.training_data.features import Features
from rasa.utils.endpoints import EndpointConfig
from tests.core.utilities import tracker_from_dialogue
from tests.dialogues import TEST_MOODBOT_DIALOGUE


class CustomSlot(Slot):
    type_name = "custom"

    def _as_feature(self):
        return [0.5]


class MockedMongoTrackerStore(MongoTrackerStore):
    """In-memory mocked version of `MongoTrackerStore`."""

    def __init__(self, _domain: Domain) -> None:
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
# update: implement fix to RuntimeError Event loop is closed issue described
# here: https://github.com/pytest-dev/pytest-asyncio/issues/371
@pytest.fixture(scope="session")
def event_loop(request: Request) -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    loop._close = loop.close
    loop.close = lambda: None
    yield loop
    loop._close()


# override loop fixture to prevent ScopeMismatch pytest error and
# align the result of the loop fixture with that of the event_loop fixture
@pytest.fixture(scope="session")
def loop(
    event_loop: asyncio.AbstractEventLoop,
) -> Generator[asyncio.AbstractEventLoop, None, None]:
    yield event_loop


@pytest.fixture
def default_channel() -> OutputChannel:
    return CollectingOutputChannel()


@pytest.fixture
async def default_processor(default_agent: Agent) -> MessageProcessor:
    return default_agent.processor


@pytest.fixture
async def tracker_with_six_scheduled_reminders(
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
    tracker = await default_processor.tracker_store.get_or_create_tracker(sender_id)
    for reminder in reminders:
        tracker.update(UserUttered("test"))
        tracker.update(ActionExecuted("action_reminder_reminder"))
        tracker.update(reminder)

    await default_processor.tracker_store.save(tracker)

    return tracker


@pytest.fixture
def default_nlg(domain: Domain) -> NaturalLanguageGenerator:
    return TemplatedNaturalLanguageGenerator(domain.responses)


@pytest.fixture
def default_tracker(domain: Domain) -> DialogueStateTracker:
    return DialogueStateTracker("my-sender", domain.slots)


@pytest.fixture(scope="session")
async def trained_formbot(trained_async: Callable) -> Text:
    return await trained_async(
        domain="examples/nlu_based/formbot/domain.yml",
        config="examples/nlu_based/formbot/config.yml",
        training_files=[
            "examples/nlu_based/formbot/data/rules.yml",
            "examples/nlu_based/formbot/data/stories.yml",
        ],
    )


@pytest.fixture(scope="module")
async def form_bot_agent(trained_formbot: Text) -> Agent:
    endpoint = EndpointConfig("https://example.com/webhooks/actions")

    return Agent.load(trained_formbot, action_endpoint=endpoint)


@pytest.fixture
def moodbot_features(
    request: Request, moodbot_domain: Domain
) -> Dict[Text, Dict[Text, Features]]:
    """Makes intent and action features for the moodbot domain to faciliate
    making expected state features.

    Returns:
      A dict containing dicts for mapping action and intent names to features.
    """
    origin = getattr(request, "param", "SingleStateFeaturizer")
    action_shape = (1, len(moodbot_domain.action_names_or_texts))
    actions = {}
    for index, action in enumerate(moodbot_domain.action_names_or_texts):
        actions[action] = Features(
            sparse.coo_matrix(([1.0], [[0], [index]]), shape=action_shape),
            FEATURE_TYPE_SENTENCE,
            ACTION_NAME,
            origin,
        )
    intent_shape = (1, len(moodbot_domain.intents))
    intents = {}
    for index, intent in enumerate(moodbot_domain.intents):
        intents[intent] = Features(
            sparse.coo_matrix(([1.0], [[0], [index]]), shape=intent_shape),
            FEATURE_TYPE_SENTENCE,
            INTENT,
            origin,
        )
    return {"intents": intents, "actions": actions}


@pytest.fixture
def moodbot_tracker(moodbot_domain: Domain) -> DialogueStateTracker:
    return tracker_from_dialogue(TEST_MOODBOT_DIALOGUE, moodbot_domain)


@pytest.fixture(scope="session", autouse=True)
def set_open_ai_env_variable():
    os.environ["OPENAI_API_KEY"] = "test"


@pytest.fixture(scope="session")
@patch("langchain.vectorstores.faiss.FAISS.from_documents")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
async def trained_flow_policy_bot(
    mock_flow_search_create_embedder: Mock,
    mock_from_documents: Mock,
    trained_async: Callable,
) -> Text:
    mock_flow_search_create_embedder.return_value = Mock()
    mock_from_documents.return_value = Mock()
    return await trained_async(
        domain="data/test_flow_policy_bot/domain.yml",
        config="data/test_flow_policy_bot/config.yml",
        training_files=[
            "data/test_flow_policy_bot/data/flows.yml",
        ],
    )


@pytest.fixture(scope="session")
@patch("langchain.vectorstores.faiss.FAISS.from_documents")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
async def trained_nlu_trigger_flow_policy_bot(
    mock_flow_search_create_embedder: Mock,
    mock_from_documents: Mock,
    trained_async: Callable,
) -> Text:
    mock_flow_search_create_embedder.return_value = Mock()
    mock_from_documents.return_value = Mock()
    return await trained_async(
        domain="data/test_nlu_trigger_flow_policy_bot/domain.yml",
        config="data/test_nlu_trigger_flow_policy_bot/config.yml",
        training_files=[
            "data/test_nlu_trigger_flow_policy_bot/data/flows.yml",
        ],
    )


@pytest.fixture
@patch("langchain.vectorstores.faiss.FAISS.load_local")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
async def flow_policy_bot_agent(
    mock_flow_search_create_embedder: Mock,
    mock_load_local: Mock,
    trained_flow_policy_bot: Text,
) -> Agent:
    mock_flow_search_create_embedder.return_value = Mock()
    mock_load_local.return_value = Mock()
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    return Agent.load(model_path=trained_flow_policy_bot, action_endpoint=endpoint)


@pytest.fixture
@patch("langchain.vectorstores.faiss.FAISS.load_local")
@patch(
    "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._create_embedder"
)
async def nlu_trigger_flow_policy_bot_agent(
    mock_flow_search_create_embedder: Mock,
    mock_load_local: Mock,
    trained_nlu_trigger_flow_policy_bot: Text,
) -> Agent:
    mock_flow_search_create_embedder.return_value = Mock()
    mock_load_local.return_value = Mock()
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    return Agent.load(
        model_path=trained_nlu_trigger_flow_policy_bot, action_endpoint=endpoint
    )
