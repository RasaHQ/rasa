import tarfile
from pathlib import Path
from typing import Text

import pytest

from rasa.architecture_prototype.graph_fingerprinting import TrainingCache
from rasa.architecture_prototype.model import ModelTrainer
from rasa.architecture_prototype.persistence import LocalModelPersistor
from rasa.architecture_prototype.processor import GraphProcessor
from rasa.core.agent import Agent
from rasa.core.channels import UserMessage, CollectingOutputChannel
from rasa.core.lock_store import InMemoryLockStore
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.shared.core.constants import ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    SessionStarted,
    DefinePrevUserUtteredFeaturization,
    BotUttered,
    UserUttered,
)
from tests.architecture_prototype.graph_schema import (
    full_model_train_graph_schema,
    predict_graph_schema,
)


@pytest.fixture()
def trained_graph_model(tmp_path: Path) -> Text:
    # Train model
    cache = TrainingCache()
    persistor = LocalModelPersistor(tmp_path)
    trainer = ModelTrainer(model_persistor=persistor, cache=cache)
    trained_model = trainer.train(full_model_train_graph_schema, predict_graph_schema,)

    # Persist model
    persisted_model = "graph_model.tar.gz"
    persistor.create_model_package(persisted_model, predict_graph_schema)

    return persisted_model


async def test_handle_message(trained_graph_model: Text, tmp_path: Path):
    placeholder_domain = Domain.empty()
    nlg = TemplatedNaturalLanguageGenerator(placeholder_domain.responses)
    tracker_store = InMemoryTrackerStore(placeholder_domain)

    with tarfile.open(trained_graph_model, mode="r:gz") as tar:
        tar.extractall(tmp_path)

    processor = GraphProcessor.create(
        str(tmp_path),
        tracker_store=tracker_store,
        lock_store=InMemoryLockStore(),
        generator=nlg,
        action_endpoint=None,
    )

    sender = "test_handle_message"
    await processor.handle_message(
        UserMessage(
            text="hi", output_channel=CollectingOutputChannel(), sender_id=sender
        )
    )

    expected_events = [
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("hi", {"name": "greet"}),
        DefinePrevUserUtteredFeaturization(False),
        ActionExecuted("utter_greet"),
        BotUttered(
            "Hey! How are you?",
            data={
                "elements": None,
                "quick_replies": None,
                "buttons": [
                    {"title": "great", "payload": "/mood_great"},
                    {"title": "super sad", "payload": "/mood_unhappy"},
                ],
                "attachment": None,
                "image": None,
                "custom": None,
            },
            metadata={"utter_action": "utter_greet"},
        ),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]
    tracker = tracker_store.retrieve(sender)
    for event, expected in zip(tracker.events, expected_events):
        assert event == expected


async def test_agent_with_graph_processor(trained_graph_model: Text, tmp_path: Path):
    # Load Agent with graph model
    placeholder_domain = Domain.empty()
    nlg = TemplatedNaturalLanguageGenerator(placeholder_domain.responses)
    tracker_store = InMemoryTrackerStore(placeholder_domain)

    agent = Agent.load_local_model(
        trained_graph_model, generator=nlg, tracker_store=tracker_store
    )

    assert isinstance(agent.processor, GraphProcessor)

    # Test we can do predictions
    sender = "test_agent_with_graph_processor"
    responses = await agent.handle_message(
        UserMessage(
            text="hi", output_channel=CollectingOutputChannel(), sender_id=sender
        )
    )

    assert responses == [
        {
            "recipient_id": "test_agent_with_graph_processor",
            "text": "Hey! How are you?",
            "buttons": [
                {"title": "great", "payload": "/mood_great"},
                {"title": "super sad", "payload": "/mood_unhappy"},
            ],
        }
    ]
