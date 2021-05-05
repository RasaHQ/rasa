from pathlib import Path
from typing import Text

import pytest

from rasa.architecture_prototype.fingerprinting import TrainingCache
from rasa.architecture_prototype.model import ModelTrainer
from rasa.architecture_prototype.persistence import LocalModelPersistor
from rasa.architecture_prototype.processor import GraphProcessor
from rasa.core.agent import Agent
from rasa.core.channels import UserMessage, CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.shared.core.domain import Domain
from tests.architecture_prototype.conftest import project
from tests.architecture_prototype.graph_schema import (
    train_graph_schema,
    predict_graph_schema,
)


@pytest.fixture()
def trained_graph_model(tmp_path: Path) -> Text:
    # Train model
    target = str(tmp_path / "graph_model.tar.gz")
    cache = TrainingCache()
    persistor = LocalModelPersistor(tmp_path)
    trainer = ModelTrainer(model_persistor=persistor, cache=cache)
    trained_model = trainer.train(project, train_graph_schema, predict_graph_schema)

    # Persist model
    persistor.create_model_package(target, trained_model)

    return target


async def test_agent_with_graph_processor(trained_graph_model: Text):
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
