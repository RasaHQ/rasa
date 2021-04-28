import os
import shutil
import tarfile
from pathlib import Path
from typing import Text, Dict, Any

import pytest

from rasa.architecture_prototype import graph
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
from tests.architecture_prototype import conftest
from tests.architecture_prototype.graph_schema import (
    full_model_train_graph_schema,
    predict_graph_schema,
)


async def test_handle_message(prediction_graph: Dict[Text, Any]):
    placeholder_domain = Domain.empty()
    nlg = TemplatedNaturalLanguageGenerator(placeholder_domain.responses)
    tracker_store = InMemoryTrackerStore(placeholder_domain)
    processor = GraphProcessor.create(
        tracker_store=tracker_store,
        lock_store=InMemoryLockStore(),
        generator=nlg,
        action_endpoint=None,
        rasa_graph=prediction_graph,
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


async def test_agent_with_graph_processor(
    prediction_graph: Dict[Text, Any], tmp_path: Path
):
    model_dir = tmp_path / "unpacked_model"

    shutil.copytree("model", model_dir)

    zipped_model = tmp_path / "graph_model.tar.gz"

    # It only works when the graph model is zipped
    with tarfile.open(zipped_model, "w:gz") as tar:
        for elem in os.scandir(model_dir):
            tar.add(elem.path, arcname=elem.name)

    placeholder_domain = Domain.empty()
    nlg = TemplatedNaturalLanguageGenerator(placeholder_domain.responses)
    tracker_store = InMemoryTrackerStore(placeholder_domain)
    agent = Agent.load_local_model(
        str(zipped_model), generator=nlg, tracker_store=tracker_store
    )

    assert isinstance(agent.processor, GraphProcessor)

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
