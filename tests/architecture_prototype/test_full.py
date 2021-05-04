import tarfile
from pathlib import Path
from typing import Text

import pytest

from rasa.architecture_prototype.config_to_graph import (
    config_to_predict_graph_schema,
    config_to_train_graph_schema,
)
from rasa.architecture_prototype.graph_fingerprinting import TrainingCache
from rasa.architecture_prototype.model import Model, ModelTrainer
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

default_config = """
pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: "char_wb"
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 1
    constrain_similarities: true
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 1
    constrain_similarities: true
policies:
  - name: MemoizationPolicy
  - name: TEDPolicy
    max_history: 5
    epochs: 1
    constrain_similarities: true
  - name: RulePolicy
"""

project = "examples/moodbot"


@pytest.mark.timeout(1000000)
async def test_full(tmp_path: Path):
    # Convert old config to new graph schemas
    (
        train_graph_schema,
        train_graph_targets,
    ) = config_to_train_graph_schema(  # TODO: targets as part of schema?
        project=project, config=default_config
    )
    predict_graph_schema, predict_graph_targets = config_to_predict_graph_schema(
        config=default_config
    )

    # Create model trainer
    target = "graph_model.tar.gz"
    cache = TrainingCache()
    persistor = LocalModelPersistor(tmp_path)
    trainer = ModelTrainer(model_persistor=persistor, cache=cache)

    # Train the model
    print("%%%%%%%%%%%%%%%%%%%%%%%%%% Training 1 %%%%%%%%%%%%%%%%%%%%%%%%%%")
    trained_model = trainer.train(
        train_graph_schema, train_graph_targets, predict_graph_schema
    )

    # Persist the model
    print("%%%%%%%%%%%%%%%%%%%%%%%%%% Persisting %%%%%%%%%%%%%%%%%%%%%%%%%%")
    trained_model.persist(target)

    # Load the model
    print("%%%%%%%%%%%%%%%%%%%%%%%%%% Loading %%%%%%%%%%%%%%%%%%%%%%%%%%")
    loaded_trained_model = Model.load(target, persistor)

    # Train again - will use the cache
    print("%%%%%%%%%%%%%%%%%%%%%%%%%% Training 2 %%%%%%%%%%%%%%%%%%%%%%%%%%")
    second_trained_model = loaded_trained_model.train()

    # Create the graph processor
    processor = GraphProcessor.from_model(second_trained_model)

    # Send a message
    print("%%%%%%%%%%%%%%%%%%%%%%%%%% Predicting %%%%%%%%%%%%%%%%%%%%%%%%%%")
    sender = "test_handle_message"
    await processor.handle_message(
        UserMessage(
            text="hi", output_channel=CollectingOutputChannel(), sender_id=sender
        )
    )

    tracker = processor.tracker_store.retrieve(sender)
    assert len(tracker.events) == 8
    assert tracker.events[6].text == "Hey! How are you?"
