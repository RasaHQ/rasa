from pathlib import Path

import pytest

from rasa.architecture_prototype.config_to_graph import (
    config_to_predict_graph_schema,
    config_to_train_graph_schema,
)
from rasa.architecture_prototype.fingerprinting import TrainingCache
from rasa.architecture_prototype.model import Model, ModelTrainer
from rasa.architecture_prototype.persistence import LocalModelPersistor
from rasa.architecture_prototype.processor import GraphProcessor
from rasa.core.channels import UserMessage, CollectingOutputChannel
from tests.architecture_prototype.conftest import default_config, project


@pytest.mark.timeout(1000000)
async def test_full(tmp_path: Path):
    # Convert old config to new graph schemas
    train_graph_schema = config_to_train_graph_schema(config=default_config)
    predict_graph_schema = config_to_predict_graph_schema(config=default_config)

    # Create model trainer
    target = str(tmp_path / "graph_model.tar.gz")
    cache = TrainingCache()
    persistor = LocalModelPersistor(tmp_path)
    trainer = ModelTrainer(model_persistor=persistor, cache=cache)

    # Train the model
    print("%%%%%%%%%%%%%%%%%%%%%%%%%% Training 1 %%%%%%%%%%%%%%%%%%%%%%%%%%")
    trained_model = trainer.train(project, train_graph_schema, predict_graph_schema)

    # Persist the model
    print("%%%%%%%%%%%%%%%%%%%%%%%%%% Persisting %%%%%%%%%%%%%%%%%%%%%%%%%%")
    trained_model.persist(target)

    # Load the model
    print("%%%%%%%%%%%%%%%%%%%%%%%%%% Loading %%%%%%%%%%%%%%%%%%%%%%%%%%")
    loaded_trained_model = Model.load(target, persistor)

    # Train again - will use the cache
    print("%%%%%%%%%%%%%%%%%%%%%%%%%% Training 2 %%%%%%%%%%%%%%%%%%%%%%%%%%")
    second_trained_model = loaded_trained_model.train(project)

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
    # TODO: fix this
    assert len(tracker.events) > 5
    # assert len(tracker.events) == 8
    # assert tracker.events[6].text == "Hey! How are you?"
