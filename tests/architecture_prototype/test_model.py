import os
import time
from pathlib import Path

from rasa.architecture_prototype.persistence import LocalModelPersistor
from rasa.architecture_prototype.model import ModelTrainer, Model
from rasa.architecture_prototype.graph_fingerprinting import TrainingCache
from rasa.core.channels import UserMessage
from rasa.shared.core.trackers import DialogueStateTracker
from tests.architecture_prototype import graph_schema
from tests.architecture_prototype.graph_schema import (
    full_model_train_graph_schema,
    predict_graph_schema,
)


def test_model_training():
    cache = TrainingCache()
    persistor = LocalModelPersistor(Path("model"))

    # Train model
    trainer = ModelTrainer(model_persistor=persistor, cache=cache)
    trained_model = trainer.train(full_model_train_graph_schema, predict_graph_schema,)

    # Persist model
    persisted_model = "graph_model.tar.gz"
    persistor.create_model_package(persisted_model, predict_graph_schema)

    # Load model
    persistor = LocalModelPersistor(Path("loaded_model"))
    Path("loaded_model").mkdir()
    model = Model.load(persisted_model, persistor)

    # Make prediction
    tracker = DialogueStateTracker.from_events("test_graph_model", [])
    prediction, user_uttered = model.handle_message(tracker, UserMessage(text="Hi"))

    assert prediction
    assert user_uttered


def test_model_training_with_timing():
    cache = TrainingCache()
    persistor = LocalModelPersistor(Path("model"))

    # Train model
    trainer = ModelTrainer(model_persistor=persistor, cache=cache)

    start = time.time()
    train_schema = graph_schema.train_schema_for(
        os.environ.get("PROJECT", graph_schema.default_project)
    )

    print(f"Started model training at {start}")
    trained_model = trainer.train(train_schema, predict_graph_schema,)

    # Persist model
    persisted_model = "graph_model.tar.gz"
    persistor.create_model_package(persisted_model, predict_graph_schema)

    end = time.time()
    print(f"Finished model training at {end}. Total time: {end - start}")


if __name__ == "__main__":
    test_model_training_with_timing()
