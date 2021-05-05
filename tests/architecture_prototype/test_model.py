import os
import time
from pathlib import Path

import pytest

from rasa.architecture_prototype.persistence import LocalModelPersistor
from rasa.architecture_prototype.model import ModelTrainer, Model
from rasa.architecture_prototype.fingerprinting import TrainingCache
from rasa.core.channels import UserMessage
from rasa.shared.core.trackers import DialogueStateTracker
from tests.architecture_prototype.conftest import project
from tests.architecture_prototype.graph_schema import (
    train_graph_schema,
    predict_graph_schema,
)


@pytest.mark.timeout(600)
def test_model_training(tmp_path: Path):
    target = str(tmp_path / "graph_model.tar.gz")
    cache = TrainingCache()
    persistor = LocalModelPersistor(tmp_path)

    # Train model
    trainer = ModelTrainer(model_persistor=persistor, cache=cache)
    trained_model = trainer.train(project, train_graph_schema, predict_graph_schema)

    # Persist model
    persisted_model = "graph_model.tar.gz"
    persistor.create_model_package(persisted_model, trained_model)

    # Load model
    persistor = LocalModelPersistor(tmp_path)
    model = Model.load(target, persistor)

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
    print(f"Started model training at {start}")
    trained_model = trainer.train(
        os.environ.get("PROJECT", project), train_graph_schema, predict_graph_schema
    )

    # Persist model
    persisted_model = "graph_model.tar.gz"
    persistor.create_model_package(persisted_model, trained_model)

    end = time.time()
    print(f"Finished model training at {end}. Total time: {end - start}")


if __name__ == "__main__":
    test_model_training_with_timing()
