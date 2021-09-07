import copy
from typing import Text

import pytest

import rasa.shared.nlu.training_data.loading
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu import registry
from rasa.nlu.classifiers.sklearn_intent_classifier import (
    SklearnIntentClassifierGraphComponent,
)
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData


@pytest.fixture()
def training_data(nlu_data_path: Text):
    return rasa.shared.nlu.training_data.loading.load_data(nlu_data_path)


@pytest.fixture()
def default_sklearn_intent_classifier(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext,
):
    return SklearnIntentClassifierGraphComponent.create(
        SklearnIntentClassifierGraphComponent.get_default_config(),
        default_model_storage,
        Resource("sklearn"),
        default_execution_context,
    )


def test_persist_and_load(
    training_data: TrainingData,
    default_sklearn_intent_classifier: SklearnIntentClassifierGraphComponent,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    pipeline = [
        {"name": "SpacyNLP", "model": "en_core_web_md"},
        {"name": "SpacyTokenizer"},
        {"name": "SpacyFeaturizer"},
    ]
    loaded_pipeline = [
        registry.get_component_class(component.pop("name")).create(
            component, RasaNLUModelConfig()
        )
        for component in copy.deepcopy(pipeline)
    ]
    for component in loaded_pipeline:
        component.train(training_data)

    default_sklearn_intent_classifier.train(training_data)

    loaded = SklearnIntentClassifierGraphComponent.load(
        SklearnIntentClassifierGraphComponent.get_default_config(),
        default_model_storage,
        Resource("sklearn"),
        default_execution_context,
    )

    predicted = copy.copy(training_data)
    actual = copy.copy(training_data)
    loaded_messages = loaded.process(predicted.training_examples)
    trained_messages = default_sklearn_intent_classifier.process(
        actual.training_examples
    )

    for m1, m2 in zip(loaded_messages, trained_messages):
        assert m1.get("intent") == m2.get("intent")
