from typing import Text, Dict, Any

import pytest
import copy

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers.keyword_intent_classifier import (
    KeywordIntentClassifierGraphComponent,
)
from rasa.shared.nlu.constants import TEXT

from rasa.shared.nlu.training_data.formats.rasa import RasaReader
import rasa.shared.nlu.training_data.loading
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData


@pytest.fixture(scope="module")
def training_data(nlu_as_json_path: Text):
    return rasa.shared.nlu.training_data.loading.load_data(nlu_as_json_path)


@pytest.fixture()
def default_keyword_intent_classifier(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext,
):
    return KeywordIntentClassifierGraphComponent.create(
        KeywordIntentClassifierGraphComponent.get_default_config(),
        default_model_storage,
        Resource("keyword"),
        default_execution_context,
    )


@pytest.mark.parametrize(
    "config", [{"case_sensitive": True}, {"case_sensitive": False}]
)
def test_persist_and_load(
    training_data: TrainingData,
    config: Dict[Text, Any],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    classifier = KeywordIntentClassifierGraphComponent.create(
        config, default_model_storage, Resource("keyword"), default_execution_context,
    )
    classifier.train(training_data)

    loaded_classifier = KeywordIntentClassifierGraphComponent.load(
        config, default_model_storage, Resource("keyword"), default_execution_context,
    )

    predicted = copy.copy(training_data)
    actual = copy.copy(training_data)
    loaded_messages = loaded_classifier.process(predicted.training_examples)
    trained_messages = classifier.process(actual.training_examples)
    for m1, m2 in zip(loaded_messages, trained_messages):
        assert m1.get("intent") == m2.get("intent")


@pytest.mark.parametrize(
    "message, intent",
    [
        ("hey there joe", "greet"),
        ("hello weiouaosdhalkh", "greet"),
        ("show me chinese restaurants in the north of town", "restaurant_search"),
        ("great", "affirm"),
        ("bye bye birdie", "goodbye"),
        ("show me a mexican place", None),
        ("i", None),
        ("in", None),
        ("eet", None),
    ],
)
def test_classification(
    message: Text,
    intent: Text,
    training_data: TrainingData,
    default_keyword_intent_classifier: KeywordIntentClassifierGraphComponent,
):
    text = Message(data={TEXT: message})
    default_keyword_intent_classifier.train(training_data)
    messages = default_keyword_intent_classifier.process([text])
    for m in messages:
        assert m.get("intent").get("name", "NOT_CLASSIFIED") == intent


def test_valid_data(
    default_keyword_intent_classifier: KeywordIntentClassifierGraphComponent,
):
    json_data = {
        "rasa_nlu_data": {
            "common_examples": [
                {"text": "good", "intent": "affirm", "entities": []},
                {"text": "bye", "intent": "goodbye", "entities": []},
                {"text": "see ya", "intent": "goodbye", "entities": []},
                {"text": "yes", "intent": "affirm", "entities": []},
                {"text": "ciao", "intent": "goodbye", "entities": []},
            ]
        }
    }
    rasa_reader = RasaReader()
    data = rasa_reader.read_from_json(json_data)

    with pytest.warns(None) as record:
        default_keyword_intent_classifier.train(data)
    assert len(record) == 0


@pytest.mark.filterwarnings("ignore:Keyword.* of keywords:UserWarning")
def test_identical_data(
    default_keyword_intent_classifier: KeywordIntentClassifierGraphComponent,
):
    json_data = {
        "rasa_nlu_data": {
            "common_examples": [
                {"text": "good", "intent": "affirm", "entities": []},
                {"text": "good", "intent": "goodbye", "entities": []},
            ]
        }
    }
    rasa_reader = RasaReader()
    data = rasa_reader.read_from_json(json_data)

    with pytest.warns(UserWarning) as record:
        default_keyword_intent_classifier.train(data)
    assert len(record) == 1
    assert (
        "Remove (one of) the duplicates from the training data."
        in record[0].message.args[0]
    )


@pytest.mark.filterwarnings("ignore:Keyword.* of keywords:UserWarning")
def test_ambiguous_data(
    default_keyword_intent_classifier: KeywordIntentClassifierGraphComponent,
):
    json_data = {
        "rasa_nlu_data": {
            "common_examples": [
                {"text": "good", "intent": "affirm", "entities": []},
                {"text": "good morning", "intent": "greet", "entities": []},
                {"text": "see you", "intent": "goodbye", "entities": []},
                {"text": "nice to see you", "intent": "greet", "entities": []},
            ]
        }
    }
    rasa_reader = RasaReader()
    data = rasa_reader.read_from_json(json_data)

    with pytest.warns(UserWarning) as record:
        default_keyword_intent_classifier.train(data)
    assert len(record) == 2
