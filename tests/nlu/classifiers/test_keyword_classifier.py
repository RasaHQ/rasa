from typing import Text, Dict, Any, Optional, Union

import pytest
import copy

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    PREDICTED_CONFIDENCE_KEY,
    INTENT_NAME_KEY,
)

from rasa.shared.nlu.training_data.formats.rasa import RasaReader
import rasa.shared.nlu.training_data.loading
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData


@pytest.fixture()
def training_data(nlu_as_json_path: Text):
    return rasa.shared.nlu.training_data.loading.load_data(nlu_as_json_path)


@pytest.fixture()
def default_keyword_intent_classifier(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    return KeywordIntentClassifier.create(
        KeywordIntentClassifier.get_default_config(),
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
    classifier = KeywordIntentClassifier.create(
        config, default_model_storage, Resource("keyword"), default_execution_context
    )
    classifier.train(training_data)

    loaded_classifier = KeywordIntentClassifier.load(
        config, default_model_storage, Resource("keyword"), default_execution_context
    )

    predicted = copy.copy(training_data)
    actual = copy.copy(training_data)
    loaded_messages = loaded_classifier.process(predicted.training_examples)
    trained_messages = classifier.process(actual.training_examples)
    for m1, m2 in zip(loaded_messages, trained_messages):
        assert m1.get("intent") == m2.get("intent")


@pytest.mark.parametrize(
    "message, previous_intent, expected_intent",
    [
        (
            "hey there joe",
            None,
            {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1.0},
        ),  # Keyword match
        # Keyword match
        (
            "hello weiouaosdhalkh",
            None,
            {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 1.0},
        ),
        # Keyword match
        (
            "show me chinese restaurants in the north of town",
            None,
            {INTENT_NAME_KEY: "restaurant_search", PREDICTED_CONFIDENCE_KEY: 1.0},
        ),
        (
            "great",
            None,
            {INTENT_NAME_KEY: "affirm", PREDICTED_CONFIDENCE_KEY: 1.0},
        ),  # Keyword match
        # Keyword match
        (
            "bye bye birdie",
            None,
            {INTENT_NAME_KEY: "goodbye", PREDICTED_CONFIDENCE_KEY: 1.0},
        ),
        # No keyword match, no previous intent
        (
            "show me a mexican place",
            None,
            {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},
        ),
        # No keyword match, no previous intent
        (
            "i",
            None,
            {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},
        ),
        # No keyword match, no previous intent
        (
            "in",
            None,
            {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},
        ),
        # No keyword match, no previous intent
        (
            "eet",
            None,
            {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},
        ),
        # previous and no keyword match
        (
            "The Neapolitan",
            {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 0.123},
            {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 0.123},
        ),
        # previous and keyword match
        (
            "I am searching for a dinner spot",
            {INTENT_NAME_KEY: "greet", PREDICTED_CONFIDENCE_KEY: 0.123},
            {INTENT_NAME_KEY: "restaurant_search", PREDICTED_CONFIDENCE_KEY: 1.0},
        ),
    ],
)
def test_classification(
    message: Text,
    previous_intent: Optional[Dict[Text, Union[str, float]]],
    expected_intent: Dict[Text, Union[str, float]],
    training_data: TrainingData,
    default_keyword_intent_classifier: KeywordIntentClassifier,
):
    text = Message(data={TEXT: message, INTENT: previous_intent})
    default_keyword_intent_classifier.train(training_data)
    messages = default_keyword_intent_classifier.process([text])
    for m in messages:
        assert m.get(INTENT) == expected_intent


def test_valid_data(default_keyword_intent_classifier: KeywordIntentClassifier):
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
def test_identical_data(default_keyword_intent_classifier: KeywordIntentClassifier):
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
def test_ambiguous_data(default_keyword_intent_classifier: KeywordIntentClassifier):
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
