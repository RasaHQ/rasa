import copy
import pytest
import pathlib
import numpy as np
from rasa.shared.nlu.training_data.message import Message
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.graph import ExecutionContext
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.classifiers.logistic_regression_classifier import (
    LogisticRegressionClassifier,
)


@pytest.fixture
def featurizer(tmpdir):
    """Generate a featurizer for tests."""
    node_storage = LocalModelStorage(pathlib.Path(tmpdir))
    node_resource = Resource("sparse_feat")
    context = ExecutionContext(node_storage, node_resource)
    return CountVectorsFeaturizer(
        config=CountVectorsFeaturizer.get_default_config(),
        resource=node_resource,
        model_storage=node_storage,
        execution_context=context,
    )


tokeniser = WhitespaceTokenizer(
    {
        "only_alphanum": False,
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_",
    }
)


@pytest.fixture()
def training_data():
    # Create training data.
    return TrainingData(
        [
            Message({"text": "hello", "intent": "greet"}),
            Message({"text": "hi there", "intent": "greet"}),
            Message({"text": "ciao", "intent": "goodbye"}),
            Message({"text": "bye", "intent": "goodbye"}),
        ]
    )


def test_predictions_added(training_data, tmpdir, featurizer):
    """Checks if the sizes are appropriate."""
    # Set up classifier
    node_storage = LocalModelStorage(pathlib.Path(tmpdir))
    node_resource = Resource("classifier")
    context = ExecutionContext(node_storage, node_resource)
    classifier = LogisticRegressionClassifier(
        config=LogisticRegressionClassifier.get_default_config(),
        name=context.node_name,
        resource=node_resource,
        model_storage=node_storage,
    )

    # First we add tokens.
    tokeniser.process(training_data.training_examples)

    # Next we add features.
    featurizer.train(training_data)
    featurizer.process(training_data.training_examples)

    # Train the classifier.
    classifier.train(training_data)

    # Make predictions.
    classifier.process(training_data.training_examples)

    # Check that the messages have been processed correctly
    for msg in training_data.training_examples:
        _, conf = msg.get("intent")["name"], msg.get("intent")["confidence"]
        # Confidence should be between 0 and 1.
        assert 0 < conf < 1
        ranking = msg.get("intent_ranking")
        assert {i["name"] for i in ranking} == {"greet", "goodbye"}
        # Confirm the sum of confidences is 1.0
        assert np.isclose(np.sum([i["confidence"] for i in ranking]), 1.0)

    classifier.persist()

    loaded_classifier = LogisticRegressionClassifier.load(
        {}, node_storage, node_resource, context
    )

    predicted = copy.copy(training_data)
    actual = copy.copy(training_data)
    loaded_messages = loaded_classifier.process(predicted.training_examples)
    trained_messages = classifier.process(actual.training_examples)
    for m1, m2 in zip(loaded_messages, trained_messages):
        assert m1.get("intent") == m2.get("intent")
