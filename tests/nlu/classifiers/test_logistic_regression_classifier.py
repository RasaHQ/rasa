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
from rasa_nlu_examples.classifiers import LogisticRegressionClassifier


@pytest.fixture
def classifier(tmpdir):
    """Generate a classifier for tests."""
    node_storage = LocalModelStorage(pathlib.Path(tmpdir))
    node_resource = Resource("sparse_feat")
    context = ExecutionContext(node_storage, node_resource)
    return LogisticRegressionClassifier(
        config=LogisticRegressionClassifier.get_default_config(),
        name=context.node_name,
        resource=node_resource,
        model_storage=node_storage,
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


def test_predictions_added(classifier, featurizer):
    """Checks if the sizes are appropriate."""
    # Create training data.
    training_data = TrainingData(
        [
            Message({"text": "hello", "intent": "greet"}),
            Message({"text": "hi there", "intent": "greet"}),
            Message({"text": "ciao", "intent": "goodbye"}),
            Message({"text": "bye", "intent": "goodbye"}),
        ]
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
        name, conf = msg.get("intent")["name"], msg.get("intent")["confidence"]
        assert name in ["greet", "goodbye"]
        assert 0 < conf
        assert conf < 1
        ranking = msg.get("intent_ranking")
        assert {i["name"] for i in ranking} == {"greet", "goodbye"}
        assert np.isclose(np.sum([i["confidence"] for i in ranking]), 1.0)
