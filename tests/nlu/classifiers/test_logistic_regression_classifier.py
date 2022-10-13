import copy
import pytest
import pathlib
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
def featurizer_sparse(tmpdir):
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
            Message({"text": "I am so hungry", "intent": "hungry"}),
            Message({"text": "I want pizza", "intent": "hungry"}),
            Message({"text": "I want pizza, now!!", "intent": "hangry"}),
            Message({"text": "just gimme a pizza already", "intent": "hangry"}),
        ]
    )


def test_predictions_added(training_data, tmpdir, featurizer_sparse):
    """Checks if the sizes are appropriate."""
    # Set up classifier
    node_storage = LocalModelStorage(pathlib.Path(tmpdir))
    node_resource = Resource("classifier")
    context = ExecutionContext(node_storage, node_resource)
    ranking_length = 2
    classifier = LogisticRegressionClassifier(
        config={"ranking_length": ranking_length},
        name=context.node_name,
        resource=node_resource,
        model_storage=node_storage,
    )
    training_intents = training_data.intents
    # First we add tokens.
    tokeniser.process(training_data.training_examples)

    # Next we add features.
    featurizer_sparse.train(training_data)
    featurizer_sparse.process(training_data.training_examples)

    # Train the classifier.
    classifier.train(training_data)

    # Make predictions.
    classifier.process(training_data.training_examples)

    # Check that the messages have been processed correctly
    for msg in training_data.training_examples:
        intent = msg.get("intent")
        ranking = msg.get("intent_ranking")
        # check that first ranking element is the same as the winning intent
        assert (
            intent["name"] == ranking[0]["name"]
            and intent["confidence"] == ranking[0]["confidence"]
        )

        # check that ranking_length is adhered to
        if len(training_intents) > ranking_length:
            assert len(ranking) == ranking_length
        else:
            assert len(ranking) == len(training_intents)

        confidences = [r["confidence"] for r in ranking]
        assert all(
            [confidences[i] > confidences[i + 1] for i in range(len(confidences) - 1)]
        )

        # check that all ranking names are from training data
        assert all([r["name"] in training_intents for r in ranking])

        # confirm that all confidences are between 0 and 1
        assert all([0 <= r["confidence"] <= 1 for r in ranking])

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
