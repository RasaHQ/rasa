import numpy as np
import pytest

import scipy.sparse

from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.training_data import TrainingData
from rasa.nlu.constants import TEXT, SPARSE_FEATURE_NAMES, SPACY_DOCS
from rasa.nlu.training_data import Message


@pytest.mark.parametrize(
    "sentence, expected_features",
    [
        (
            "hello goodbye hello",
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0],
            ],
        ),
        (
            "a 1",
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
        ),
    ],
)
def test_text_featurizer(sentence, expected_features):
    featurizer = LexicalSyntacticFeaturizer(
        {
            "features": [
                ["BOS", "upper"],
                ["BOS", "EOS", "prefix2", "digit"],
                ["EOS", "low"],
            ]
        }
    )

    train_message = Message(sentence)
    test_message = Message(sentence)

    WhitespaceTokenizer().process(train_message)
    WhitespaceTokenizer().process(test_message)

    featurizer.train(TrainingData([train_message]))

    featurizer.process(test_message)

    assert isinstance(
        test_message.get(SPARSE_FEATURE_NAMES[TEXT]), scipy.sparse.coo_matrix
    )

    actual = test_message.get(SPARSE_FEATURE_NAMES[TEXT]).toarray()

    assert np.all(actual == expected_features)


@pytest.mark.parametrize(
    "sentence, expected, expected_cls",
    [
        (
            "hello 123 hello 123 hello",
            [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]],
            [[2.0, 2.0, 3.0, 2.0, 3.0, 2.0, 2.0]],
        )
    ],
)
def test_text_featurizer_window_size(sentence, expected, expected_cls):
    featurizer = LexicalSyntacticFeaturizer(
        {"features": [["upper"], ["digit"], ["low"], ["digit"]]}
    )

    train_message = Message(sentence)
    test_message = Message(sentence)

    WhitespaceTokenizer().process(train_message)
    WhitespaceTokenizer().process(test_message)

    featurizer.train(TrainingData([train_message]))

    featurizer.process(test_message)

    assert isinstance(
        test_message.get(SPARSE_FEATURE_NAMES[TEXT]), scipy.sparse.coo_matrix
    )

    actual = test_message.get(SPARSE_FEATURE_NAMES[TEXT]).toarray()

    assert np.all(actual[0] == expected)
    assert np.all(actual[-1] == expected_cls)


@pytest.mark.parametrize(
    "sentence, expected",
    [
        (
            "The sun is shining",
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            ],
        )
    ],
)
def test_text_featurizer_using_pos(sentence, expected, spacy_nlp):
    featurizer = LexicalSyntacticFeaturizer({"features": [["pos", "pos2"]]})

    train_message = Message(sentence)
    test_message = Message(sentence)

    train_message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    test_message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))

    SpacyTokenizer().process(train_message)
    SpacyTokenizer().process(test_message)

    featurizer.train(TrainingData([train_message]))

    featurizer.process(test_message)

    assert isinstance(
        test_message.get(SPARSE_FEATURE_NAMES[TEXT]), scipy.sparse.coo_matrix
    )

    actual = test_message.get(SPARSE_FEATURE_NAMES[TEXT]).toarray()

    assert np.all(actual == expected)
