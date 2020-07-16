import numpy as np
import pytest

import scipy.sparse

from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.training_data import TrainingData
from rasa.nlu.constants import TEXT, SPACY_DOCS
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

    seq_vec, sen_vec = test_message.get_sparse_features(TEXT, [])

    assert isinstance(seq_vec, scipy.sparse.coo_matrix)
    assert sen_vec is None

    assert np.all(seq_vec.toarray() == expected_features[:-1])


@pytest.mark.parametrize(
    "sentence, expected",
    [("hello 123 hello 123 hello", [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])],
)
def test_text_featurizer_window_size(sentence, expected):
    featurizer = LexicalSyntacticFeaturizer(
        {"features": [["upper"], ["digit"], ["low"], ["digit"]]}
    )

    train_message = Message(sentence)
    test_message = Message(sentence)

    WhitespaceTokenizer().process(train_message)
    WhitespaceTokenizer().process(test_message)

    featurizer.train(TrainingData([train_message]))

    featurizer.process(test_message)

    seq_vec, sen_vec = test_message.get_sparse_features(TEXT, [])

    assert isinstance(seq_vec, scipy.sparse.coo_matrix)
    assert sen_vec is None

    assert np.all(seq_vec.toarray()[0] == expected)


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

    seq_vec, sen_vec = test_message.get_sparse_features(TEXT, [])

    assert isinstance(seq_vec, scipy.sparse.coo_matrix)
    assert sen_vec is None

    assert np.all(seq_vec.toarray() == expected)
