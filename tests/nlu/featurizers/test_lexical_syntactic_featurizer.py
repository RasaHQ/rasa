import numpy as np
import pytest

import scipy.sparse
from typing import Text, List

from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import SPACY_DOCS, TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT, ACTION_TEXT


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

    train_message = Message(data={TEXT: sentence})
    test_message = Message(data={TEXT: sentence})

    WhitespaceTokenizer().process(train_message)
    WhitespaceTokenizer().process(test_message)

    featurizer.train(TrainingData([train_message]))

    featurizer.process(test_message)

    seq_vec, sen_vec = test_message.get_sparse_features(TEXT, [])
    if seq_vec:
        seq_vec = seq_vec.features
    if sen_vec:
        sen_vec = sen_vec.features

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

    train_message = Message(data={TEXT: sentence})
    test_message = Message(data={TEXT: sentence})

    WhitespaceTokenizer().process(train_message)
    WhitespaceTokenizer().process(test_message)

    featurizer.train(TrainingData([train_message]))

    featurizer.process(test_message)

    seq_vec, sen_vec = test_message.get_sparse_features(TEXT, [])
    if seq_vec:
        seq_vec = seq_vec.features
    if sen_vec:
        sen_vec = sen_vec.features

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

    train_message = Message(data={TEXT: sentence})
    test_message = Message(data={TEXT: sentence})

    train_message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    test_message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))

    SpacyTokenizer().process(train_message)
    SpacyTokenizer().process(test_message)

    featurizer.train(TrainingData([train_message]))

    featurizer.process(test_message)

    seq_vec, sen_vec = test_message.get_sparse_features(TEXT, [])
    if seq_vec:
        seq_vec = seq_vec.features
    if sen_vec:
        sen_vec = sen_vec.features

    assert isinstance(seq_vec, scipy.sparse.coo_matrix)
    assert sen_vec is None

    assert np.all(seq_vec.toarray() == expected)


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
def test_text_featurizer_using_pos_with_action_text(
    sentence: Text, expected: np.ndarray, spacy_nlp
):
    featurizer = LexicalSyntacticFeaturizer({"features": [["pos", "pos2"]]})

    train_message = Message(data={TEXT: sentence, ACTION_TEXT: sentence})
    test_message = Message(data={TEXT: sentence, ACTION_TEXT: sentence})

    train_message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    train_message.set(SPACY_DOCS[ACTION_TEXT], spacy_nlp(sentence))
    test_message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    test_message.set(SPACY_DOCS[ACTION_TEXT], spacy_nlp(sentence))

    SpacyTokenizer().process(train_message)
    SpacyTokenizer().process(test_message)

    featurizer.train(TrainingData([train_message]))
    # Checking that text is processed as expected
    featurizer.process(test_message)

    seq_vec, sen_vec = test_message.get_sparse_features(TEXT, [])
    if seq_vec:
        seq_vec = seq_vec.features
    if sen_vec:
        sen_vec = sen_vec.features

    assert isinstance(seq_vec, scipy.sparse.coo_matrix)
    assert sen_vec is None

    assert np.all(seq_vec.toarray() == expected)

    # Checking that action_text does not get processed and passing attribute works
    featurizer.process(test_message)

    seq_vec, sen_vec = test_message.get_sparse_features(ACTION_TEXT, [])

    assert seq_vec is None
    assert sen_vec is None


@pytest.mark.parametrize(
    "sentence, featurizer_config",
    [
        (
            "At 0700 the SUN is shining",
            [
                ["prefix2", "suffix1", "suffix3", "pos", "upper"],
                [
                    "low",
                    "title",
                    "prefix2",
                    "suffix1",
                    "suffix3",
                    "pos",
                    "upper",
                    "digit",
                ],
                ["prefix2", "suffix1", "suffix3", "pos", "upper"],
            ],
        )
    ],
)
def test_text_featurizer_using_feature_to_idx_dict(
    sentence: Text, featurizer_config: List[List[Text]], spacy_nlp
):
    featurizer = LexicalSyntacticFeaturizer({"features": featurizer_config})

    train_message = Message(data={TEXT: sentence, ACTION_TEXT: sentence})
    test_message = Message(data={TEXT: sentence, ACTION_TEXT: sentence})

    train_message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    train_message.set(SPACY_DOCS[ACTION_TEXT], spacy_nlp(sentence))
    test_message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    test_message.set(SPACY_DOCS[ACTION_TEXT], spacy_nlp(sentence))

    SpacyTokenizer().process(train_message)
    SpacyTokenizer().process(test_message)

    featurizer.train(TrainingData([train_message]))
    featurizer.process(test_message)

    seq_vec, sen_vec = test_message.get_sparse_features(TEXT, [])
    if seq_vec:
        seq_vec = seq_vec.features
    if sen_vec:
        sen_vec = sen_vec.features

    # Check that all requested features were indexed
    window_center = len(featurizer_config) // 2
    for index in range(len(featurizer_config)):
        for feature in featurizer_config[index]:
            assert (
                f"{index - window_center}:{feature}" in featurizer.feature_to_idx_dict
            )

    # Check that all indexed features were requested
    for feature in featurizer.feature_to_idx_dict.keys():
        i, feature_function = feature.split(":")
        index = int(i)
        assert index + window_center < len(featurizer_config)
        assert feature_function in featurizer_config[index + window_center]

    feature_vec = seq_vec.toarray()
    tokens = test_message.get(TOKENS_NAMES[TEXT])

    assert len(tokens) == feature_vec.shape[0]

    # For each word, check that the features we expect are the features we received
    for i, _ in enumerate(tokens):
        token_features = feature_vec[i, :]

        # For each expected feature, check that it's 1, then zero it out
        for j, position_features in enumerate(featurizer_config):
            position = i + j - window_center
            if position >= 0 and position < len(tokens):
                for feature_function in position_features:
                    feature_name = f"{j - window_center}:{feature_function}"
                    feature_value_str = str(
                        featurizer.function_dict[feature_function](tokens[position])
                    )
                    feature_idx = featurizer.feature_to_idx_dict[feature_name][
                        feature_value_str
                    ]
                    assert token_features[feature_idx] == 1.0
                    token_features[feature_idx] = 0.0

        # Since all expected features were zeroed out, result should be all zeros
        assert not np.any(token_features)
