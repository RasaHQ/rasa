from typing import Optional, Text, List

import pytest
import numpy as np
import scipy.sparse

from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import (
    TEXT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    ACTION_TEXT,
    ACTION_NAME,
    INTENT,
    RESPONSE,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize(
    "features, attribute, featurizers, expected_seq_features, expected_sen_features",
    [
        (None, TEXT, [], None, None),
        (
            [Features(np.array([1, 1, 0]), FEATURE_TYPE_SEQUENCE, TEXT, "test")],
            TEXT,
            [],
            [1, 1, 0],
            None,
        ),
        (
            [
                Features(np.array([1, 1, 0]), FEATURE_TYPE_SEQUENCE, TEXT, "c2"),
                Features(np.array([1, 2, 2]), FEATURE_TYPE_SENTENCE, TEXT, "c1"),
                Features(np.array([1, 2, 1]), FEATURE_TYPE_SEQUENCE, TEXT, "c1"),
            ],
            TEXT,
            [],
            [1, 1, 0, 1, 2, 1],
            [1, 2, 2],
        ),
        (
            [
                Features(np.array([1, 1, 0]), FEATURE_TYPE_SEQUENCE, TEXT, "c1"),
                Features(np.array([1, 2, 1]), FEATURE_TYPE_SENTENCE, TEXT, "test"),
                Features(np.array([1, 1, 1]), FEATURE_TYPE_SEQUENCE, TEXT, "test"),
            ],
            TEXT,
            ["c1"],
            [1, 1, 0],
            None,
        ),
    ],
)
def test_get_dense_features(
    features: Optional[List[Features]],
    attribute: Text,
    featurizers: List[Text],
    expected_seq_features: Optional[List[Features]],
    expected_sen_features: Optional[List[Features]],
):

    message = Message(data={TEXT: "This is a test sentence."}, features=features)

    actual_seq_features, actual_sen_features = message.get_dense_features(
        attribute, featurizers
    )
    if actual_seq_features:
        actual_seq_features = actual_seq_features.features
    if actual_sen_features:
        actual_sen_features = actual_sen_features.features

    assert np.all(actual_sen_features == expected_sen_features)
    assert np.all(actual_seq_features == expected_seq_features)


@pytest.mark.parametrize(
    "features, attribute, featurizers, expected_seq_features, expected_sen_features",
    [
        (None, TEXT, [], None, None),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "test",
                )
            ],
            TEXT,
            [],
            [1, 1, 0],
            None,
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c2",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 2, 2]),
                    FEATURE_TYPE_SENTENCE,
                    TEXT,
                    "c1",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 2, 1]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c1",
                ),
            ],
            TEXT,
            [],
            [1, 1, 0, 1, 2, 1],
            [1, 2, 2],
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c1",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 2, 1]),
                    FEATURE_TYPE_SENTENCE,
                    TEXT,
                    "test",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 1, 1]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "test",
                ),
            ],
            TEXT,
            ["c1"],
            [1, 1, 0],
            None,
        ),
    ],
)
def test_get_sparse_features(
    features: Optional[List[Features]],
    attribute: Text,
    featurizers: List[Text],
    expected_seq_features: Optional[List[Features]],
    expected_sen_features: Optional[List[Features]],
):
    message = Message(data={TEXT: "This is a test sentence."}, features=features)

    actual_seq_features, actual_sen_features = message.get_sparse_features(
        attribute, featurizers
    )
    if actual_seq_features:
        actual_seq_features = actual_seq_features.features
    if actual_sen_features:
        actual_sen_features = actual_sen_features.features

    if expected_seq_features is None:
        assert actual_seq_features is None
    else:
        assert actual_seq_features is not None
        assert np.all(actual_seq_features.toarray() == expected_seq_features)

    if expected_sen_features is None:
        assert actual_sen_features is None
    else:
        assert actual_sen_features is not None
        assert np.all(actual_sen_features.toarray() == expected_sen_features)


@pytest.mark.parametrize(
    "features, attribute, featurizers, "
    "expected_sequence_sizes, expected_sentence_sizes",
    [
        (None, TEXT, [], [], []),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "test",
                )
            ],
            TEXT,
            [],
            [3],
            [],
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c2",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 2, 2]),
                    FEATURE_TYPE_SENTENCE,
                    TEXT,
                    "c1",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 2, 1]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c1",
                ),
            ],
            TEXT,
            [],
            [3, 3],
            [3],
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c1",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 2, 1]),
                    FEATURE_TYPE_SENTENCE,
                    TEXT,
                    "test",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 1, 1]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "test",
                ),
            ],
            TEXT,
            ["c1"],
            [3],
            [],
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c1",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 2, 1]),
                    FEATURE_TYPE_SENTENCE,
                    TEXT,
                    "test",
                ),
                Features(np.array([1, 1, 0]), FEATURE_TYPE_SEQUENCE, TEXT, "c1"),
                Features(np.array([1, 2, 1]), FEATURE_TYPE_SENTENCE, TEXT, "test"),
            ],
            TEXT,
            ["c1"],
            [4],
            [],
        ),
    ],
)
def test_get_sparse_feature_sizes(
    features: Optional[List[Features]],
    attribute: Text,
    featurizers: List[Text],
    expected_sequence_sizes: List[int],
    expected_sentence_sizes: List[int],
):
    message = Message(data={TEXT: "This is a test sentence."}, features=features)
    feature_sizes = message.get_sparse_feature_sizes(attribute, featurizers)

    assert feature_sizes[FEATURE_TYPE_SEQUENCE] == expected_sequence_sizes
    assert feature_sizes[FEATURE_TYPE_SENTENCE] == expected_sentence_sizes


@pytest.mark.parametrize(
    "features, attribute, featurizers, expected",
    [
        (None, TEXT, [], False),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "test",
                )
            ],
            TEXT,
            [],
            True,
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c2",
                ),
                Features(np.ndarray([1, 2, 2]), FEATURE_TYPE_SEQUENCE, TEXT, "c1"),
            ],
            TEXT,
            [],
            True,
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c2",
                ),
                Features(np.ndarray([1, 2, 2]), FEATURE_TYPE_SEQUENCE, TEXT, "c1"),
            ],
            TEXT,
            ["c1"],
            True,
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c2",
                ),
                Features(np.ndarray([1, 2, 2]), FEATURE_TYPE_SEQUENCE, TEXT, "c1"),
            ],
            TEXT,
            ["other"],
            False,
        ),
    ],
)
def test_features_present(
    features: Optional[List[Features]],
    attribute: Text,
    featurizers: List[Text],
    expected: bool,
):
    message = Message(data={TEXT: "This is a test sentence."}, features=features)

    actual = message.features_present(attribute, featurizers)

    assert actual == expected


@pytest.mark.parametrize(
    "message, result",
    [
        (Message({INTENT: "intent", TEXT: "text"}), False),
        (Message({RESPONSE: "response", TEXT: "text"}), False),
        (Message({INTENT: "intent"}), True),
        (Message({ACTION_TEXT: "action text"}), True),
        (Message({ACTION_NAME: "action name"}), True),
        (Message({TEXT: "text"}), True),
    ],
)
def test_is_core_or_domain_message(message: Message, result: bool):
    assert result == message.is_core_or_domain_message()


def test_add_diagnostic_data_with_repeated_component_raises_warning():
    message = Message()
    message.add_diagnostic_data("a", {})
    with pytest.warns(UserWarning):
        message.add_diagnostic_data("a", {})


def test_message_fingerprint_includes_data_and_features(
    whitespace_tokenizer: WhitespaceTokenizer,
):
    message = Message(data={TEXT: "This is a test sentence."})
    fp1 = message.fingerprint()
    whitespace_tokenizer.process([message])
    fp2 = message.fingerprint()

    assert fp1 != fp2

    message.add_features(
        Features(scipy.sparse.csr_matrix([1, 1, 0]), FEATURE_TYPE_SEQUENCE, TEXT, "c2")
    )

    fp3 = message.fingerprint()
    assert fp2 != fp3

    message.add_features(
        Features(np.ndarray([1, 2, 2]), FEATURE_TYPE_SEQUENCE, TEXT, "c1")
    )

    fp4 = message.fingerprint()

    assert fp3 != fp4

    assert len({fp1, fp2, fp3, fp4}) == 4


def test_message_fingerprint_is_recalculated_after_setting_data():
    message = Message(data={TEXT: "This is a test sentence."})
    fp1 = message.fingerprint()
    message.set(INTENT, "test")
    fp2 = message.fingerprint()
    assert fp1 != fp2


def test_message_fingerprint_is_recalculated_after_adding_diagnostics_data():
    message = Message(data={TEXT: "This is a test sentence."})
    fp1 = message.fingerprint()
    message.add_diagnostic_data("origin", "test")
    fp2 = message.fingerprint()
    assert fp1 != fp2
