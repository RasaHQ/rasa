import numpy as np
import pytest

from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.constants import FEATURIZER_CLASS_ALIAS
from rasa.shared.nlu.constants import FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE, TEXT
from rasa.utils.tensorflow.constants import FEATURIZERS, SENTENCE, SEQUENCE, LABEL


@pytest.mark.parametrize(
    "pooling, features, expected",
    [
        (
            "mean",
            np.array([[0.5, 3, 0.4, 0.1], [0, 0, 0, 0], [0.5, 3, 0.4, 0.1]]),
            np.array([[0.5, 3, 0.4, 0.1]]),
        ),
        (
            "max",
            np.array([[1.0, 3.0, 0.0, 2.0], [4.0, 3.0, 1.0, 0.0]]),
            np.array([[4.0, 3.0, 1.0, 2.0]]),
        ),
        (
            "max",
            np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 0.0]]),
        ),
    ],
)
def test_calculate_cls_vector(pooling, features, expected):
    actual = DenseFeaturizer._calculate_sentence_features(features, pooling)

    assert np.all(actual == expected)


def test_flexible_nlu_pipeline():
    message = Message(data={TEXT: "This is a test message.", "intent": "test"})
    training_data = TrainingData([message, message, message, message, message])

    tokenizer = WhitespaceTokenizer()
    tokenizer.train(training_data)

    featurizer = CountVectorsFeaturizer(
        component_config={FEATURIZER_CLASS_ALIAS: "cvf_word"}
    )
    featurizer.train(training_data)

    featurizer = CountVectorsFeaturizer(
        component_config={
            FEATURIZER_CLASS_ALIAS: "cvf_char",
            "min_ngram": 1,
            "max_ngram": 3,
            "analyzer": "char_wb",
        }
    )
    featurizer.train(training_data)

    featurizer = LexicalSyntacticFeaturizer({})
    featurizer.train(training_data)

    # cvf word is also extracted for the intent
    origin_test_array = [
        "cvf_word",
        "cvf_word",
        "cvf_word",
        "cvf_char",
        "cvf_char",
        "LexicalSyntacticFeaturizer",
    ]
    type_test_array = [
        FEATURE_TYPE_SEQUENCE,
        FEATURE_TYPE_SENTENCE,
        FEATURE_TYPE_SEQUENCE,
        FEATURE_TYPE_SEQUENCE,
        FEATURE_TYPE_SENTENCE,
        FEATURE_TYPE_SEQUENCE,
    ]
    message_features_len = len(message.features)
    assert message_features_len == 6
    for i in range(message_features_len):
        assert message.features[i].origin == origin_test_array[i]
        assert message.features[i].type == type_test_array[i]

    sequence_feature_dim = (
        message.features[0].features.shape[1] + message.features[5].features.shape[1]
    )
    sentence_feature_dim = message.features[0].features.shape[1]

    classifier = DIETClassifier(
        component_config={FEATURIZERS: ["cvf_word", "LexicalSyntacticFeaturizer"]}
    )
    model_data = classifier.preprocess_train_data(training_data)

    model_data_tests = [
        model_data.get(TEXT).get(SEQUENCE),
        model_data.get(TEXT).get(SENTENCE),
        model_data.get(LABEL).get(SEQUENCE),
    ]
    shape_tests_results = [(5, sequence_feature_dim), (1, sentence_feature_dim), (1, 1)]
    for i in range(3):
        assert len(model_data_tests[i]) == 1
        assert model_data_tests[i][0][0].shape == shape_tests_results[i]
    assert model_data.get(LABEL).get(SENTENCE) is None
