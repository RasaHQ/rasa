import numpy as np
import pytest
import scipy.sparse

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.constants import TOKENS_NAMES, TEXT, INTENT, RESPONSE
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.training_data import Message
from rasa.nlu.training_data import TrainingData
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)


@pytest.mark.parametrize(
    "sentence, expected, expected_cls",
    [
        ("hello hello hello hello hello", [[1]], [[5]]),
        ("hello goodbye hello", [[0, 1]], [[1, 2]]),
        ("a b c d e f", [[1, 0, 0, 0, 0, 0]], [[1, 1, 1, 1, 1, 1]]),
        ("a 1 2", [[0, 1]], [[2, 1]]),
    ],
)
def test_count_vector_featurizer(sentence, expected, expected_cls):
    ftr = CountVectorsFeaturizer()

    train_message = Message(sentence)
    test_message = Message(sentence)

    WhitespaceTokenizer().process(train_message)
    WhitespaceTokenizer().process(test_message)

    ftr.train(TrainingData([train_message]))

    ftr.process(test_message)

    seq_vecs, sen_vecs = test_message.get_sparse_features(TEXT, [])

    assert isinstance(seq_vecs, scipy.sparse.coo_matrix)
    assert isinstance(sen_vecs, scipy.sparse.coo_matrix)

    actual_seq_vecs = seq_vecs.toarray()
    actual_sen_vecs = sen_vecs.toarray()

    assert np.all(actual_seq_vecs[0] == expected)
    assert np.all(actual_sen_vecs[-1] == expected_cls)


@pytest.mark.parametrize(
    "sentence, intent, response, intent_features, response_features",
    [("hello", "greet", None, [[1]], None), ("hello", "greet", "hi", [[1]], [[1]])],
)
def test_count_vector_featurizer_response_attribute_featurization(
    sentence, intent, response, intent_features, response_features
):
    ftr = CountVectorsFeaturizer()
    tk = WhitespaceTokenizer()

    train_message = Message(sentence)
    # this is needed for a valid training example
    train_message.set(INTENT, intent)
    train_message.set(RESPONSE, response)

    # add a second example that has some response, so that the vocabulary for
    # response exists
    second_message = Message("hello")
    second_message.set(RESPONSE, "hi")
    second_message.set(INTENT, "greet")

    data = TrainingData([train_message, second_message])

    tk.train(data)
    ftr.train(data)

    intent_seq_vecs, intent_sen_vecs = train_message.get_sparse_features(INTENT, [])
    response_seq_vecs, response_sen_vecs = train_message.get_sparse_features(
        RESPONSE, []
    )

    if intent_features:
        assert intent_seq_vecs.toarray()[0] == intent_features
        assert intent_sen_vecs is None
    else:
        assert intent_seq_vecs is None
        assert intent_sen_vecs is None

    if response_features:
        assert response_seq_vecs.toarray()[0] == response_features
        assert response_sen_vecs is not None
    else:
        assert response_seq_vecs is None
        assert response_sen_vecs is None


@pytest.mark.parametrize(
    "sentence, intent, response, intent_features, response_features",
    [
        ("hello hello hello hello hello ", "greet", None, [[1]], None),
        ("hello goodbye hello", "greet", None, [[1]], None),
        ("a 1 2", "char", "char char", [[1]], [[1]]),
    ],
)
def test_count_vector_featurizer_attribute_featurization(
    sentence, intent, response, intent_features, response_features
):
    ftr = CountVectorsFeaturizer()
    tk = WhitespaceTokenizer()

    train_message = Message(sentence)
    # this is needed for a valid training example
    train_message.set(INTENT, intent)
    train_message.set(RESPONSE, response)

    data = TrainingData([train_message])

    tk.train(data)
    ftr.train(data)

    intent_seq_vecs, intent_sen_vecs = train_message.get_sparse_features(INTENT, [])
    response_seq_vecs, response_sen_vecs = train_message.get_sparse_features(
        RESPONSE, []
    )
    if intent_features:
        assert intent_seq_vecs.toarray()[0] == intent_features
        assert intent_sen_vecs is None
    else:
        assert intent_seq_vecs is None
        assert intent_sen_vecs is None

    if response_features:
        assert response_seq_vecs.toarray()[0] == response_features
        assert response_sen_vecs is not None
    else:
        assert response_seq_vecs is None
        assert response_sen_vecs is None


@pytest.mark.parametrize(
    "sentence, intent, response, text_features, intent_features, response_features",
    [
        ("hello hello greet ", "greet", "hello", [[0, 1]], [[1, 0]], [[0, 1]]),
        (
            "I am fine",
            "acknowledge",
            "good",
            [[0, 0, 0, 0, 1]],
            [[1, 0, 0, 0, 0]],
            [[0, 0, 0, 1, 0]],
        ),
    ],
)
def test_count_vector_featurizer_shared_vocab(
    sentence, intent, response, text_features, intent_features, response_features
):
    ftr = CountVectorsFeaturizer({"use_shared_vocab": True})
    tk = WhitespaceTokenizer()

    train_message = Message(sentence)
    # this is needed for a valid training example
    train_message.set(INTENT, intent)
    train_message.set(RESPONSE, response)

    data = TrainingData([train_message])
    tk.train(data)
    ftr.train(data)

    seq_vec, sen_vec = train_message.get_sparse_features(TEXT, [])
    assert np.all(seq_vec.toarray()[0] == text_features)
    assert sen_vec is not None
    seq_vec, sen_vec = train_message.get_sparse_features(INTENT, [])
    assert np.all(seq_vec.toarray()[0] == intent_features)
    assert sen_vec is None
    seq_vec, sen_vec = train_message.get_sparse_features(RESPONSE, [])
    assert np.all(seq_vec.toarray()[0] == response_features)
    assert sen_vec is not None


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("hello hello hello hello hello __OOV__", [[0, 1]]),
        ("hello goodbye hello __oov__", [[0, 0, 1]]),
        ("a b c d e f __oov__ __OOV__ __OOV__", [[0, 1, 0, 0, 0, 0, 0]]),
        ("__OOV__ a 1 2 __oov__ __OOV__", [[0, 1, 0]]),
    ],
)
def test_count_vector_featurizer_oov_token(sentence, expected):
    ftr = CountVectorsFeaturizer({"OOV_token": "__oov__"})
    train_message = Message(sentence)
    WhitespaceTokenizer().process(train_message)

    data = TrainingData([train_message])
    ftr.train(data)

    test_message = Message(sentence)
    ftr.process(test_message)

    seq_vec, sen_vec = train_message.get_sparse_features(TEXT, [])
    assert np.all(seq_vec.toarray()[0] == expected)
    assert sen_vec is not None


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("hello hello hello hello hello oov_word0", [[0, 1]]),
        ("hello goodbye hello oov_word0 OOV_word0", [[0, 0, 1]]),
        ("a b c d e f __oov__ OOV_word0 oov_word1", [[0, 1, 0, 0, 0, 0, 0]]),
        ("__OOV__ a 1 2 __oov__ OOV_word1", [[0, 1, 0]]),
    ],
)
def test_count_vector_featurizer_oov_words(sentence, expected):

    ftr = CountVectorsFeaturizer(
        {"OOV_token": "__oov__", "OOV_words": ["oov_word0", "OOV_word1"]}
    )
    train_message = Message(sentence)
    WhitespaceTokenizer().process(train_message)

    data = TrainingData([train_message])
    ftr.train(data)

    test_message = Message(sentence)
    ftr.process(test_message)

    seq_vec, sen_vec = train_message.get_sparse_features(TEXT, [])
    assert np.all(seq_vec.toarray()[0] == expected)
    assert sen_vec is not None


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (["hello", "hello", "hello", "hello", "hello"], [[1]]),
        (["你好", "你好", "你好", "你好", "你好"], [[1]]),  # test for unicode chars
        (["hello", "goodbye", "hello"], [[0, 1]]),
        # Note: order has changed in Chinese version of "hello" & "goodbye"
        (["你好", "再见", "你好"], [[1, 0]]),  # test for unicode chars
        (["a", "b", "c", "d", "e", "f"], [[1, 0, 0, 0, 0, 0]]),
        (["a", "1", "2"], [[0, 1]]),
    ],
)
def test_count_vector_featurizer_using_tokens(tokens, expected):

    ftr = CountVectorsFeaturizer()

    # using empty string instead of real text string to make sure
    # count vector only can come from `tokens` feature.
    # using `message.text` can not get correct result

    tokens_feature = [Token(i, 0) for i in tokens]

    train_message = Message("")
    train_message.set(TOKENS_NAMES[TEXT], tokens_feature)

    data = TrainingData([train_message])

    ftr.train(data)

    test_message = Message("")
    test_message.set(TOKENS_NAMES[TEXT], tokens_feature)

    ftr.process(test_message)

    seq_vec, sen_vec = train_message.get_sparse_features(TEXT, [])
    assert np.all(seq_vec.toarray()[0] == expected)
    assert sen_vec is not None


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("ababab", [[3, 3, 3, 2]]),
        ("ab ab ab", [[0, 0, 1, 1, 1, 0]]),
        ("abc", [[1, 1, 1, 1, 1]]),
    ],
)
def test_count_vector_featurizer_char(sentence, expected):
    ftr = CountVectorsFeaturizer({"min_ngram": 1, "max_ngram": 2, "analyzer": "char"})

    train_message = Message(sentence)
    WhitespaceTokenizer().process(train_message)

    data = TrainingData([train_message])
    ftr.train(data)

    test_message = Message(sentence)
    WhitespaceTokenizer().process(test_message)
    ftr.process(test_message)

    seq_vec, sen_vec = train_message.get_sparse_features(TEXT, [])
    assert np.all(seq_vec.toarray()[0] == expected)
    assert sen_vec is not None


def test_count_vector_featurizer_persist_load(tmp_path):

    # set non default values to config
    config = {
        "analyzer": "char",
        "strip_accents": "ascii",
        "stop_words": "stop",
        "min_df": 2,
        "max_df": 3,
        "min_ngram": 2,
        "max_ngram": 3,
        "max_features": 10,
        "lowercase": False,
    }
    train_ftr = CountVectorsFeaturizer(config)

    sentence1 = "ababab 123 13xc лаомтгцу sfjv oö aà"
    sentence2 = "abababalidcn 123123 13xcdc лаомтгцу sfjv oö aà"
    train_message1 = Message(sentence1)
    train_message2 = Message(sentence2)

    data = TrainingData([train_message1, train_message2])
    train_ftr.train(data)

    # persist featurizer
    file_dict = train_ftr.persist("ftr", str(tmp_path))
    train_vect_params = {
        attribute: vectorizer.get_params()
        for attribute, vectorizer in train_ftr.vectorizers.items()
    }

    # add trained vocabulary to vectorizer params
    for attribute, attribute_vect_params in train_vect_params.items():
        if hasattr(train_ftr.vectorizers[attribute], "vocabulary_"):
            train_vect_params[attribute].update(
                {"vocabulary": train_ftr.vectorizers[attribute].vocabulary_}
            )

    # load featurizer
    meta = train_ftr.component_config.copy()
    meta.update(file_dict)
    test_ftr = CountVectorsFeaturizer.load(meta, str(tmp_path))
    test_vect_params = {
        attribute: vectorizer.get_params()
        for attribute, vectorizer in test_ftr.vectorizers.items()
    }

    assert train_vect_params == test_vect_params

    # check if vocaculary was loaded correctly
    assert hasattr(test_ftr.vectorizers[TEXT], "vocabulary_")

    test_message1 = Message(sentence1)
    test_ftr.process(test_message1)
    test_message2 = Message(sentence2)
    test_ftr.process(test_message2)

    test_seq_vec_1, test_sen_vec_1 = test_message1.get_sparse_features(TEXT, [])
    train_seq_vec_1, train_sen_vec_1 = train_message1.get_sparse_features(TEXT, [])
    test_seq_vec_2, test_sen_vec_2 = test_message2.get_sparse_features(TEXT, [])
    train_seq_vec_2, train_sen_vec_2 = train_message2.get_sparse_features(TEXT, [])

    # check that train features and test features after loading are the same
    assert np.all(test_seq_vec_1.toarray() == train_seq_vec_1.toarray())
    assert np.all(test_sen_vec_1.toarray() == train_sen_vec_1.toarray())
    assert np.all(test_seq_vec_2.toarray() == train_seq_vec_2.toarray())
    assert np.all(test_sen_vec_2.toarray() == train_sen_vec_2.toarray())


def test_count_vectors_featurizer_train():

    featurizer = CountVectorsFeaturizer.create({}, RasaNLUModelConfig())

    sentence = "Hey how are you today ?"
    message = Message(sentence)
    message.set(RESPONSE, sentence)
    message.set(INTENT, "intent")
    WhitespaceTokenizer().train(TrainingData([message]))

    featurizer.train(TrainingData([message]), RasaNLUModelConfig())

    expected = np.array([0, 1, 0, 0, 0])
    expected_cls = np.array([1, 1, 1, 1, 1])

    seq_vec, sen_vec = message.get_sparse_features(TEXT, [])

    assert (5, 5) == seq_vec.shape
    assert (1, 5) == sen_vec.shape
    assert np.all(seq_vec.toarray()[0] == expected)
    assert np.all(sen_vec.toarray()[-1] == expected_cls)

    seq_vec, sen_vec = message.get_sparse_features(RESPONSE, [])

    assert (5, 5) == seq_vec.shape
    assert (1, 5) == sen_vec.shape
    assert np.all(seq_vec.toarray()[0] == expected)
    assert np.all(sen_vec.toarray()[-1] == expected_cls)

    seq_vec, sen_vec = message.get_sparse_features(INTENT, [])

    assert sen_vec is None
    assert (1, 1) == seq_vec.shape
    assert np.all(seq_vec.toarray()[0] == np.array([1]))
