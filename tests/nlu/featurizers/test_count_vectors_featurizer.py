import numpy as np
import pytest
import scipy.sparse

from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.training_data import Message
from rasa.nlu.training_data import TrainingData


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("hello hello hello hello hello ", [[1]]),
        ("hello goodbye hello", [[0, 1]]),
        ("a b c d e f", [[1, 0, 0, 0, 0, 0]]),
        ("a 1 2", [[0, 1]]),
    ],
)
def test_count_vector_featurizer(sentence, expected):
    from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
        CountVectorsFeaturizer,
    )

    ftr = CountVectorsFeaturizer(
        {"token_pattern": r"(?u)\b\w+\b", "return_sequence": True}
    )
    train_message = Message(sentence)
    # this is needed for a valid training example
    train_message.set("intent", "bla")
    data = TrainingData([train_message])
    ftr.train(data)

    test_message = Message(sentence)
    ftr.process(test_message)

    assert isinstance(test_message.get("text_sparse_features"), scipy.sparse.coo_matrix)

    actual = test_message.get("text_sparse_features").toarray()

    assert np.all(actual[0] == expected)


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("hello hello hello hello hello ", [[5]]),
        ("hello goodbye hello", [[1, 2]]),
        ("a b c d e f", [[1, 1, 1, 1, 1, 1]]),
        ("a 1 2", [[2, 1]]),
    ],
)
def test_count_vector_featurizer_no_sequence(sentence, expected):
    from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
        CountVectorsFeaturizer,
    )

    ftr = CountVectorsFeaturizer(
        {"token_pattern": r"(?u)\b\w+\b", "return_sequence": False}
    )
    train_message = Message(sentence)
    # this is needed for a valid training example
    train_message.set("intent", "bla")
    data = TrainingData([train_message])
    ftr.train(data)

    test_message = Message(sentence)
    ftr.process(test_message)

    assert isinstance(test_message.get("text_sparse_features"), scipy.sparse.coo_matrix)

    actual = test_message.get("text_sparse_features").toarray()

    assert np.all(actual == expected)


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
    from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
        CountVectorsFeaturizer,
    )

    ftr = CountVectorsFeaturizer(
        {"token_pattern": r"(?u)\b\w+\b", "return_sequence": True}
    )
    train_message = Message(sentence)

    # this is needed for a valid training example
    train_message.set("intent", intent)
    train_message.set("response", response)

    data = TrainingData([train_message])
    ftr.train(data)

    if intent_features:
        assert (
            train_message.get("intent_sparse_features").toarray()[0] == intent_features
        )
    else:
        assert train_message.get("intent_sparse_features") is None

    if response_features:
        assert (
            train_message.get("response_sparse_features").toarray()[0]
            == response_features
        )
    else:
        assert train_message.get("response_sparse_features") is None


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
    from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
        CountVectorsFeaturizer,
    )

    ftr = CountVectorsFeaturizer(
        {
            "token_pattern": r"(?u)\b\w+\b",
            "use_shared_vocab": True,
            "return_sequence": True,
        }
    )
    train_message = Message(sentence)

    # this is needed for a valid training example
    train_message.set("intent", intent)
    train_message.set("response", response)

    data = TrainingData([train_message])
    ftr.train(data)

    assert np.all(
        train_message.get("text_sparse_features").toarray()[0] == text_features
    )
    assert np.all(
        train_message.get("intent_sparse_features").toarray()[0] == intent_features
    )
    assert np.all(
        train_message.get("response_sparse_features").toarray()[0] == response_features
    )


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
    from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
        CountVectorsFeaturizer,
    )

    ftr = CountVectorsFeaturizer(
        {
            "token_pattern": r"(?u)\b\w+\b",
            "OOV_token": "__oov__",
            "return_sequence": True,
        }
    )
    train_message = Message(sentence)
    # this is needed for a valid training example
    train_message.set("intent", "bla")
    data = TrainingData([train_message])
    ftr.train(data)

    test_message = Message(sentence)
    ftr.process(test_message)

    assert np.all(test_message.get("text_sparse_features").toarray()[0] == expected)


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
    from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
        CountVectorsFeaturizer,
    )

    ftr = CountVectorsFeaturizer(
        {
            "token_pattern": r"(?u)\b\w+\b",
            "OOV_token": "__oov__",
            "OOV_words": ["oov_word0", "OOV_word1"],
            "return_sequence": True,
        }
    )
    train_message = Message(sentence)
    # this is needed for a valid training example
    train_message.set("intent", "bla")
    data = TrainingData([train_message])
    ftr.train(data)

    test_message = Message(sentence)
    ftr.process(test_message)

    assert np.all(test_message.get("text_sparse_features").toarray()[0] == expected)


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
    from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
        CountVectorsFeaturizer,
    )

    ftr = CountVectorsFeaturizer(
        {"token_pattern": r"(?u)\b\w+\b", "return_sequence": True}
    )

    # using empty string instead of real text string to make sure
    # count vector only can come from `tokens` feature.
    # using `message.text` can not get correct result

    tokens_feature = [Token(i, 0) for i in tokens]

    train_message = Message("")
    train_message.set("tokens", tokens_feature)
    # this is needed for a valid training example
    train_message.set("intent", "bla")
    data = TrainingData([train_message])

    ftr.train(data)

    test_message = Message("")
    test_message.set("tokens", tokens_feature)

    ftr.process(test_message)

    assert np.all(test_message.get("text_sparse_features").toarray()[0] == expected)


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("ababab", [[3, 3, 3, 2]]),
        ("ab ab ab", [[0, 0, 1, 1, 1, 0]]),
        ("abc", [[1, 1, 1, 1, 1]]),
    ],
)
def test_count_vector_featurizer_char(sentence, expected):
    from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
        CountVectorsFeaturizer,
    )

    ftr = CountVectorsFeaturizer(
        {"min_ngram": 1, "max_ngram": 2, "analyzer": "char", "return_sequence": True}
    )
    train_message = Message(sentence)
    # this is needed for a valid training example
    train_message.set("intent", "bla")
    data = TrainingData([train_message])
    ftr.train(data)

    test_message = Message(sentence)
    ftr.process(test_message)

    assert np.all(test_message.get("text_sparse_features").toarray()[0] == expected)


def test_count_vector_featurizer_persist_load(tmpdir):
    from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
        CountVectorsFeaturizer,
    )

    # set non default values to config
    config = {
        "analyzer": "char",
        "token_pattern": r"(?u)\b\w+\b",
        "strip_accents": "ascii",
        "stop_words": "stop",
        "min_df": 2,
        "max_df": 3,
        "min_ngram": 2,
        "max_ngram": 3,
        "max_features": 10,
        "lowercase": False,
        "return_sequence": True,
    }
    train_ftr = CountVectorsFeaturizer(config)

    sentence1 = "ababab 123 13xc лаомтгцу sfjv oö aà"
    sentence2 = "abababalidcn 123123 13xcdc лаомтгцу sfjv oö aà"
    train_message1 = Message(sentence1)
    train_message2 = Message(sentence2)

    # this is needed for a valid training example
    train_message1.set("intent", "bla")
    train_message2.set("intent", "bla")
    data = TrainingData([train_message1, train_message2])
    train_ftr.train(data)
    # persist featurizer
    file_dict = train_ftr.persist("ftr", tmpdir.strpath)
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
    test_ftr = CountVectorsFeaturizer.load(meta, tmpdir.strpath)
    test_vect_params = {
        attribute: vectorizer.get_params()
        for attribute, vectorizer in test_ftr.vectorizers.items()
    }

    assert train_vect_params == test_vect_params

    test_message1 = Message(sentence1)
    test_ftr.process(test_message1)
    test_message2 = Message(sentence2)
    test_ftr.process(test_message2)

    # check that train features and test features after loading are the same
    assert np.all(
        [
            train_message1.get("text_features") == test_message1.get("text_features"),
            train_message2.get("text_features") == test_message2.get("text_features"),
        ]
    )
