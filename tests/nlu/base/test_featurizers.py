# -*- coding: utf-8 -
import numpy as np
import pytest

from rasa.nlu import training_data
from rasa.nlu.tokenizers import Token
from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.training_data import Message
from rasa.nlu.training_data import TrainingData
from rasa.nlu.config import RasaNLUModelConfig


@pytest.mark.parametrize(
    "sentence, expected",
    [
        (
            "hey how are you today",
            [-0.19649599, 0.32493639, -0.37408298, -0.10622784, 0.062756],
        )
    ],
)
def test_spacy_featurizer(sentence, expected, spacy_nlp):
    from rasa.nlu.featurizers import spacy_featurizer

    doc = spacy_nlp(sentence)
    vecs = spacy_featurizer.features_for_doc(doc)
    assert np.allclose(doc.vector[:5], expected, atol=1e-5)
    assert np.allclose(vecs, doc.vector, atol=1e-5)


def test_spacy_training_sample_alignment(spacy_nlp_component):
    from spacy.tokens import Doc

    m1 = Message.build(text="I have a feeling", intent="feeling")
    m2 = Message.build(text="", intent="feeling")
    m3 = Message.build(text="I am the last message", intent="feeling")
    td = TrainingData(training_examples=[m1, m2, m3])

    attribute_docs = spacy_nlp_component.docs_for_training_data(td)

    assert isinstance(attribute_docs["text"][0], Doc)
    assert isinstance(attribute_docs["text"][1], Doc)
    assert isinstance(attribute_docs["text"][2], Doc)

    assert [t.text for t in attribute_docs["text"][0]] == ["i", "have", "a", "feeling"]
    assert [t.text for t in attribute_docs["text"][1]] == []
    assert [t.text for t in attribute_docs["text"][2]] == [
        "i",
        "am",
        "the",
        "last",
        "message",
    ]


def test_spacy_intent_featurizer(spacy_nlp_component):
    from rasa.nlu.featurizers.spacy_featurizer import SpacyFeaturizer

    td = training_data.load_data("data/examples/rasa/demo-rasa.json")
    spacy_nlp_component.train(td, config=None)
    spacy_featurizer = SpacyFeaturizer()
    spacy_featurizer.train(td, config=None)

    intent_features_exist = np.array(
        [
            True if example.get("intent_features") is not None else False
            for example in td.intent_examples
        ]
    )

    # no intent features should have been set
    assert not any(intent_features_exist)


def test_mitie_featurizer(mitie_feature_extractor, default_config):
    from rasa.nlu.featurizers.mitie_featurizer import MitieFeaturizer

    mitie_component_config = {"name": "MitieFeaturizer"}
    ftr = MitieFeaturizer.create(mitie_component_config, RasaNLUModelConfig())
    sentence = "Hey how are you today"
    tokens = MitieTokenizer().tokenize(sentence)
    vecs = ftr.features_for_tokens(tokens, mitie_feature_extractor)
    expected = np.array([0.0, -4.4551446, 0.26073121, -1.46632245, -1.84205751])
    assert np.allclose(vecs[:5], expected, atol=1e-5)


def test_ngram_featurizer(spacy_nlp):
    from rasa.nlu.featurizers.ngram_featurizer import NGramFeaturizer

    ftr = NGramFeaturizer({"max_number_of_ngrams": 10})

    # ensures that during random sampling of the ngram CV we don't end up
    # with a one-class-split
    repetition_factor = 5

    greet = {"intent": "greet", "text_features": [0.5]}
    goodbye = {"intent": "goodbye", "text_features": [0.5]}
    labeled_sentences = [
        Message("heyheyheyhey", greet),
        Message("howdyheyhowdy", greet),
        Message("heyhey howdyheyhowdy", greet),
        Message("howdyheyhowdy heyhey", greet),
        Message("astalavistasista", goodbye),
        Message("astalavistasista sistala", goodbye),
        Message("sistala astalavistasista", goodbye),
    ] * repetition_factor

    for m in labeled_sentences:
        m.set("spacy_doc", spacy_nlp(m.text))

    ftr.min_intent_examples_for_ngram_classification = 2
    ftr.train_on_sentences(labeled_sentences)
    assert len(ftr.all_ngrams) > 0
    assert ftr.best_num_ngrams > 0


@pytest.mark.parametrize(
    "sentence, expected, labeled_tokens",
    [
        ("hey how are you today", [0.0, 1.0, 0.0], [0]),
        ("hey 456 how are you", [1.0, 1.0, 0.0], [1, 0]),
        ("blah balh random eh", [0.0, 0.0, 0.0], []),
        ("a 1 digit number", [1.0, 0.0, 1.0], [1, 1]),
    ],
)
def test_regex_featurizer(sentence, expected, labeled_tokens, spacy_nlp):
    from rasa.nlu.featurizers.regex_featurizer import RegexFeaturizer

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]
    ftr = RegexFeaturizer(known_patterns=patterns)

    # adds tokens to the message
    tokenizer = SpacyTokenizer()
    message = Message(sentence)
    message.set("spacy_doc", spacy_nlp(sentence))
    tokenizer.process(message)

    result = ftr.features_for_patterns(message)
    assert np.allclose(result, expected, atol=1e-10)

    # the tokenizer should have added tokens
    assert len(message.get("tokens", [])) > 0
    # the number of regex matches on each token should match
    for i, token in enumerate(message.get("tokens")):
        token_matches = token.get("pattern").values()
        num_matches = sum(token_matches)
        assert num_matches == labeled_tokens.count(i)


@pytest.mark.parametrize(
    "sentence, expected, labeled_tokens",
    [
        ("lemonade and mapo tofu", [1, 1], [0.0, 2.0, 3.0]),
        ("a cup of tea", [1, 0], [3.0]),
        ("Is burrito my favorite food?", [0, 1], [1.0]),
        ("I want club?mate", [1, 0], [2.0, 3.0]),
    ],
)
def test_lookup_tables(sentence, expected, labeled_tokens, spacy_nlp):
    from rasa.nlu.featurizers.regex_featurizer import RegexFeaturizer

    lookups = [
        {
            "name": "drinks",
            "elements": ["mojito", "lemonade", "sweet berry wine", "tea", "club?mate"],
        },
        {"name": "plates", "elements": "data/test/lookup_tables/plates.txt"},
    ]
    ftr = RegexFeaturizer(lookup_tables=lookups)

    # adds tokens to the message
    tokenizer = SpacyTokenizer()
    message = Message(sentence)
    message.set("spacy_doc", spacy_nlp(sentence))
    tokenizer.process(message)

    result = ftr.features_for_patterns(message)
    assert np.allclose(result, expected, atol=1e-10)

    # the tokenizer should have added tokens
    assert len(message.get("tokens", [])) > 0
    # the number of regex matches on each token should match
    for i, token in enumerate(message.get("tokens")):
        token_matches = token.get("pattern").values()
        num_matches = sum(token_matches)
        assert num_matches == labeled_tokens.count(i)


def test_spacy_featurizer_casing(spacy_nlp):
    from rasa.nlu.featurizers import spacy_featurizer

    # if this starts failing for the default model, we should think about
    # removing the lower casing the spacy nlp component does when it
    # retrieves vectors. For compressed spacy models (e.g. models
    # ending in _sm) this test will most likely fail.

    td = training_data.load_data("data/examples/rasa/demo-rasa.json")
    for e in td.intent_examples:
        doc = spacy_nlp(e.text)
        doc_capitalized = spacy_nlp(e.text.capitalize())

        vecs = spacy_featurizer.features_for_doc(doc)
        vecs_capitalized = spacy_featurizer.features_for_doc(doc_capitalized)

        assert np.allclose(
            vecs, vecs_capitalized, atol=1e-5
        ), "Vectors are unequal for texts '{}' and '{}'".format(
            e.text, e.text.capitalize()
        )


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("hello hello hello hello hello ", [5]),
        ("hello goodbye hello", [1, 2]),
        ("a b c d e f", [1, 1, 1, 1, 1, 1]),
        ("a 1 2", [2, 1]),
    ],
)
def test_count_vector_featurizer(sentence, expected):
    from rasa.nlu.featurizers.count_vectors_featurizer import CountVectorsFeaturizer

    ftr = CountVectorsFeaturizer({"token_pattern": r"(?u)\b\w+\b"})
    train_message = Message(sentence)
    # this is needed for a valid training example
    train_message.set("intent", "bla")
    data = TrainingData([train_message])
    ftr.train(data)

    test_message = Message(sentence)
    ftr.process(test_message)

    assert np.all(test_message.get("text_features") == expected)


@pytest.mark.parametrize(
    "sentence, intent, response, intent_features, response_features",
    [
        ("hello hello hello hello hello ", "greet", None, [1], None),
        ("hello goodbye hello", "greet", None, [1], None),
        ("a 1 2", "char", "char char", [1], [2]),
    ],
)
def test_count_vector_featurizer_attribute_featurization(
    sentence, intent, response, intent_features, response_features
):
    from rasa.nlu.featurizers.count_vectors_featurizer import CountVectorsFeaturizer

    ftr = CountVectorsFeaturizer({"token_pattern": r"(?u)\b\w+\b"})
    train_message = Message(sentence)

    # this is needed for a valid training example
    train_message.set("intent", intent)
    train_message.set("response", response)

    data = TrainingData([train_message])
    ftr.train(data)

    assert train_message.get("intent_features") == intent_features
    assert train_message.get("response_features") == response_features


@pytest.mark.parametrize(
    "sentence, intent, response, text_features, intent_features, response_features",
    [
        ("hello hello greet ", "greet", "hello", [1, 2], [1, 0], [0, 1]),
        (
            "I am fine",
            "acknowledge",
            "good",
            [0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ),
    ],
)
def test_count_vector_featurizer_shared_vocab(
    sentence, intent, response, text_features, intent_features, response_features
):
    from rasa.nlu.featurizers.count_vectors_featurizer import CountVectorsFeaturizer

    ftr = CountVectorsFeaturizer(
        {"token_pattern": r"(?u)\b\w+\b", "use_shared_vocab": True}
    )
    train_message = Message(sentence)

    # this is needed for a valid training example
    train_message.set("intent", intent)
    train_message.set("response", response)

    data = TrainingData([train_message])
    ftr.train(data)

    assert np.all(train_message.get("text_features") == text_features)
    assert np.all(train_message.get("intent_features") == intent_features)
    assert np.all(train_message.get("response_features") == response_features)


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("hello hello hello hello hello __OOV__", [1, 5]),
        ("hello goodbye hello __oov__", [1, 1, 2]),
        ("a b c d e f __oov__ __OOV__ __OOV__", [3, 1, 1, 1, 1, 1, 1]),
        ("__OOV__ a 1 2 __oov__ __OOV__", [2, 3, 1]),
    ],
)
def test_count_vector_featurizer_oov_token(sentence, expected):
    from rasa.nlu.featurizers.count_vectors_featurizer import CountVectorsFeaturizer

    ftr = CountVectorsFeaturizer(
        {"token_pattern": r"(?u)\b\w+\b", "OOV_token": "__oov__"}
    )
    train_message = Message(sentence)
    # this is needed for a valid training example
    train_message.set("intent", "bla")
    data = TrainingData([train_message])
    ftr.train(data)

    test_message = Message(sentence)
    ftr.process(test_message)

    assert np.all(test_message.get("text_features") == expected)


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("hello hello hello hello hello oov_word0", [1, 5]),
        ("hello goodbye hello oov_word0 OOV_word0", [2, 1, 2]),
        ("a b c d e f __oov__ OOV_word0 oov_word1", [3, 1, 1, 1, 1, 1, 1]),
        ("__OOV__ a 1 2 __oov__ OOV_word1", [2, 3, 1]),
    ],
)
def test_count_vector_featurizer_oov_words(sentence, expected):
    from rasa.nlu.featurizers.count_vectors_featurizer import CountVectorsFeaturizer

    ftr = CountVectorsFeaturizer(
        {
            "token_pattern": r"(?u)\b\w+\b",
            "OOV_token": "__oov__",
            "OOV_words": ["oov_word0", "OOV_word1"],
        }
    )
    train_message = Message(sentence)
    # this is needed for a valid training example
    train_message.set("intent", "bla")
    data = TrainingData([train_message])
    ftr.train(data)

    test_message = Message(sentence)
    ftr.process(test_message)

    assert np.all(test_message.get("text_features") == expected)


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (["hello", "hello", "hello", "hello", "hello"], [5]),
        (["你好", "你好", "你好", "你好", "你好"], [5]),  # test for unicode chars
        (["hello", "goodbye", "hello"], [1, 2]),
        # Note: order has changed in Chinese version of "hello" & "goodbye"
        (["你好", "再见", "你好"], [2, 1]),  # test for unicode chars
        (["a", "b", "c", "d", "e", "f"], [1, 1, 1, 1, 1, 1]),
        (["a", "1", "2"], [2, 1]),
    ],
)
def test_count_vector_featurizer_using_tokens(tokens, expected):
    from rasa.nlu.featurizers.count_vectors_featurizer import CountVectorsFeaturizer

    ftr = CountVectorsFeaturizer({"token_pattern": r"(?u)\b\w+\b"})

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

    assert np.all(test_message.get("text_features") == expected)


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("ababab", [3, 3, 3, 2]),
        ("ab ab ab", [2, 2, 3, 3, 3, 2]),
        ("abc", [1, 1, 1, 1, 1]),
    ],
)
def test_count_vector_featurizer_char(sentence, expected):
    from rasa.nlu.featurizers.count_vectors_featurizer import CountVectorsFeaturizer

    ftr = CountVectorsFeaturizer({"min_ngram": 1, "max_ngram": 2, "analyzer": "char"})
    train_message = Message(sentence)
    # this is needed for a valid training example
    train_message.set("intent", "bla")
    data = TrainingData([train_message])
    ftr.train(data)

    test_message = Message(sentence)
    ftr.process(test_message)

    assert np.all(test_message.get("text_features") == expected)


def test_count_vector_featurizer_persist_load(tmpdir):
    from rasa.nlu.featurizers.count_vectors_featurizer import CountVectorsFeaturizer

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
