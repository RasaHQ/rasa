from typing import Text, List, Any

import numpy as np
import pytest

from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.constants import SPACY_DOCS, TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT, INTENT, RESPONSE
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer


@pytest.mark.parametrize(
    "sentence, expected, labeled_tokens",
    [
        (
            "hey how are you today",
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [0],
        ),
        (
            "hey 456 how are you",
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            [1, 0],
        ),
        (
            "blah balh random eh",
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [],
        ),
        (
            "a 1 digit number",
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            [1, 1],
        ),
    ],
)
def test_regex_featurizer(sentence, expected, labeled_tokens, spacy_nlp):
    from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]
    ftr = RegexFeaturizer({}, known_patterns=patterns)

    # adds tokens to the message
    tokenizer = SpacyTokenizer({})
    message = Message(data={TEXT: sentence, RESPONSE: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    tokenizer.process(message)

    sequence_features, sentence_features = ftr._features_for_patterns(message, TEXT)
    assert np.allclose(sequence_features.toarray(), expected[:-1], atol=1e-10)
    assert np.allclose(sentence_features.toarray(), expected[-1], atol=1e-10)

    # the tokenizer should have added tokens
    assert len(message.get(TOKENS_NAMES[TEXT], [])) > 0
    # the number of regex matches on each token should match
    for i, token in enumerate(message.get(TOKENS_NAMES[TEXT])):
        token_matches = token.get("pattern").values()
        num_matches = sum(token_matches)
        assert num_matches == labeled_tokens.count(i)


@pytest.mark.parametrize(
    "sentence, expected, labeled_tokens",
    [
        (
            "lemonade and mapo tofu",
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
            [0.0, 2.0, 3.0],
        ),
        (
            "a cup of tea",
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [3.0],
        ),
        (
            "Is burrito my favorite food?",
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
            ],
            [1.0],
        ),
        ("I want club?mate", [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], [2.0]),
    ],
)
def test_lookup_tables(sentence, expected, labeled_tokens, spacy_nlp):
    from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer

    lookups = [
        {
            "name": "drinks",
            "elements": ["mojito", "lemonade", "sweet berry wine", "tea", "club?mate"],
        },
        {"name": "plates", "elements": "data/test/lookup_tables/plates.txt"},
    ]
    ftr = RegexFeaturizer()
    training_data = TrainingData()
    training_data.lookup_tables = lookups
    ftr.train(training_data)

    # adds tokens to the message
    component_config = {"name": "SpacyTokenizer"}
    tokenizer = SpacyTokenizer(component_config)
    message = Message(data={TEXT: sentence})
    message.set("text_spacy_doc", spacy_nlp(sentence))
    tokenizer.process(message)

    sequence_features, sentence_features = ftr._features_for_patterns(message, TEXT)
    assert np.allclose(sequence_features.toarray(), expected[:-1], atol=1e-10)
    assert np.allclose(sentence_features.toarray(), expected[-1], atol=1e-10)

    # the tokenizer should have added tokens
    assert len(message.get(TOKENS_NAMES[TEXT], [])) > 0
    # the number of regex matches on each token should match
    for i, token in enumerate(message.get(TOKENS_NAMES[TEXT])):
        token_matches = token.get("pattern").values()
        num_matches = sum(token_matches)
        assert num_matches == labeled_tokens.count(i)


@pytest.mark.parametrize(
    "sentence, expected, expected_cls",
    [
        ("hey how are you today", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ("hey 456 how are you", [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]),
        ("blah balh random eh", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ("a 1 digit number", [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]),
    ],
)
def test_regex_featurizer_no_sequence(sentence, expected, expected_cls, spacy_nlp):

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]
    ftr = RegexFeaturizer({}, known_patterns=patterns)

    # adds tokens to the message
    tokenizer = SpacyTokenizer()
    message = Message(data={TEXT: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    tokenizer.process(message)

    sequence_featrures, sentence_features = ftr._features_for_patterns(message, TEXT)
    assert np.allclose(sequence_featrures.toarray()[0], expected, atol=1e-10)
    assert np.allclose(sentence_features.toarray()[-1], expected_cls, atol=1e-10)


def test_regex_featurizer_train():

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]

    featurizer = RegexFeaturizer.create({}, RasaNLUModelConfig())

    sentence = "hey how are you today 19.12.2019 ?"
    message = Message(data={TEXT: sentence})
    message.set(RESPONSE, sentence)
    message.set(INTENT, "intent")
    WhitespaceTokenizer().train(TrainingData([message]))

    featurizer.train(
        TrainingData([message], regex_features=patterns), RasaNLUModelConfig()
    )

    expected = np.array([0, 1, 0])
    expected_cls = np.array([1, 1, 1])

    seq_vecs, sen_vec = message.get_sparse_features(TEXT, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vec:
        sen_vec = sen_vec.features

    assert (6, 3) == seq_vecs.shape
    assert (1, 3) == sen_vec.shape
    assert np.all(seq_vecs.toarray()[0] == expected)
    assert np.all(sen_vec.toarray()[-1] == expected_cls)

    seq_vecs, sen_vec = message.get_sparse_features(RESPONSE, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vec:
        sen_vec = sen_vec.features

    assert (6, 3) == seq_vecs.shape
    assert (1, 3) == sen_vec.shape
    assert np.all(seq_vecs.toarray()[0] == expected)
    assert np.all(sen_vec.toarray()[-1] == expected_cls)

    seq_vecs, sen_vec = message.get_sparse_features(INTENT, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vec:
        sen_vec = sen_vec.features

    assert seq_vecs is None
    assert sen_vec is None


@pytest.mark.parametrize(
    "sentence, sequence_vector, sentence_vector, case_sensitive",
    [
        ("Hey How are you today", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], True),
        ("Hey How are you today", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], False),
        ("Hey 456 How are you", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], True),
        ("Hey 456 How are you", [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], False),
    ],
)
def test_regex_featurizer_case_sensitive(
    sentence: Text,
    sequence_vector: List[float],
    sentence_vector: List[float],
    case_sensitive: bool,
    spacy_nlp: Any,
):

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]
    ftr = RegexFeaturizer({"case_sensitive": case_sensitive}, known_patterns=patterns)

    # adds tokens to the message
    tokenizer = SpacyTokenizer()
    message = Message(data={TEXT: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    tokenizer.process(message)

    sequence_featrures, sentence_features = ftr._features_for_patterns(message, TEXT)
    assert np.allclose(sequence_featrures.toarray()[0], sequence_vector, atol=1e-10)
    assert np.allclose(sentence_features.toarray()[-1], sentence_vector, atol=1e-10)
