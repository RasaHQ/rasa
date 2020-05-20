import numpy as np
import pytest

from rasa.nlu.training_data import TrainingData
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.constants import (
    TEXT,
    RESPONSE,
    SPACY_DOCS,
    TOKENS_NAMES,
    INTENT,
    SPARSE_FEATURE_NAMES,
)
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.training_data import Message


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
    message = Message(sentence, data={RESPONSE: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    tokenizer.process(message)

    result = ftr._features_for_patterns(message, TEXT)
    assert np.allclose(result.toarray(), expected, atol=1e-10)

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
    ftr = RegexFeaturizer({}, lookup_tables=lookups)

    # adds tokens to the message
    component_config = {"name": "SpacyTokenizer"}
    tokenizer = SpacyTokenizer(component_config)
    message = Message(sentence)
    message.set("text_spacy_doc", spacy_nlp(sentence))
    tokenizer.process(message)

    result = ftr._features_for_patterns(message, TEXT)
    assert np.allclose(result.toarray(), expected, atol=1e-10)

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
    message = Message(sentence)
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    tokenizer.process(message)

    result = ftr._features_for_patterns(message, TEXT)
    assert np.allclose(result.toarray()[0], expected, atol=1e-10)
    assert np.allclose(result.toarray()[-1], expected_cls, atol=1e-10)


def test_regex_featurizer_train():

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]

    featurizer = RegexFeaturizer.create({}, RasaNLUModelConfig())

    sentence = "hey how are you today 19.12.2019 ?"
    message = Message(sentence)
    message.set(RESPONSE, sentence)
    message.set(INTENT, "intent")
    WhitespaceTokenizer().train(TrainingData([message]))

    featurizer.train(
        TrainingData([message], regex_features=patterns), RasaNLUModelConfig()
    )

    expected = np.array([0, 1, 0])
    expected_cls = np.array([1, 1, 1])

    vecs = message.get(SPARSE_FEATURE_NAMES[TEXT])

    assert (7, 3) == vecs.shape
    assert np.all(vecs.toarray()[0] == expected)
    assert np.all(vecs.toarray()[-1] == expected_cls)

    vecs = message.get(SPARSE_FEATURE_NAMES[RESPONSE])

    assert (7, 3) == vecs.shape
    assert np.all(vecs.toarray()[0] == expected)
    assert np.all(vecs.toarray()[-1] == expected_cls)

    vecs = message.get(SPARSE_FEATURE_NAMES[INTENT])

    assert vecs is None
