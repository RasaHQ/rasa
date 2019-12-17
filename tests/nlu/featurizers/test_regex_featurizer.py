import numpy as np
import pytest

from rasa.nlu.constants import TEXT_ATTRIBUTE, RESPONSE_ATTRIBUTE, SPACY_DOCS
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
            ],
            [1, 0],
        ),
        (
            "blah balh random eh",
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [],
        ),
        (
            "a 1 digit number",
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
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
    ftr = RegexFeaturizer({"return_sequence": True}, known_patterns=patterns)

    # adds tokens to the message
    tokenizer = SpacyTokenizer({"use_cls_token": False})
    message = Message(sentence, data={RESPONSE_ATTRIBUTE: sentence})
    message.set(SPACY_DOCS[TEXT_ATTRIBUTE], spacy_nlp(sentence))
    tokenizer.process(message)

    result = ftr._features_for_patterns(message, TEXT_ATTRIBUTE)
    assert np.allclose(result.toarray(), expected, atol=1e-10)

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
        (
            "lemonade and mapo tofu",
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            [0.0, 2.0, 3.0],
        ),
        ("a cup of tea", [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]], [3.0]),
        (
            "Is burrito my favorite food?",
            [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [1.0],
        ),
        ("I want club?mate", [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]], [2.0, 3.0]),
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
    ftr = RegexFeaturizer({"return_sequence": True}, lookup_tables=lookups)

    # adds tokens to the message
    component_config = {"name": "SpacyTokenizer", "use_cls_token": False}
    tokenizer = SpacyTokenizer(component_config)
    message = Message(sentence)
    message.set("spacy_doc", spacy_nlp(sentence))
    tokenizer.process(message)

    result = ftr._features_for_patterns(message, TEXT_ATTRIBUTE)
    assert np.allclose(result.toarray(), expected, atol=1e-10)

    # the tokenizer should have added tokens
    assert len(message.get("tokens", [])) > 0
    # the number of regex matches on each token should match
    for i, token in enumerate(message.get("tokens")):
        token_matches = token.get("pattern").values()
        num_matches = sum(token_matches)
        assert num_matches == labeled_tokens.count(i)


@pytest.mark.parametrize(
    "sentence, expected ",
    [
        ("hey how are you today", [0.0, 1.0, 0.0]),
        ("hey 456 how are you", [1.0, 1.0, 0.0]),
        ("blah balh random eh", [0.0, 0.0, 0.0]),
        ("a 1 digit number", [1.0, 0.0, 1.0]),
    ],
)
def test_regex_featurizer_no_sequence(sentence, expected, spacy_nlp):
    from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]
    ftr = RegexFeaturizer({"return_sequence": False}, known_patterns=patterns)

    # adds tokens to the message
    tokenizer = SpacyTokenizer()
    message = Message(sentence)
    message.set("spacy_doc", spacy_nlp(sentence))
    tokenizer.process(message)

    result = ftr._features_for_patterns(message, TEXT_ATTRIBUTE)
    assert np.allclose(result.toarray()[0], expected, atol=1e-10)
