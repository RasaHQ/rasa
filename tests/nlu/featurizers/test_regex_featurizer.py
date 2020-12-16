from typing import Text, List, Any, Tuple

import numpy as np
import pytest
from pathlib import Path
from _pytest.logging import LogCaptureFixture
import logging

from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.constants import SPACY_DOCS, TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT, INTENT, RESPONSE
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer


@pytest.mark.parametrize(
    "sentence, expected_sequence_features, expected_sentence_features,"
    "labeled_tokens, additional_vocabulary_size",
    [
        (
            "hey how are you today",
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0],
            2,
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
            [1.0, 1.0, 0.0],
            [1, 0],
            0,
        ),
        (
            "blah balh random eh",
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [],
            None,
        ),
        (
            "a 1 digit number",
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1, 1],
            None,
        ),
    ],
)
def test_regex_featurizer(
    sentence: Text,
    expected_sequence_features: List[float],
    expected_sentence_features: List[float],
    labeled_tokens: List[int],
    additional_vocabulary_size: int,
    spacy_nlp: Any,
):
    from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]
    ftr = RegexFeaturizer(
        {"number_additional_patterns": additional_vocabulary_size},
        known_patterns=patterns,
    )

    # adds tokens to the message
    tokenizer = SpacyTokenizer({})
    message = Message(data={TEXT: sentence, RESPONSE: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    tokenizer.process(message)

    sequence_features, sentence_features = ftr._features_for_patterns(message, TEXT)
    assert np.allclose(
        sequence_features.toarray(), expected_sequence_features, atol=1e-10
    )
    assert np.allclose(
        sentence_features.toarray(), expected_sentence_features, atol=1e-10
    )

    # the tokenizer should have added tokens
    assert len(message.get(TOKENS_NAMES[TEXT], [])) > 0
    # the number of regex matches on each token should match
    for i, token in enumerate(message.get(TOKENS_NAMES[TEXT])):
        token_matches = token.get("pattern").values()
        num_matches = sum(token_matches)
        assert num_matches == labeled_tokens.count(i)


@pytest.mark.parametrize(
    "sentence, tokens, expected_sequence_features, expected_sentence_features,"
    "labeled_tokens",
    [
        (
            "明天上海的天气怎么样？",
            [("明天", 0), ("上海", 2), ("的", 4), ("天气", 5), ("怎么样", 7), ("？", 10)],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [1.0, 1.0],
            [0.0, 1.0],
        ),
        (
            "北京的天气如何？",
            [("北京", 0), ("的", 2), ("天气", 3), ("如何", 5), ("？", 7)],
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [1.0, 0.0],
            [0.0],
        ),
        (
            "昨天和今天的天气都不错",
            [("昨天", 0), ("和", 2), ("今天", 3), ("的", 5), ("天气", 6), ("都", 8), ("不错", 9)],
            [
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [0.0, 1.0],
            [0.0, 2.0],
        ),
        (
            "后天呢？",
            [("后天", 0), ("呢", 2), ("？", 3)],
            [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            [0.0, 1.0],
            [0.0],
        ),
    ],
)
def test_lookup_tables_without_use_word_boundaries(
    sentence: Text,
    tokens: List[Tuple[Text, float]],
    expected_sequence_features: List[float],
    expected_sentence_features: List[float],
    labeled_tokens: List[float],
):
    from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
    from rasa.nlu.tokenizers.tokenizer import Token

    lookups = [
        {"name": "cites", "elements": ["北京", "上海", "广州", "深圳", "杭州"],},
        {"name": "dates", "elements": ["昨天", "今天", "明天", "后天"],},
    ]
    ftr = RegexFeaturizer(
        {"use_word_boundaries": False, "number_additional_patterns": 0}
    )
    training_data = TrainingData()
    training_data.lookup_tables = lookups
    ftr.train(training_data)

    # adds tokens to the message
    message = Message(data={TEXT: sentence})
    message.set(TOKENS_NAMES[TEXT], [Token(word, start) for (word, start) in tokens])

    sequence_features, sentence_features = ftr._features_for_patterns(message, TEXT)
    assert np.allclose(
        sequence_features.toarray(), expected_sequence_features, atol=1e-10
    )
    assert np.allclose(
        sentence_features.toarray(), expected_sentence_features, atol=1e-10
    )

    # the number of regex matches on each token should match
    for i, token in enumerate(message.get(TOKENS_NAMES[TEXT])):
        token_matches = token.get("pattern").values()
        num_matches = sum(token_matches)
        assert num_matches == labeled_tokens.count(i)


@pytest.mark.parametrize(
    "sentence, expected_sequence_features, expected_sentence_features, "
    "labeled_tokens",
    [
        (
            "lemonade and mapo tofu",
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            [1.0, 1.0],
            [0.0, 2.0, 3.0],
        ),
        (
            "a cup of tea",
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            [1.0, 0.0],
            [3.0],
        ),
        (
            "Is burrito my favorite food?",
            [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],],
            [0.0, 1.0],
            [1.0],
        ),
        ("I want club?mate", [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]], [1.0, 0.0], [2.0]),
    ],
)
def test_lookup_tables(
    sentence: Text,
    expected_sequence_features: List[float],
    expected_sentence_features: List[float],
    labeled_tokens: List[float],
    spacy_nlp: Any,
):
    from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer

    lookups = [
        {
            "name": "drinks",
            "elements": ["mojito", "lemonade", "sweet berry wine", "tea", "club?mate"],
        },
        {"name": "plates", "elements": "data/test/lookup_tables/plates.txt"},
    ]
    ftr = RegexFeaturizer({"number_additional_patterns": 0})
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
    assert np.allclose(
        sequence_features.toarray(), expected_sequence_features, atol=1e-10
    )
    assert np.allclose(
        sentence_features.toarray(), expected_sentence_features, atol=1e-10
    )

    # the tokenizer should have added tokens
    assert len(message.get(TOKENS_NAMES[TEXT], [])) > 0
    # the number of regex matches on each token should match
    for i, token in enumerate(message.get(TOKENS_NAMES[TEXT])):
        token_matches = token.get("pattern").values()
        num_matches = sum(token_matches)
        assert num_matches == labeled_tokens.count(i)


@pytest.mark.parametrize(
    "sentence, expected_sequence_features, expected_sentence_features",
    [
        ("hey how are you today", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ("hey 456 how are you", [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]),
        ("blah balh random eh", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ("a 1 digit number", [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]),
    ],
)
def test_regex_featurizer_no_sequence(
    sentence: Text,
    expected_sequence_features: List[float],
    expected_sentence_features: List[float],
    spacy_nlp: Any,
):

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]
    ftr = RegexFeaturizer({"number_additional_patterns": 0}, known_patterns=patterns)

    # adds tokens to the message
    tokenizer = SpacyTokenizer()
    message = Message(data={TEXT: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    tokenizer.process(message)

    sequence_features, sentence_features = ftr._features_for_patterns(message, TEXT)
    assert np.allclose(
        sequence_features.toarray()[0], expected_sequence_features, atol=1e-10
    )
    assert np.allclose(
        sentence_features.toarray()[-1], expected_sentence_features, atol=1e-10
    )


def test_regex_featurizer_train():

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]

    featurizer = RegexFeaturizer.create(
        {"number_additional_patterns": 0}, RasaNLUModelConfig()
    )

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
    "sentence, expected_sequence_features, expected_sentence_features,"
    "case_sensitive",
    [
        ("Hey How are you today", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], True),
        ("Hey How are you today", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], False),
        ("Hey 456 How are you", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], True),
        ("Hey 456 How are you", [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], False),
    ],
)
def test_regex_featurizer_case_sensitive(
    sentence: Text,
    expected_sequence_features: List[float],
    expected_sentence_features: List[float],
    case_sensitive: bool,
    spacy_nlp: Any,
):

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]
    ftr = RegexFeaturizer(
        {"case_sensitive": case_sensitive, "number_additional_patterns": 0},
        known_patterns=patterns,
    )

    # adds tokens to the message
    tokenizer = SpacyTokenizer()
    message = Message(data={TEXT: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    tokenizer.process(message)

    sequence_features, sentence_features = ftr._features_for_patterns(message, TEXT)
    assert np.allclose(
        sequence_features.toarray()[0], expected_sequence_features, atol=1e-10
    )
    assert np.allclose(
        sentence_features.toarray()[-1], expected_sentence_features, atol=1e-10
    )


@pytest.mark.parametrize(
    "sentence, expected_sequence_features, expected_sentence_features,"
    "labeled_tokens, use_word_boundaries",
    [
        ("how are you", [[1.0], [0.0], [0.0]], [1.0], [0.0], True),
        ("how are you", [[1.0], [0.0], [0.0]], [1.0], [0.0], False),
        ("Take a shower", [[0.0], [0.0], [0.0]], [0.0], [], True),
        ("Take a shower", [[0.0], [0.0], [1.0]], [1.0], [2.0], False),
        ("What a show", [[0.0], [0.0], [0.0]], [0.0], [], True),
        ("What a show", [[0.0], [0.0], [1.0]], [1.0], [2.0], False),
        ("The wolf howled", [[0.0], [0.0], [0.0]], [0.0], [], True),
        ("The wolf howled", [[0.0], [0.0], [1.0]], [1.0], [2.0], False),
    ],
)
def test_lookup_with_and_without_boundaries(
    sentence: Text,
    expected_sequence_features: List[List[float]],
    expected_sentence_features: List[float],
    labeled_tokens: List[float],
    use_word_boundaries: bool,
    spacy_nlp: Any,
):
    ftr = RegexFeaturizer(
        {"use_word_boundaries": use_word_boundaries, "number_additional_patterns": 0}
    )
    training_data = TrainingData()

    # we use lookups because the "use_word_boundaries" flag is only used when
    # producing patterns from lookup tables
    lookups = [{"name": "how", "elements": ["how"]}]
    training_data.lookup_tables = lookups
    ftr.train(training_data)

    # adds tokens to the message
    tokenizer = SpacyTokenizer()
    message = Message(data={TEXT: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    tokenizer.process(message)

    (sequence_features, sentence_features) = ftr._features_for_patterns(message, TEXT)

    sequence_features = sequence_features.toarray()
    sentence_features = sentence_features.toarray()
    num_of_patterns = sum([len(lookup["elements"]) for lookup in lookups])
    assert sequence_features.shape == (
        len(message.get(TOKENS_NAMES[TEXT])),
        num_of_patterns,
    )
    num_of_lookup_tables = len(lookups)
    assert sentence_features.shape == (num_of_lookup_tables, num_of_patterns)

    # sequence_features should be {0,1} for each token: 1 if match, 0 if not
    assert np.allclose(sequence_features, expected_sequence_features, atol=1e-10)
    # sentence_features should be {0,1} for each lookup table: 1 if sentence
    # contains match from that table, 0 if not
    assert np.allclose(sentence_features, expected_sentence_features, atol=1e-10)

    # the tokenizer should have added tokens
    assert len(message.get(TOKENS_NAMES[TEXT], [])) > 0

    # the number of regex matches on each token should match
    for i, token in enumerate(message.get(TOKENS_NAMES[TEXT])):
        token_matches = token.get("pattern").values()
        num_matches = sum(token_matches)
        # labeled_tokens should list the token(s) which match a pattern
        assert num_matches == labeled_tokens.count(i)


def test_persist_load_for_finetuning(tmp_path: Path):
    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]

    featurizer = RegexFeaturizer.create(
        {"number_additional_patterns": 5}, RasaNLUModelConfig()
    )

    sentence = "hey how are you today 19.12.2019 ?"
    message = Message(data={TEXT: sentence})
    message.set(RESPONSE, sentence)
    message.set(INTENT, "intent")
    WhitespaceTokenizer().train(TrainingData([message]))

    featurizer.train(
        TrainingData([message], regex_features=patterns), RasaNLUModelConfig()
    )

    persist_value = featurizer.persist("ftr", str(tmp_path))

    # Test all artifacts stored as part of persist
    assert persist_value["file"] == "ftr"
    assert (tmp_path / "ftr.patterns.pkl").exists()
    assert (tmp_path / "ftr.vocabulary_stats.pkl").exists()
    assert featurizer.vocabulary_stats == {
        "max_number_patterns": 8,
        "pattern_slots_filled": 3,
    }

    loaded_featurizer = RegexFeaturizer.load(
        meta={"number_additional_patterns": 5, "file": persist_value["file"],},
        should_finetune=True,
        model_dir=str(tmp_path),
    )

    # Test component loaded in finetune mode and also with
    # same patterns as before and vocabulary statistics
    assert loaded_featurizer.known_patterns == featurizer.known_patterns
    assert loaded_featurizer.finetune_mode
    assert loaded_featurizer.pattern_vocabulary_stats == featurizer.vocabulary_stats

    new_lookups = [{"name": "plates", "elements": "data/test/lookup_tables/plates.txt"}]

    training_data = TrainingData()
    training_data.lookup_tables = new_lookups
    loaded_featurizer.train(training_data)

    # Test merging of a new pattern to an already trained component.
    assert len(loaded_featurizer.known_patterns) == 4
    assert loaded_featurizer.vocabulary_stats == {
        "max_number_patterns": 8,
        "pattern_slots_filled": 4,
    }


def test_incremental_train_featurization(tmp_path: Path):
    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]

    featurizer = RegexFeaturizer.create(
        {"number_additional_patterns": 5}, RasaNLUModelConfig()
    )

    sentence = "hey how are you today 19.12.2019 ?"
    message = Message(data={TEXT: sentence})
    message.set(RESPONSE, sentence)
    message.set(INTENT, "intent")
    WhitespaceTokenizer().train(TrainingData([message]))

    featurizer.train(
        TrainingData([message], regex_features=patterns), RasaNLUModelConfig()
    )

    # Test featurization of message
    expected = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    expected_cls = np.array([1, 1, 1, 0, 0, 0, 0, 0])

    seq_vecs, sen_vec = message.get_sparse_features(TEXT, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vec:
        sen_vec = sen_vec.features

    assert (6, 8) == seq_vecs.shape
    assert (1, 8) == sen_vec.shape
    assert np.all(seq_vecs.toarray()[0] == expected)
    assert np.all(sen_vec.toarray()[-1] == expected_cls)

    persist_value = featurizer.persist("ftr", str(tmp_path))
    loaded_featurizer = RegexFeaturizer.load(
        meta={"number_additional_patterns": 5, "file": persist_value["file"],},
        should_finetune=True,
        model_dir=str(tmp_path),
    )

    new_patterns = [
        {"pattern": "\\btoday*", "name": "day", "usage": "intent"},
        {"pattern": "\\bhey+", "name": "hello", "usage": "intent"},
    ]

    message = Message(data={TEXT: sentence})
    message.set(RESPONSE, sentence)
    message.set(INTENT, "intent")
    WhitespaceTokenizer().train(TrainingData([message]))

    loaded_featurizer.train(
        TrainingData([message], regex_features=patterns + new_patterns),
        RasaNLUModelConfig(),
    )

    # Test featurization of message, this time for the extra pattern as well.
    expected_token_1 = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    expected_token_2 = np.array([0, 0, 0, 1, 0, 0, 0, 0])
    expected_cls = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    seq_vecs, sen_vec = message.get_sparse_features(TEXT, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vec:
        sen_vec = sen_vec.features

    assert (6, 8) == seq_vecs.shape
    assert (1, 8) == sen_vec.shape
    assert np.all(seq_vecs.toarray()[0] == expected_token_1)
    assert np.all(seq_vecs.toarray()[-2] == expected_token_2)
    assert np.all(sen_vec.toarray()[-1] == expected_cls)

    # we also modified a pattern, check if that is correctly modified
    pattern_to_check = [
        pattern
        for pattern in loaded_featurizer.known_patterns
        if pattern["name"] == "hello"
    ]
    assert pattern_to_check == [new_patterns[1]]


def test_vocabulary_overflow_log():
    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]

    featurizer = RegexFeaturizer(
        {"number_additional_patterns": 1},
        known_patterns=patterns,
        finetune_mode=True,
        pattern_vocabulary_stats={"max_number_patterns": 4, "pattern_slots_filled": 3},
    )

    additional_patterns = [
        {"pattern": "\\btoday*", "name": "day", "usage": "intent"},
        {"pattern": "\\bhello+", "name": "greet", "usage": "intent"},
    ]

    with pytest.warns(UserWarning) as warning:
        featurizer.train(TrainingData([], regex_features=additional_patterns))
    assert (
        "The originally trained model was configured to handle a maximum number of 4 patterns"
        in warning[0].message.args[0]
    )
