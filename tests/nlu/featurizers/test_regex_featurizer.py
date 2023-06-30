from typing import Text, List, Any, Tuple, Callable, Dict, Optional

import dataclasses
import numpy as np
import pytest

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.constants import SPACY_DOCS, TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT, INTENT, RESPONSE
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer


@pytest.fixture()
def resource() -> Resource:
    return Resource("regex_featurizer")


@pytest.fixture()
def create_featurizer(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    resource: Resource,
) -> Callable[..., RegexFeaturizer]:
    def inner(
        config: Dict[Text, Any] = None,
        known_patterns: Optional[List[Dict[Text, Any]]] = None,
    ) -> RegexFeaturizer:
        config = config or {}
        return RegexFeaturizer(
            {**RegexFeaturizer.get_default_config(), **config},
            default_model_storage,
            resource,
            default_execution_context,
            known_patterns,
        )

    return inner


@pytest.mark.parametrize(
    "sentence, expected_sequence_features, expected_sentence_features,"
    "labeled_tokens",
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
            [0.0, 1.0, 0.0],
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
            [1.0, 1.0, 0.0],
            [1, 0],
        ),
        (
            "blah balh random eh",
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [0.0, 0.0, 0.0],
            [],
        ),
        (
            "a 1 digit number",
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [1.0, 0.0, 1.0],
            [1, 1],
        ),
    ],
)
def test_regex_featurizer(
    sentence: Text,
    expected_sequence_features: List[float],
    expected_sentence_features: List[float],
    labeled_tokens: List[int],
    spacy_nlp: Any,
    create_featurizer: Callable[..., RegexFeaturizer],
    spacy_tokenizer: SpacyTokenizer,
):
    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]
    ftr = create_featurizer(known_patterns=patterns)

    # adds tokens to the message
    message = Message(data={TEXT: sentence, RESPONSE: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    spacy_tokenizer.process([message])

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
            [
                ("明天", 0),
                ("上海", 2),
                ("的", 4),
                ("天气", 5),
                ("怎么样", 7),
                ("？", 10),
            ],
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
            [
                ("昨天", 0),
                ("和", 2),
                ("今天", 3),
                ("的", 5),
                ("天气", 6),
                ("都", 8),
                ("不错", 9),
            ],
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
    create_featurizer: Callable[..., RegexFeaturizer],
):
    from rasa.nlu.tokenizers.tokenizer import Token

    lookups = [
        {"name": "cites", "elements": ["北京", "上海", "广州", "深圳", "杭州"]},
        {"name": "dates", "elements": ["昨天", "今天", "明天", "后天"]},
    ]
    ftr = create_featurizer({"use_word_boundaries": False})
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
            [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
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
    spacy_tokenizer: SpacyTokenizer,
    create_featurizer: Callable[..., RegexFeaturizer],
):
    lookups = [
        {
            "name": "drinks",
            "elements": ["mojito", "lemonade", "sweet berry wine", "tea", "club?mate"],
        },
        {"name": "plates", "elements": "data/test/lookup_tables/plates.txt"},
    ]
    ftr = create_featurizer()
    training_data = TrainingData()
    training_data.lookup_tables = lookups
    ftr.train(training_data)
    ftr.process_training_data(training_data)

    # adds tokens to the message
    message = Message(data={TEXT: sentence})
    message.set("text_spacy_doc", spacy_nlp(sentence))
    spacy_tokenizer.process([message])

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
    create_featurizer: Callable[..., RegexFeaturizer],
    spacy_tokenizer: SpacyTokenizer,
):

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]
    ftr = create_featurizer(known_patterns=patterns)

    # adds tokens to the message
    message = Message(data={TEXT: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    spacy_tokenizer.process([message])

    sequence_features, sentence_features = ftr._features_for_patterns(message, TEXT)
    assert np.allclose(
        sequence_features.toarray()[0], expected_sequence_features, atol=1e-10
    )
    assert np.allclose(
        sentence_features.toarray()[-1], expected_sentence_features, atol=1e-10
    )


def test_regex_featurizer_train(
    create_featurizer: Callable[..., RegexFeaturizer],
    whitespace_tokenizer: WhitespaceTokenizer,
):
    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]

    featurizer = create_featurizer()
    sentence = "hey how are you today 19.12.2019 ?"
    message = Message(data={TEXT: sentence})
    message.set(RESPONSE, sentence)
    message.set(INTENT, "intent")

    whitespace_tokenizer.process_training_data(TrainingData([message]))
    training_data = TrainingData([message], regex_features=patterns)

    featurizer.train(training_data)
    featurizer.process_training_data(training_data)

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
    create_featurizer: Callable[..., RegexFeaturizer],
    spacy_tokenizer: SpacyTokenizer,
):

    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]
    ftr = create_featurizer({"case_sensitive": case_sensitive}, known_patterns=patterns)

    # adds tokens to the message
    message = Message(data={TEXT: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    spacy_tokenizer.process([message])

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
    create_featurizer: Callable[..., RegexFeaturizer],
    spacy_tokenizer: SpacyTokenizer,
):
    ftr = create_featurizer({"use_word_boundaries": use_word_boundaries})
    training_data = TrainingData()

    # we use lookups because the "use_word_boundaries" flag is only used when
    # producing patterns from lookup tables
    lookups = [{"name": "how", "elements": ["how"]}]
    training_data.lookup_tables = lookups
    ftr.train(training_data)

    # adds tokens to the message
    message = Message(data={TEXT: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    spacy_tokenizer.process([message])

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


def test_persist_load_for_finetuning(
    create_featurizer: Callable[..., RegexFeaturizer],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    resource: Resource,
    whitespace_tokenizer: WhitespaceTokenizer,
):
    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]

    featurizer = create_featurizer()

    sentence = "hey how are you today 19.12.2019 ?"
    message = Message(data={TEXT: sentence})
    message.set(RESPONSE, sentence)
    message.set(INTENT, "intent")
    training_data = TrainingData([message], regex_features=patterns)
    whitespace_tokenizer.process_training_data(training_data)

    featurizer.train(training_data)

    loaded_featurizer = RegexFeaturizer.load(
        RegexFeaturizer.get_default_config(),
        default_model_storage,
        resource,
        dataclasses.replace(default_execution_context, is_finetuning=True),
    )

    # Test component loaded in finetune mode and also with
    # same patterns as before and vocabulary statistics
    assert loaded_featurizer.known_patterns == featurizer.known_patterns
    assert loaded_featurizer.finetune_mode

    new_lookups = [{"name": "plates", "elements": "data/test/lookup_tables/plates.txt"}]

    training_data = TrainingData()
    training_data.lookup_tables = new_lookups
    loaded_featurizer.train(training_data)

    # Test merging of a new pattern to an already trained component.
    assert len(loaded_featurizer.known_patterns) == 4


def test_vocabulary_expand_for_finetuning(
    create_featurizer: Callable[..., RegexFeaturizer],
    default_model_storage: ModelStorage,
    resource: Resource,
    default_execution_context: ExecutionContext,
    whitespace_tokenizer: WhitespaceTokenizer,
):
    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
    ]

    featurizer = create_featurizer()

    sentence = "hey hey 2020"
    message = Message(data={TEXT: sentence})
    message.set(RESPONSE, sentence)
    message.set(INTENT, "intent")
    training_data = TrainingData([message], regex_features=patterns)

    whitespace_tokenizer.process_training_data(training_data)

    featurizer.train(training_data)
    featurizer.process_training_data(training_data)

    # Test featurization of message
    expected = np.array([1, 0])
    expected_cls = np.array([1, 1])
    seq_vecs, sen_vec = message.get_sparse_features(TEXT, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vec:
        sen_vec = sen_vec.features

    assert (3, 2) == seq_vecs.shape
    assert (1, 2) == sen_vec.shape
    assert np.all(seq_vecs.toarray()[0] == expected)
    assert np.all(sen_vec.toarray()[-1] == expected_cls)

    loaded_featurizer = RegexFeaturizer.load(
        RegexFeaturizer.get_default_config(),
        default_model_storage,
        resource,
        dataclasses.replace(default_execution_context, is_finetuning=True),
    )

    new_patterns = [
        {"pattern": "\\btoday*", "name": "day", "usage": "intent"},
        {"pattern": "\\bhey+", "name": "hello", "usage": "intent"},
    ]
    new_sentence = "hey today"
    message = Message(data={TEXT: new_sentence})
    message.set(RESPONSE, new_sentence)
    message.set(INTENT, "intent")
    new_training_data = TrainingData([message], regex_features=patterns + new_patterns)

    whitespace_tokenizer.process_training_data(new_training_data)

    loaded_featurizer.train(new_training_data)
    loaded_featurizer.process_training_data(new_training_data)

    # Test featurization of message, this time for the extra pattern as well.
    expected_token_1 = np.array([1, 0, 0])
    expected_token_2 = np.array([0, 0, 1])
    expected_cls = np.array([1, 0, 1])

    seq_vecs, sen_vec = message.get_sparse_features(TEXT, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vec:
        sen_vec = sen_vec.features

    assert (2, 3) == seq_vecs.shape
    assert (1, 3) == sen_vec.shape
    assert np.all(seq_vecs.toarray()[0] == expected_token_1)
    assert np.all(seq_vecs.toarray()[1] == expected_token_2)
    assert np.all(sen_vec.toarray()[-1] == expected_cls)

    # let's check if the order of patterns is preserved
    for old_index, pattern in enumerate(featurizer.known_patterns):
        assert pattern["name"] == loaded_featurizer.known_patterns[old_index]["name"]

    # we also modified a pattern, check if that is correctly modified
    pattern_to_check = [
        pattern
        for pattern in loaded_featurizer.known_patterns
        if pattern["name"] == "hello"
    ]
    assert pattern_to_check == [new_patterns[1]]
