import numpy as np
import pytest
import re
from typing import Text, Dict, Any, Callable, List, Optional, Union

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    MESSAGE_ATTRIBUTES,
    TOKENS_NAMES,
)
from rasa.nlu.tokenizers.spacy_tokenizer import POS_TAG_KEY
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
    FEATURES,
)
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import FEATURE_TYPE_SEQUENCE, TEXT
from rasa.shared.exceptions import InvalidConfigException
from rasa.nlu.tokenizers.tokenizer import Token


@pytest.fixture
def resource_lexical_syntactic_featurizer() -> Resource:
    return Resource("LexicalSyntacticFeaturizer")


@pytest.fixture
def create_lexical_syntactic_featurizer(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    resource_lexical_syntactic_featurizer: Resource,
) -> Callable[[Dict[Text, Any]], LexicalSyntacticFeaturizer]:
    def inner(config: Dict[Text, Any]):
        return LexicalSyntacticFeaturizer.create(
            config={**LexicalSyntacticFeaturizer.get_default_config(), **config},
            model_storage=default_model_storage,
            execution_context=default_execution_context,
            resource=resource_lexical_syntactic_featurizer,
        )

    return inner


@pytest.mark.parametrize(
    "sentence,part_of_speech,feature_config,expected_features",
    [
        # simple example 1
        (
            "hello goodbye hello",
            None,
            [["BOS", "upper"], ["BOS", "EOS", "prefix2", "digit"], ["EOS", "low"]],
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        ),
        # simple example 2
        (
            "a 1",
            None,
            [["BOS", "upper"], ["BOS", "EOS", "prefix2", "digit"], ["EOS", "low"]],
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ],
        ),
        # larger window size
        (
            "hello 123 hello 123 hello",
            None,
            [["upper"], ["digit"], ["low"], ["digit"]],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            # Note:
            # 1. we just describe the features for first token here
            # 2. "123".islower() == "123".isupper() == False, which is why we end
            #     up with 7 features
        ),
        # with part of speech
        (
            "The sun is shining",
            ["DET", "NOUN", "AUX", "VERB"],
            [["pos", "pos2"]],
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ],
        ),
    ],
)
def test_feature_computation(
    create_lexical_syntactic_featurizer: Callable[
        [Dict[Text, Any]], LexicalSyntacticFeaturizer
    ],
    sentence: Text,
    part_of_speech: Optional[List[Text]],
    feature_config: List[List[Text]],
    expected_features: List[Union[int, List[int]]],
):
    featurizer = create_lexical_syntactic_featurizer(
        {"alias": "lsf", "features": feature_config}
    )

    # build the message
    tokens = [
        Token(text=match[0], start=match.start())
        for match in re.finditer(r"\w+", sentence)
    ]
    # ... and add part of speech tags (str) to tokens (if specified)
    if part_of_speech:
        assert len(tokens) == len(part_of_speech)
        for token, pos in zip(tokens, part_of_speech):
            token.data = {POS_TAG_KEY: pos}
    message = Message(data={TOKENS_NAMES[TEXT]: tokens})

    # train
    featurizer.train(TrainingData([message]))
    assert not message.features

    # process
    featurizer.process([message])
    assert len(message.features) == 1
    feature = message.features[0]
    assert feature.attribute == TEXT
    assert feature.is_sparse()
    assert feature.type == FEATURE_TYPE_SEQUENCE
    assert feature.features.shape[0] == len(tokens)

    if isinstance(expected_features[0], List):
        assert len(expected_features) == feature.features.shape[0]
        # we specified the full matrix
        assert np.all(feature.features.todense() == expected_features)
    else:
        assert len(expected_features) == feature.features.shape[1]
        # just check features for the first token
        assert np.all(feature.features.todense()[0] == expected_features)


def test_features_for_messages_with_missing_part_of_speech_tags(
    create_lexical_syntactic_featurizer: Callable[
        [Dict[Text, Any]], LexicalSyntacticFeaturizer
    ]
):
    # build the message and do NOT add part of speech information
    sentence = "hello goodbye hello"
    message_data = {
        TOKENS_NAMES[TEXT]: [
            Token(text=match[0], start=match.start())
            for match in re.finditer(r"\w+", sentence)
        ]
    }
    message = Message(data=message_data)

    # train and process
    featurizer = create_lexical_syntactic_featurizer(
        {"alias": "lsf", "features": [["BOS", "pos"]]}
    )
    featurizer.train(TrainingData([message]))
    featurizer.process([message])
    feature = message.features[0]
    assert feature.features.shape[1] == 3  # BOS = True/False, pos = None


def test_only_featurizes_text_attribute(
    create_lexical_syntactic_featurizer: Callable[
        [Dict[Text, Any]], LexicalSyntacticFeaturizer
    ]
):
    # build a message with tokens for lots of attributes
    sentence = "hello goodbye hello"
    tokens = [
        Token(text=match[0], start=match.start())
        for match in re.finditer(r"\w+", sentence)
    ]
    message_data = {}
    for attribute in MESSAGE_ATTRIBUTES + DENSE_FEATURIZABLE_ATTRIBUTES:
        message_data[attribute] = sentence
        message_data[TOKENS_NAMES[attribute]] = tokens
    message = Message(data=message_data)

    # train and process
    featurizer = create_lexical_syntactic_featurizer(
        {"alias": "lsf", "features": [["BOS"]]}
    )
    featurizer.train(TrainingData([message]))
    featurizer.process([message])
    assert len(message.features) == 1
    assert message.features[0].attribute == TEXT


def test_process_multiple_messages(
    create_lexical_syntactic_featurizer: Callable[
        [Dict[Text, Any]], LexicalSyntacticFeaturizer
    ]
):
    # build a message with tokens for lots of attributes
    multiple_messages = []
    for sentence in ["hello", "hello there"]:
        tokens = [
            Token(text=match[0], start=match.start())
            for match in re.finditer(r"\w+", sentence)
        ]

        multiple_messages.append(Message(data={TOKENS_NAMES[TEXT]: tokens}))

    # train and process
    featurizer = create_lexical_syntactic_featurizer(
        {"alias": "lsf", "features": [["prefix2"]]}
    )
    featurizer.train(TrainingData(multiple_messages))
    featurizer.process(multiple_messages)
    for message in multiple_messages:
        assert len(message.features) == 1
        assert message.features[0].attribute == TEXT

    # we know both texts where used for training if more than one feature has been
    # extracted e.g. for the first message from which only the prefix "he" can be
    # extracted
    assert multiple_messages[0].features[0].features.shape[-1] > 1


@pytest.mark.parametrize("feature_config", [(["pos", "BOS"],)])
def test_create_train_load_and_process(
    create_lexical_syntactic_featurizer: Callable[
        [Dict[Text, Any]], LexicalSyntacticFeaturizer
    ],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    resource_lexical_syntactic_featurizer: Resource,
    feature_config: List[Text],
) -> Callable[..., LexicalSyntacticFeaturizer]:

    config = {"alias": "lsf", "features": feature_config}
    featurizer = create_lexical_syntactic_featurizer(config)

    sentence = "Hello how are you"
    tokens = [
        Token(text=match[0], start=match.start())
        for match in re.finditer(r"\w+", sentence)
    ]
    message = Message(data={TOKENS_NAMES[TEXT]: tokens})

    featurizer.train(TrainingData([message]))

    loaded_featurizer = LexicalSyntacticFeaturizer.load(
        config={**LexicalSyntacticFeaturizer.get_default_config(), **config},
        model_storage=default_model_storage,
        execution_context=default_execution_context,
        resource=resource_lexical_syntactic_featurizer,
    )

    assert loaded_featurizer._feature_to_idx_dict == featurizer._feature_to_idx_dict


@pytest.mark.parametrize(
    "config,raises",
    [
        # do not raise
        ({}, False),
        ({**LexicalSyntacticFeaturizer.get_default_config()}, False),
        ({FEATURES: [["suffix2"]]}, False),
        (
            {
                "bla": "obviously an unknown extra feature",
                "faeturizer": "typos are also unknown features",
            },
            False,
        ),
        # raise
        ({FEATURES: ["pos", "suffix2"]}, True),
        ({FEATURES: ["suffix1234"]}, True),
    ],
)
def test_validate_config(config: Dict[Text, Any], raises: bool):
    if not raises:
        LexicalSyntacticFeaturizer.validate_config(config)
    else:
        with pytest.raises(InvalidConfigException):
            LexicalSyntacticFeaturizer.validate_config(config)


@pytest.mark.parametrize(
    "sentence, feature_config, expected_features",
    [("The sun is shining", [["pos", "pos2"]], np.ones(shape=(4, 2)))],
)
def test_warn_if_part_of_speech_features_cannot_be_computed(
    create_lexical_syntactic_featurizer: Callable[
        [Dict[Text, Any]], LexicalSyntacticFeaturizer
    ],
    sentence: Text,
    feature_config: Dict[Text, Any],
    expected_features: np.ndarray,
):

    featurizer = create_lexical_syntactic_featurizer(
        {"alias": "lsf", "features": feature_config}
    )

    # build the message - with tokens but *no* part-of-speech tags
    tokens = [
        Token(text=match[0], start=match.start())
        for match in re.finditer(r"\w+", sentence)
    ]
    message = Message(data={TOKENS_NAMES[TEXT]: tokens})

    # train
    with pytest.warns(
        UserWarning,
        match="Expected training data to include tokens with part-of-speech tags",
    ):
        featurizer.train(TrainingData([message]))
    assert not message.features

    # process
    with pytest.warns(None) as records:
        featurizer.process([message])
    assert len(records) == 0
    assert len(message.features) == 1
    feature = message.features[0]
    assert np.all(feature.features.todense() == expected_features)
