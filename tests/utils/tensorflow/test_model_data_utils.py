from typing import Any, Text, Optional, Dict, List

import pytest
import scipy.sparse
import numpy as np
import copy

from spacy import Language

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.nlu.constants import SPACY_DOCS
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.utils.tensorflow import model_data_utils
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import (
    ACTION_NAME,
    TEXT,
    INTENT,
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)
from rasa.utils.tensorflow.constants import SENTENCE
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.tensorflow.model_data_utils import TAG_ID_ORIGIN

shape = 100


def test_create_fake_features():
    # DENSE FEATURES
    dense_feature_sentence_features = Features(
        features=np.random.rand(shape),
        attribute=INTENT,
        feature_type=SENTENCE,
        origin=[],
    )
    features = [[None, None, [dense_feature_sentence_features]]]

    fake_features = model_data_utils._create_fake_features(features)
    assert len(fake_features) == 1
    assert fake_features[0].is_dense()
    assert fake_features[0].features.shape == (0, shape)

    # SPARSE FEATURES
    sparse_feature_sentence_features = Features(
        features=scipy.sparse.coo_matrix(np.random.rand(shape)),
        attribute=INTENT,
        feature_type=SENTENCE,
        origin=[],
    )
    features = [[None, None, [sparse_feature_sentence_features]]]
    fake_features = model_data_utils._create_fake_features(features)
    assert len(fake_features) == 1
    assert fake_features[0].is_sparse()
    assert fake_features[0].features.shape == (0, shape)
    assert fake_features[0].features.nnz == 0


def test_surface_attributes():
    intent_features = {
        INTENT: [
            Features(
                features=np.random.rand(shape),
                attribute=INTENT,
                feature_type=SENTENCE,
                origin="featurizer-a",
            ),
            Features(
                features=np.random.rand(shape),
                attribute=INTENT,
                feature_type=SENTENCE,
                origin="featurizer-b",
            ),
        ]
    }

    action_name_features = scipy.sparse.coo_matrix(np.random.rand(shape))
    action_name_features = {
        ACTION_NAME: [
            Features(
                features=action_name_features,
                attribute=ACTION_NAME,
                feature_type=SENTENCE,
                origin="featurizer-c",
            )
        ]
    }
    state_features = copy.deepcopy(intent_features)
    state_features.update(copy.deepcopy(action_name_features))
    # test on 2 dialogs -- one with dialog length 3 the other one with dialog length 2
    dialogs = [[state_features, intent_features, {}], [{}, action_name_features]]
    surfaced_features = model_data_utils._surface_attributes(
        dialogs, featurizers=["featurizer-a", "featurizer-c"]
    )
    assert INTENT in surfaced_features and ACTION_NAME in surfaced_features
    # check that number of lists corresponds to number of dialogs
    assert (
        len(surfaced_features.get(INTENT)) == 2
        and len(surfaced_features.get(ACTION_NAME)) == 2
    )
    # length of each list corresponds to length of the dialog
    assert (
        len(surfaced_features.get(INTENT)[0]) == 3
        and len(surfaced_features.get(INTENT)[1]) == 2
    )
    assert (
        len(surfaced_features.get(ACTION_NAME)[0]) == 3
        and len(surfaced_features.get(ACTION_NAME)[1]) == 2
    )
    # check that features are correctly populated with `None`s
    assert (
        surfaced_features.get(INTENT)[0][2] is None
        and surfaced_features.get(INTENT)[1][0] is None
        and surfaced_features.get(INTENT)[1][1] is None
    )
    assert (
        surfaced_features.get(ACTION_NAME)[0][1] is None
        and surfaced_features.get(ACTION_NAME)[0][2] is None
        and surfaced_features.get(ACTION_NAME)[1][0] is None
    )
    # check that all features are the same as before
    assert all(
        [
            (turn[0].features == intent_features[INTENT][0].features).all()
            for dialogue in surfaced_features.get(INTENT)
            for turn in dialogue
            if turn is not None
        ]
    )
    assert all(
        [
            (turn[0].features != action_name_features[ACTION_NAME][0].features).nnz == 0
            for dialogue in surfaced_features.get(ACTION_NAME)
            for turn in dialogue
            if turn is not None
        ]
    )


def test_extract_features():
    fake_features = np.zeros(shape)
    fake_features_as_features = Features(
        features=fake_features, attribute=INTENT, feature_type=SENTENCE, origin=[]
    )
    # create zero features
    fake_features_list = [fake_features_as_features]

    # create tracker state features by setting a random index in the array to 1
    random_inds = np.random.randint(shape, size=6)
    list_of_features = []
    for idx in random_inds:
        current_features = copy.deepcopy(fake_features_as_features)
        current_features.features[idx] = 1
        list_of_features.append([current_features])

    # organize the created features into lists ~ dialog history
    tracker_features = [
        [list_of_features[0], None, list_of_features[1]],
        [None, None, list_of_features[2]],
        [list_of_features[3], list_of_features[4], list_of_features[5]],
    ]

    (
        attribute_masks,
        dense_features,
        sparse_features,
    ) = model_data_utils._extract_features(tracker_features, fake_features_list, INTENT)
    expected_mask = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 1]])

    assert np.all(np.squeeze(np.array(attribute_masks), 2) == expected_mask)
    assert np.array(dense_features[SENTENCE]).shape[-1] == fake_features.shape[-1]
    assert sparse_features == {}


@pytest.mark.parametrize(
    "text, intent, entities, attributes, real_sparse_feature_sizes",
    [
        ("Hello!", "greet", None, [TEXT], {"text": {"sequence": [1], "sentence": [1]}}),
        (
            "Hello!",
            "greet",
            None,
            [TEXT, INTENT],
            {
                "intent": {"sentence": [], "sequence": [1]},
                "text": {"sequence": [1], "sentence": [1]},
            },
        ),
        (
            "Hello Max!",
            "greet",
            [{"entity": "name", "value": "Max", "start": 6, "end": 9}],
            [TEXT, ENTITIES],
            {"text": {"sequence": [2], "sentence": [2]}},
        ),
    ],
)
def test_convert_training_examples(
    spacy_nlp: Language,
    text: Text,
    intent: Optional[Text],
    entities: Optional[List[Dict[Text, Any]]],
    attributes: List[Text],
    real_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    message = Message(data={TEXT: text, INTENT: intent, ENTITIES: entities})

    tokenizer = SpacyTokenizer.create(
        SpacyTokenizer.get_default_config(),
        default_model_storage,
        Resource("tokenizer"),
        default_execution_context,
    )
    count_vectors_featurizer = CountVectorsFeaturizer.create(
        CountVectorsFeaturizer.get_default_config(),
        default_model_storage,
        Resource("count_featurizer"),
        default_execution_context,
    )
    spacy_featurizer = SpacyFeaturizer.create(
        SpacyFeaturizer.get_default_config(),
        default_model_storage,
        Resource("spacy_featurizer"),
        default_execution_context,
    )

    message.set(SPACY_DOCS[TEXT], spacy_nlp(text))

    training_data = TrainingData([message])
    tokenizer.process_training_data(training_data)
    count_vectors_featurizer.train(training_data)
    count_vectors_featurizer.process_training_data(training_data)
    spacy_featurizer.process_training_data(training_data)

    entity_tag_spec = [
        EntityTagSpec(
            "entity",
            {0: "O", 1: "name", 2: "location"},
            {"O": 0, "name": 1, "location": 2},
            3,
        )
    ]
    output, sparse_feature_sizes = model_data_utils.featurize_training_examples(
        [message], attributes=attributes, entity_tag_specs=entity_tag_spec
    )

    assert len(output) == 1
    for attribute in attributes:
        assert attribute in output[0]
    for attribute in {INTENT, TEXT, ENTITIES} - set(attributes):
        assert attribute not in output[0]
    # we have sparse sentence, sparse sequence, dense sentence, and dense sequence
    # features in the list
    assert len(output[0][TEXT]) == 4
    if INTENT in attributes:
        # we will just have space sentence features
        assert len(output[0][INTENT]) == 1
    if ENTITIES in attributes:
        # we will just have space sentence features
        assert len(output[0][ENTITIES]) == len(entity_tag_spec)
    # check that it calculates sparse_feature_sizes correctly
    assert sparse_feature_sizes == real_sparse_feature_sizes


@pytest.mark.parametrize(
    "features, featurizers, expected_features",
    [
        ([], None, []),
        (None, ["featurizer-a"], None),
        (
            [
                Features(
                    np.random.rand(5, 14), FEATURE_TYPE_SENTENCE, TEXT, "featurizer-a"
                )
            ],
            None,
            [
                Features(
                    np.random.rand(5, 14), FEATURE_TYPE_SENTENCE, TEXT, "featurizer-a"
                )
            ],
        ),
        (
            [
                Features(
                    np.random.rand(5, 14), FEATURE_TYPE_SENTENCE, TEXT, "featurizer-a"
                )
            ],
            ["featurizer-b"],
            [],
        ),
        (
            [
                Features(
                    np.random.rand(5, 14), FEATURE_TYPE_SENTENCE, TEXT, "featurizer-a"
                ),
                Features(
                    np.random.rand(5, 14),
                    FEATURE_TYPE_SEQUENCE,
                    ACTION_NAME,
                    "featurizer-b",
                ),
            ],
            ["featurizer-b"],
            [
                Features(
                    np.random.rand(5, 14),
                    FEATURE_TYPE_SEQUENCE,
                    ACTION_NAME,
                    "featurizer-b",
                )
            ],
        ),
        (
            [
                Features(
                    np.random.rand(5, 14), FEATURE_TYPE_SEQUENCE, "role", TAG_ID_ORIGIN
                ),
                Features(
                    np.random.rand(5, 14),
                    FEATURE_TYPE_SEQUENCE,
                    ACTION_NAME,
                    "featurizer-b",
                ),
            ],
            ["featurizer-b"],
            [
                Features(
                    np.random.rand(5, 14), FEATURE_TYPE_SEQUENCE, "role", TAG_ID_ORIGIN
                ),
                Features(
                    np.random.rand(5, 14),
                    FEATURE_TYPE_SEQUENCE,
                    ACTION_NAME,
                    "featurizer-b",
                ),
            ],
        ),
    ],
)
def test_filter_features(
    features: Optional[List["Features"]],
    featurizers: Optional[List[Text]],
    expected_features: Optional[List["Features"]],
):
    actual_features = model_data_utils._filter_features(features, featurizers)

    if expected_features is None:
        assert actual_features is None
        return

    assert len(actual_features) == len(expected_features)
    for actual_feature, expected_feature in zip(actual_features, expected_features):
        assert expected_feature.origin == actual_feature.origin
        assert expected_feature.type == actual_feature.type
        assert expected_feature.attribute == actual_feature.attribute
