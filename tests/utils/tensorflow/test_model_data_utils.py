from typing import Any, Text, Optional, Dict, List, Tuple, Union

import pytest
import scipy.sparse
import numpy as np
import copy
import itertools

from spacy import Language

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.nlu.constants import (
    ENTITY_ATTRIBUTE_CONFIDENCE_GROUP,
    SENTENCE_FEATURES,
    SEQUENCE_FEATURES,
    SPACY_DOCS,
)
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import (
    SpacyFeaturizerGraphComponent,
)
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizerGraphComponent,
)
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizerGraphComponent
from rasa.utils.tensorflow import model_data_utils
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import (
    ACTION_NAME,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_TYPE,
    TEXT,
    INTENT,
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)
from rasa.utils.tensorflow.constants import MASK, SENTENCE, SEQUENCE
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.tensorflow.model_data_utils import TAG_ID_ORIGIN, convert_to_data_format

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
        (
            "Hello!",
            "greet",
            None,
            [TEXT],
            {"text": {"sequence": [1], "sentence": [1]}},
        ),
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

    tokenizer = SpacyTokenizerGraphComponent.create(
        SpacyTokenizerGraphComponent.get_default_config(),
        default_model_storage,
        Resource("tokenizer"),
        default_execution_context,
    )
    count_vectors_featurizer = CountVectorsFeaturizerGraphComponent.create(
        CountVectorsFeaturizerGraphComponent.get_default_config(),
        default_model_storage,
        Resource("count_featurizer"),
        default_execution_context,
    )
    spacy_featurizer = SpacyFeaturizerGraphComponent.create(
        SpacyFeaturizerGraphComponent.get_default_config(),
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
        [message], attributes=attributes, entity_tag_specs=entity_tag_spec,
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
        # we will just have sparse sentence features
        assert len(output[0][INTENT]) == 1
    if ENTITIES in attributes:
        # we will just have sparse sentence features
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


def _create_dummy_features(
    is_sequence: bool,
    is_sparse: bool,
    num_records: int = 10,
    feature_dim: int = 3,
    message_attribute: Text = "attribute_featurized_only_once",
    feature_attribute: Text = "attribute_featurized_only_once",
    feature_origin: Text = "the-only-featurizer",
    seed: int = 1234,
) -> Tuple[List[Dict[Text, List[Features]]], List[int]]:
    """Creates some dummy data containing a single feature for each record.

    We generate features such that:
    1. The features for the `i`-th record is a matrix filled with `i` if is a
       dense feature, and `i` on the diagonal otherwise
    2. We generate features with sequence lengths `1,..,num_records` (in randomized
       order) if we generate sequence features
    """
    feature_type = SEQUENCE_FEATURES if is_sequence else SENTENCE_FEATURES
    rng = np.random.default_rng(seed)
    features_per_record = []
    sequence_lengths = (
        rng.permutation(num_records) if is_sequence else np.ones(num_records)
    )
    for record_idx in range(num_records):
        seq_len = int(sequence_lengths[record_idx])
        if is_sparse:
            matrix = scipy.sparse.eye(m=seq_len, n=feature_dim) * record_idx
        else:
            matrix = np.full(shape=(seq_len, feature_dim), fill_value=record_idx)
        features = Features(
            features=matrix,
            attribute=feature_attribute,
            origin=feature_origin,
            feature_type=feature_type,
        )
        features_per_record.append({message_attribute: [features]})
    return features_per_record, sequence_lengths


@pytest.mark.parametrize(
    "is_sparse, is_sequence, attribute, feature_attribute",
    [
        (is_sparse, is_sequence, message_attribute, feature_attribute)
        for is_sparse in [True, False]
        for is_sequence in [True, False]
        for message_attribute, feature_attribute in [
            ("something-new", "something-new"),
            (TEXT, TEXT),
            (INTENT, INTENT),
            (ENTITIES, "something-new"),
            (ENTITIES, ENTITY_ATTRIBUTE_TYPE),
            (ENTITIES, ENTITY_ATTRIBUTE_CONFIDENCE_GROUP,),
            (ENTITIES, ENTITY_ATTRIBUTE_ROLE),
        ]
    ],
)
def test_convert_to_data_format_without_fake_features_for_single_feature(
    is_sparse: bool,
    is_sequence: bool,
    message_attribute: Text,
    feature_attribute: Text,
):
    feature_dim = 3
    num_records = 10
    featurizer_name = "featurizer1"
    features, sequence_lengths = _create_dummy_features(
        message_attribute=message_attribute,
        feature_dim=feature_dim,
        feature_attribute=feature_attribute,
        feature_origin=featurizer_name,
        is_sparse=is_sparse,
        is_sequence=is_sequence,
        num_records=10,
    )

    # with pytest.warns(None) as warnings_recorder:
    converted_features, fake_features = convert_to_data_format(
        features,
        fake_features=None,
        consider_dialogue_dimension=False,
        featurizers=None,
    )

    # convert to a regular dict to not add entries by accident
    converted_features = dict(converted_features)

    feature_type = SEQUENCE_FEATURES if is_sequence else SENTENCE_FEATURES

    # One fake feature will be created for the message attribute
    assert list(fake_features.keys()) == [message_attribute]
    assert len(fake_features[message_attribute]) == 1
    assert fake_features[message_attribute][0].features.shape == (0, feature_dim)
    assert fake_features[message_attribute][0].attribute == feature_attribute
    assert fake_features[message_attribute][0].origin == featurizer_name
    assert fake_features[message_attribute][0].type == feature_type

    # Several real features could be extracted. Here, we just extract features from
    # a dataset with 1 feature. Hence, we only consider one sub-key here:
    if message_attribute == ENTITIES and feature_attribute in [
        ENTITY_ATTRIBUTE_GROUP,
        ENTITY_ATTRIBUTE_ROLE,
        ENTITY_ATTRIBUTE_TYPE,
    ]:
        # If the message attribute is entities, then all features will be collected
        # under a key that is equal to the attribute stored in the respective `Feature`:
        expected_sub_key = feature_attribute
    else:
        # For all other message attributes, the features are collected by type:
        expected_sub_key = feature_type
    assert set(converted_features[message_attribute].keys()) == {MASK, expected_sub_key}

    # There is exactly one feature array stored under the `expected_sub_key`
    assert len(converted_features[message_attribute][expected_sub_key]) == 1
    feature_array = converted_features[message_attribute][expected_sub_key][0]
    assert len(feature_array) == num_records
    for idx, feature_values in enumerate(feature_array):
        try:
            assert feature_values.shape == (sequence_lengths[idx], feature_dim)
        except:
            breakpoint()
        if not is_sparse:
            # cast is necessary because that slice is a feature array
            assert np.all(np.array(feature_values) == idx)
        else:
            # no cast necessary, this is a sparse scipy matrix
            assert np.all(feature_values.data == idx)

    # In addition to the feature extraction, a mask is constructed for the
    # message attribute:
    assert len(converted_features[message_attribute][MASK]) == 1
    mask_feature_array = converted_features[message_attribute][MASK][0]
    assert mask_feature_array.shape == (num_records, 1, 1)
    assert np.all(np.array(mask_feature_array) == 1)
