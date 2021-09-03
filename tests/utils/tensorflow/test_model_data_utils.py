from typing import Any, Text, Optional, Dict, List

import pytest
import scipy.sparse
import numpy as np
import copy
import itertools

from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.nlu.constants import SPACY_DOCS
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.shared.nlu.training_data.formats.markdown import INTENT
from rasa.utils.tensorflow import model_data_utils
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import (
    ACTION_NAME,
    TEXT,
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)
from rasa.utils.tensorflow.constants import SENTENCE, SEQUENCE
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.tensorflow.model_data_utils import TAG_ID_ORIGIN

shape = 100


@pytest.mark.parametrize("seed", [123, 234])
def test_combine_attribute_features_from_message_filters_correctly(seed: int,):
    """Checks that we extract and combine the right features.

    We check that we correctly filter
    (1) which feature goes under which key in the output dictionary according to
        whether it `is_sparse` and according to it's `type` and
    (2) which features should be included according to the requested attribute and
        list of featurizers,

    Note that for (2) we create attributes and origin information at random such that
    there are exactly two attributes and two featurizers used overall and we
    request exactly one of the two attributes and one of the two origins, respectively.
    """
    rng = np.random.default_rng(seed=seed)

    # the total number of `Features` we want to create
    sparse_type_combinations = list(
        itertools.product([True, False], [SEQUENCE, SENTENCE])
    )
    num_feats_per_sparse_type_combination = 4
    total_number_of_features = (
        len(sparse_type_combinations) * num_feats_per_sparse_type_combination
    )

    # create 2 dummy attributes and featurizers ("origin")
    attribute_flag = rng.choice(2, size=total_number_of_features)
    featurizer_flag = rng.choice(2, size=total_number_of_features)
    for flags in [attribute_flag, featurizer_flag]:
        # ensure both appear by flipping first element in case all are equal
        if sum(flags) == 0 or sum(flags) == total_number_of_features:
            flags[0] = abs(1 - flags[0])

    # choose different dimensions for sparse and dense features (bool = is_sparse)
    last_dimensions = {True: 3, False: 4}

    # create dummy features
    message_features = []
    feat_idx = 1  # ensure each feature has uniue values
    for idx, (is_sparse, feature_type) in enumerate(sparse_type_combinations):
        seq_len = 1 if feature_type == SENTENCE else 4
        feat_shape = (seq_len, last_dimensions[is_sparse])
        for _ in range(num_feats_per_sparse_type_combination):
            feat_data = (
                np.eye(*feat_shape) if not is_sparse else scipy.sparse.eye(*feat_shape)
            )  # shape doesn't matter, should be checked/enforced by `Feature`
            feat_data = feat_data * feat_idx
            feat_idx += 1
            feat = Features(
                feat_data,
                feature_type=feature_type,
                attribute=str(attribute_flag[idx]),
                origin=str(featurizer_flag[idx]),
            )
            message_features.append(feat)

    # shuffle them
    rng.shuffle(message_features)

    # finally, our input message
    message = Message(features=message_features)
    requested_attribute = "0"
    requested_featurizer = "0"
    result = model_data_utils.combine_attribute_features_from_message(
        message, attribute=requested_attribute, featurizers=[requested_featurizer]
    )

    # checks
    # (1) check that result contains the type of features we expect
    for feat in message_features:
        combined_feature = result.get((feat.is_sparse(), feat.type))
        if (
            feat.attribute == requested_attribute
            and feat.origin == requested_featurizer
        ):
            assert combined_feature is not None
            # Note: Remember how we created the `feat_data` using unique indices > 0
            feat_value = (
                feat.features.data[0]
                if isinstance(feat, scipy.sparse.spmatrix)
                else int(np.max(feat.features))
            )
            assert feat_value in combined_feature
        else:
            assert combined_feature is None
    # (2) try to check that no new features were created (Note that, to be precise,
    #     we won't notice new features if they contain the special values we used above)
    for feature_list in result.values():
        for feat in feature_list:
            unique_values_geq_1 = (
                set(feat.data)
                if isinstance(feat, scipy.sparse.spmatrix)
                else set(np.unique(feat))
            ) - {0}
            assert all(
                1 <= val <= total_number_of_features + 1 for val in unique_values_geq_1
            )
    # (3) assert there are no empty lists in the result
    for feature_list in result.values():
        assert len(feature_list) > 0


@pytest.mark.parametrize("type", [SENTENCE, SEQUENCE])
def test_combine_attribute_features_from_message_raises_if_sequence_dims_are_wrong(
    type: Text,
):
    dummy_attribute = "attribute"
    dummy_origin = "origin"
    message_features = []
    for is_sparse, first_dim in [(True, 1), (False, 2)]:
        feat_shape = (first_dim, 3)
        feat_data = (
            np.eye(*feat_shape) if not is_sparse else scipy.sparse.eye(*feat_shape)
        )
        feat = Features(
            feat_data, feature_type=type, attribute=dummy_attribute, origin=dummy_origin
        )
        message_features.append(feat)
    message = Message(features=message_features)
    if type == SEQUENCE:
        error_message = (
            f"dimensions for sparse and dense {type} features don't coincide"
        )
    else:
        error_message = f"dimensions for sparse and dense {type} features aren't all 1"
    with pytest.raises(ValueError, match=error_message):
        model_data_utils.combine_attribute_features_from_message(
            message, attribute=dummy_attribute, featurizers=[dummy_origin]
        )


def test_combine_attribute_features_from_all_messages_raises_if_msgs_are_inconsistent():
    """Checks that we fail if the given messages aren't consistent.
    """
    # create one feature per type/sparseness combination
    dummy_attribute = "attribute"
    message_features = []
    for is_sparse, type in itertools.product((True, False), (SEQUENCE, SENTENCE)):
        first_dim = 1 if type == SENTENCE else 4
        feat_shape = (first_dim, 3)
        feat_data = (
            np.eye(*feat_shape) if not is_sparse else scipy.sparse.eye(*feat_shape)
        )
        feat = Features(
            feat_data, feature_type=type, attribute=dummy_attribute, origin="origin"
        )
        message_features.append(feat)
    # put each into one message
    messages = [Message(features=[feat]) for feat in message_features]
    # and see it fail
    with pytest.raises(
        ValueError, match="Expected all messages to contain the same kind of features"
    ):
        model_data_utils.combine_attribute_features_from_all_messages(
            messages, attribute=dummy_attribute, featurizers=None
        )


@pytest.mark.parametrize(
    "is_sparse_options,type_options",
    [((True, False), (SEQUENCE, SENTENCE)), ((True,), (SENTENCE,))],
)
def test_combine_attribute_features_from_all_messages_extracts_all(
    is_sparse_options: List[bool], type_options: List[Text],
):
    # create 2 feature per type/sparseness/attribute/origin combination
    # where we choose 2 > 1 to ensure that the "combination" works
    features_per_combination_and_message = 2
    attributes = ["a1", "a2"]
    origins = ["o1", "o2"]
    message_features = []
    sparseness_and_type_combinations = list(
        itertools.product(is_sparse_options, type_options)
    )
    for attribute, origin in zip(attributes, origins):
        for is_sparse, type in sparseness_and_type_combinations:
            first_dim = 1 if type == SENTENCE else 4
            feat_shape = (first_dim, 3)
            feat_data = (
                np.eye(*feat_shape) if not is_sparse else scipy.sparse.eye(*feat_shape)
            )
            feat = Features(
                feat_data, feature_type=type, attribute=attribute, origin=origin
            )
            message_features.extend([feat] * features_per_combination_and_message)
    # create messages that are all alike...
    num_messages = 7
    messages = [Message(features=message_features) for _ in range(num_messages)]
    (
        sequence_features,
        sentence_features,
    ) = model_data_utils.combine_attribute_features_from_all_messages(
        messages, attribute=attributes[0], featurizers=None
    )
    for feature_array_list, type in [
        (sequence_features, SEQUENCE),
        (sentence_features, SENTENCE),
    ]:
        if type in type_options:
            assert len(feature_array_list) == len(is_sparse_options)
            for feature_array in feature_array_list:
                assert feature_array.shape[0] == num_messages
            if True in is_sparse_options:
                assert feature_array_list[0].is_sparse
            if False in is_sparse_options:
                assert not feature_array_list[-1].is_sparse
        else:
            assert len(feature_array_list) == 0


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
    "text, intent, entities, attributes, real_sparse_feature_sizes, type",
    [
        (
            "Hello!",
            "greet",
            None,
            [TEXT],
            {"text": {"sequence": [1], "sentence": [1]}},
            None,
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
            None,
        ),
        (
            "Hello!",
            "greet",
            None,
            [TEXT, INTENT],
            {"text": {"sentence": [1]},},  # due to sentence type filter
            SENTENCE,
        ),
        (
            "Hello Max!",
            "greet",
            [{"entity": "name", "value": "Max", "start": 6, "end": 9}],
            [TEXT, ENTITIES],
            {"text": {"sequence": [2], "sentence": [2]}},
            None,
        ),
    ],
)
def test_convert_training_examples(
    spacy_nlp: Any,
    text: Text,
    intent: Optional[Text],
    entities: Optional[List[Dict[Text, Any]]],
    attributes: List[Text],
    real_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    type: Optional[Text],
):
    message = Message(data={TEXT: text, INTENT: intent, ENTITIES: entities})

    tokenizer = SpacyTokenizer()
    count_vectors_featurizer = CountVectorsFeaturizer()
    spacy_featurizer = SpacyFeaturizer()

    message.set(SPACY_DOCS[TEXT], spacy_nlp(text))

    training_data = TrainingData([message])
    tokenizer.train(training_data)
    count_vectors_featurizer.train(training_data)
    spacy_featurizer.train(training_data)

    entity_tag_spec = [
        EntityTagSpec(
            "entity",
            {0: "O", 1: "name", 2: "location"},
            {"O": 0, "name": 1, "location": 2},
            3,
        )
    ]
    output, sparse_feature_sizes = model_data_utils.featurize_training_examples(
        [message], attributes=attributes, entity_tag_specs=entity_tag_spec, type=type,
    )

    assert len(output) == 1
    for attribute in attributes:
        assert attribute in output[0]
    for attribute in {INTENT, TEXT, ENTITIES} - set(attributes):
        assert attribute not in output[0]
    if type is None:
        # we have sparse sentence, sparse sequence, dense sentence, and dense sequence
        # features in the list
        assert len(output[0][TEXT]) == 4
        if INTENT in attributes:
            # we will just have sparse sequence features
            assert len(output[0][INTENT]) == 1
    elif type == SENTENCE:
        assert len(output[0][TEXT]) == 2
        if INTENT in attributes:
            # we will just have sparse sequence features - and filter them out
            assert len(output[0][INTENT]) == 0
    else:
        raise NotImplementedError(
            f"Expected None or {SENTENCE} for type but received {type}."
        )
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
