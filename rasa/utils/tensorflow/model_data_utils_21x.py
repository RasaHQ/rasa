import typing
from typing import List, Optional, Text, Dict, Tuple, Union, Any
import copy
import numpy as np
from collections import defaultdict, OrderedDict
import scipy.sparse

from rasa.utils.tensorflow.model_data import FeatureArray, Data
from rasa.utils.tensorflow.constants import SEQUENCE, SENTENCE
from rasa.shared.nlu.training_data.features import Features

MASK = "mask"


def surface_attributes(
    tracker_state_features: List[List[Dict[Text, List[Features]]]]
) -> Dict[Text, List[List[List[Features]]]]:
    """Restructure the input.
    Args:
        tracker_state_features: a dictionary of attributes (INTENT, TEXT, ACTION_NAME,
            ACTION_TEXT, ENTITIES, SLOTS, FORM) to a list of features for all
            dialogue turns in all training trackers
    Returns:
        A dictionary of attributes to a list of features for all dialogue turns
        and all training trackers.
    """
    # collect all attributes
    attributes = set(
        attribute
        for features_in_tracker in tracker_state_features
        for features_in_turn in features_in_tracker
        for attribute in features_in_turn.keys()
    )

    attribute_to_features = defaultdict(list)

    for features_in_tracker in tracker_state_features:
        intermediate_features = defaultdict(list)

        for features_in_dialogue in features_in_tracker:
            for attribute in attributes:
                # if attribute is not present in the example, populate it with None
                intermediate_features[attribute].append(
                    features_in_dialogue.get(attribute)
                )

        for key, value in intermediate_features.items():
            attribute_to_features[key].append(value)

    return attribute_to_features


def create_zero_features(
    tracker_features: List[List[List[Features]]],
) -> List[Features]:
    # all features should have the same types
    """
    Computes default feature values for an attribute;
    Args:
        tracker_features: list containing all feature values encountered
        in the dataset for an attribute;
    """

    example_features = next(
        iter(
            [
                list_of_features
                for turn_features in tracker_features
                for list_of_features in turn_features
                if list_of_features is not None
            ]
        )
    )

    # create zero_features for nones
    zero_features = []
    for features in example_features:
        new_features = copy.deepcopy(features)
        if features.is_dense():
            new_features.features = np.zeros_like(features.features)
        if features.is_sparse():
            new_features.features = scipy.sparse.coo_matrix(
                features.features.shape, features.features.dtype
            )
        zero_features.append(new_features)

    return zero_features


def convert_to_data_format(
    tracker_state_features: Union[
        List[List[Dict[Text, List[Features]]]], List[Dict[Text, List[Features]]]
    ],
    zero_state_features: Optional[Dict[Text, List[Features]]] = None,
) -> Tuple[Data, Optional[Dict[Text, List[Features]]]]:
    """Converts the input into "Data" format.
    Args:
        tracker_state_features: a dictionary of attributes (INTENT, TEXT, ACTION_NAME,
            ACTION_TEXT, ENTITIES, SLOTS, FORM) to a list of features for all
            dialogue turns in all training trackers
        zero_state_features: Contains default feature values for attributes
    Returns:
        Input in "Data" format and zero state features
    """
    training = False
    if not zero_state_features:
        training = True
        zero_state_features = defaultdict(list)

    # unify format of incoming features
    if isinstance(tracker_state_features[0], Dict):
        tracker_state_features = [[dicts] for dicts in tracker_state_features]

    state_to_tracker_features = surface_attributes(tracker_state_features)

    attribute_data = {}

    # During prediction we need to iterate over the zero features attributes to
    # have all keys in the resulting model data
    if training:
        attributes = list(state_to_tracker_features.keys())
    else:
        attributes = list(zero_state_features.keys())

    # In case an attribute is not present during prediction, replace it with
    # None values that will then be replaced by zero features
    dialogue_length = 1
    for tracker_features in state_to_tracker_features.values():
        dialogue_length = max(dialogue_length, len(tracker_features[0]))
    empty_features = [[None] * dialogue_length]

    for attribute in attributes:
        attribute_data[attribute] = _features_for_attribute(
            attribute,
            empty_features,
            state_to_tracker_features,
            training,
            zero_state_features,
        )

    # ensure that all attributes are in the same order
    attribute_data = OrderedDict(sorted(attribute_data.items()))

    return attribute_data, zero_state_features


def _features_for_attribute(
    attribute: Text,
    empty_features: List[Any],
    state_to_tracker_features: Dict[Text, List[List[List[Features]]]],
    training: bool,
    zero_state_features: Dict[Text, List[Features]],
) -> Dict[Text, List[FeatureArray]]:
    """Create the features for the given attribute from the tracker features.
    Args:
        attribute: the attribute
        empty_features: empty features
        state_to_tracker_features: tracker features for every state
        training: boolean indicating whether we are currently in training or not
        zero_state_features: zero features
    Returns:
        A dictionary of feature type to actual features for the given attribute.
    """
    tracker_features = (
        state_to_tracker_features[attribute]
        if attribute in state_to_tracker_features
        else empty_features
    )

    # in case some features for a specific attribute and dialogue turn are
    # missing, replace them with a feature vector of zeros
    if training:
        zero_state_features[attribute] = create_zero_features(tracker_features)

    (attribute_masks, _dense_features, _sparse_features) = map_tracker_features(
        tracker_features, zero_state_features[attribute]
    )

    list_of_collated_sentence_feature_batches: List[FeatureArray] = []
    for feature_batch_collection in [
        _sparse_features,
        _dense_features,
    ]:
        batch_of_feature_lists = feature_batch_collection.get(SENTENCE, None)
        if not batch_of_feature_lists:
            continue
        # For each record, stack all the features of the same kind, and wrap the
        # resulting batch of stacked features into a feature array.
        collated_sentence_features = FeatureArray(
            np.array(
                [
                    Features.combine(feature_list, horizontal=False).features
                    for feature_list in batch_of_feature_lists
                ]
            ),
            number_of_dimensions=3,
        )
        list_of_collated_sentence_feature_batches.append(collated_sentence_features)

    return {
        MASK: [FeatureArray(np.array(attribute_masks), number_of_dimensions=3)],
        SENTENCE: list_of_collated_sentence_feature_batches,
    }


def map_tracker_features(
    tracker_features: List[List[List[Features]]], zero_features: List[Features]
) -> Tuple[
    List[np.ndarray],
    Dict[Text, List[List[Features]]],
    Dict[Text, List[List[Features]]],
]:
    """Create masks for all attributes of the given features and split the features
    into sparse and dense features.
    Args:
        tracker_features: all features
        zero_features: list of zero features
    Returns:
        - a list of attribute masks
        - a map of attribute to dense features
        - a map of attribute to sparse features
    """
    sparse_features = defaultdict(list)
    dense_features = defaultdict(list)
    attribute_masks = []

    for turn_features in tracker_features:
        dialogue_sparse_features = defaultdict(list)
        dialogue_dense_features = defaultdict(list)

        # create a mask for every state
        # to capture which turn has which input
        attribute_mask = np.expand_dims(np.ones(len(turn_features), np.float32), -1)

        for i, list_of_features in enumerate(turn_features):

            if list_of_features is None:
                # use zero features and set mask to zero
                attribute_mask[i] = 0
                list_of_features = zero_features

            for features in list_of_features:
                # all features should have the same types
                if features.is_sparse():
                    dialogue_sparse_features[features.type].append(features)
                else:
                    dialogue_dense_features[features.type].append(features)

        for key, value in dialogue_sparse_features.items():
            sparse_features[key].append(value)
        for key, value in dialogue_dense_features.items():
            dense_features[key].append(value)

        attribute_masks.append(attribute_mask)

    return attribute_masks, dense_features, sparse_features
