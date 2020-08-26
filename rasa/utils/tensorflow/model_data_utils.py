from typing import Any, List, Optional, Text, Dict, Tuple, Union
import copy
import numpy as np
from collections import defaultdict
import scipy.sparse

from rasa.utils.tensorflow.model_data import Data
from rasa.utils.tensorflow.constants import SEQUENCE
MASK = "mask"

def surface_attributes(
    features: List[List[Dict[Text, List["Features"]]]]
) -> Dict[Text, List[List[List["Features"]]]]:
    """Restructure the input.

    Args:
        features: a dictionary of attributes (INTENT, TEXT, ACTION_NAME,
            ACTION_TEXT, ENTITIES, SLOTS, FORM) to a list of features for all
            dialogue turns in all training trackers

    Returns:
        A dictionary of attributes to a list of features for all dialogue turns
        and all training trackers.
    """
    # collect all attributes
    attributes = set(
        attribute
        for features_in_tracker in features
        for features_in_dialogue in features_in_tracker
        for attribute in features_in_dialogue.keys()
    )

    attribute_to_features = defaultdict(list)

    for features_in_tracker in features:
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
    features: List[List[List["Features"]]],
) -> List["Features"]:
    # all features should have the same types
    """
    Computes default feature values for an attribute;
    Args:
        features: list containing all feature values encountered
        in the dataset for an attribute;
    """

    example_features = next(
        iter(
            [
                features_in_dialogue
                for features_in_tracker in features
                for features_in_dialogue in features_in_tracker
                if features_in_dialogue is not None
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
    features: Union[
        List[List[Dict[Text, List["Features"]]]], List[Dict[Text, List["Features"]]]
    ],
    zero_features: Optional[Dict] = {},
    training: bool = True,
) -> Data:
    """Converts the input into "Data" format.

    Args:
        features: a dictionary of attributes (INTENT, TEXT, ACTION_NAME,
            ACTION_TEXT, ENTITIES, SLOTS, FORM) to a list of features for all
            dialogue turns in all training trackers

    Returns:
        Input in "Data" format.
    """
    if training:
        zero_features = defaultdict(list)
    else:
        zero_features = zero_features
    remove_sequence_dimension = False
    # unify format of incoming features
    if isinstance(features[0], Dict):
        features = [[dicts] for dicts in features]
        remove_sequence_dimension = True

    features = surface_attributes(features)

    attribute_data = {}

    # During prediction we need to iterate over the zero features attributes to
    # have all keys in the resulting model data
    if training:
        attributes = list(features.keys())
    else:
        attributes = list(zero_features.keys())

    dialogue_length = 1
    for key, values in features.items():
        dialogue_length = max(dialogue_length, len(values[0]))

    empty_features = [[None] * dialogue_length]

    for attribute in attributes:
        features_in_tracker = (
            features[attribute] if attribute in features else empty_features
        )

        # in case some features for a specific attribute and dialogue turn are
        # missing, replace them with a feature vector of zeros
        if training:
            zero_features[attribute] = create_zero_features(
                features_in_tracker
            )

        (
            attribute_masks,
            _dense_features,
            _sparse_features,
        ) = map_tracker_features(
            features_in_tracker, zero_features[attribute]
        )

        sparse_features = defaultdict(list)
        dense_features = defaultdict(list)

        if remove_sequence_dimension:
            # remove added sequence dimension
            for key, values in _sparse_features.items():
                sparse_features[key] = [value[0] for value in values]
            for key, values in _dense_features.items():
                dense_features[key] = [value[0] for value in values]
        else:
            for key, values in _sparse_features.items():
                sparse_features[key] = [
                    scipy.sparse.vstack(value) for value in values
                ]
            for key, values in _dense_features.items():
                dense_features[key] = [np.vstack(value) for value in values]

        # TODO not sure about expand_dims
        attribute_features = {MASK: [np.array(attribute_masks)]}

        feature_types = set()
        feature_types.update(list(dense_features.keys()))
        feature_types.update(list(sparse_features.keys()))

        for feature_type in feature_types:
            if feature_type == SEQUENCE:
                # TODO we don't take sequence features because that makes us deal
                #  with 4D sparse tensors
                continue

            attribute_features[feature_type] = []
            if feature_type in sparse_features:
                attribute_features[feature_type].append(
                    np.array(sparse_features[feature_type])
                )
            if feature_type in dense_features:
                attribute_features[feature_type].append(
                    np.array(dense_features[feature_type])
                )

        attribute_data[attribute] = attribute_features
        
    return attribute_data, zero_features

def map_tracker_features(
    features_in_tracker: List[List[List["Features"]]],
    zero_features: List["Features"],
) -> Tuple[
    List[np.ndarray],
    Dict[Text, List[List["Features"]]],
    Dict[Text, List[List["Features"]]],
]:
    """Create masks for all attributes of the given features and split the features
    into sparse and dense features.

    Args:
        features_in_tracker: all features
        zero_features: list of zero features

    Returns:
        - a list of attribute masks
        - a map of attribute to dense features
        - a map of attribute to sparse features
    """
    sparse_features = defaultdict(list)
    dense_features = defaultdict(list)
    attribute_masks = []

    for features_in_dialogue in features_in_tracker:
        dialogue_sparse_features = defaultdict(list)
        dialogue_dense_features = defaultdict(list)

        # create a mask for every state
        # to capture which turn has which input
        attribute_mask = np.expand_dims(
            np.ones(len(features_in_dialogue), np.float32), -1
        )

        for i, turn_features in enumerate(features_in_dialogue):

            if turn_features is None:
                # use zero features and set mask to zero
                attribute_mask[i] = 0
                turn_features = zero_features

            for features in turn_features:
                # all features should have the same types
                if features.is_sparse():
                    dialogue_sparse_features[features.type].append(
                        features.features
                    )
                else:
                    dialogue_dense_features[features.type].append(features.features)

        for key, value in dialogue_sparse_features.items():
            sparse_features[key].append(value)
        for key, value in dialogue_dense_features.items():
            dense_features[key].append(value)

        attribute_masks.append(attribute_mask)

    return attribute_masks, dense_features, sparse_features