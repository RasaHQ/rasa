import typing
import copy
import numpy as np
import scipy.sparse
from collections import defaultdict, OrderedDict
from typing import List, Optional, Text, Dict, Tuple, Union, Any

from rasa.nlu.constants import TOKENS_NAMES
from rasa.utils.tensorflow.model_data import Data, FeatureArray
from rasa.utils.tensorflow.constants import MASK
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    TEXT,
    ENTITIES,
    FEATURE_TYPE_SEQUENCE,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
)

if typing.TYPE_CHECKING:
    from rasa.shared.nlu.training_data.features import Features
    from rasa.nlu.classifiers.diet_classifier import EntityTagSpec


TAG_ID_ORIGIN = "tag_id_origin"


def featurize_training_examples(
    training_examples: List[Message],
    attributes: List[Text],
    entity_tag_specs: Optional[List["EntityTagSpec"]] = None,
    featurizers: Optional[List[Text]] = None,
    bilou_tagging: bool = False,
) -> List[Dict[Text, List["Features"]]]:
    """Converts training data into a list of attribute to features.

    Possible attributes are, for example, INTENT, RESPONSE, TEXT, ACTION_TEXT,
    ACTION_NAME or ENTITIES.

    Args:
        training_examples: the list of training examples
        attributes: the attributes to consider
        entity_tag_specs: the entity specs
        featurizers: the featurizers to consider
        bilou_tagging: indicates whether BILOU tagging should be used or not

    Returns:
        A list of attribute to features.
    """
    output = []

    for example in training_examples:
        attribute_to_features = {}
        for attribute in attributes:
            if attribute == ENTITIES:
                attribute_to_features[attribute] = []
                # in case of entities add the tag_ids
                for tag_spec in entity_tag_specs:
                    attribute_to_features[attribute].append(
                        _get_tag_ids(example, tag_spec, bilou_tagging)
                    )
            elif attribute in example.data:
                attribute_to_features[attribute] = example.get_all_features(
                    attribute, featurizers
                )
        output.append(attribute_to_features)

    return output


def _get_tag_ids(
    example: Message, tag_spec: "EntityTagSpec", bilou_tagging: bool
) -> "Features":
    """Creates a feature array containing the entity tag ids of the given example."""
    from rasa.nlu.test import determine_token_labels
    from rasa.nlu.utils.bilou_utils import bilou_tags_to_ids
    from rasa.shared.nlu.training_data.features import Features

    if bilou_tagging:
        _tags = bilou_tags_to_ids(example, tag_spec.tags_to_ids, tag_spec.tag_name)
    else:
        _tags = []
        for token in example.get(TOKENS_NAMES[TEXT]):
            _tag = determine_token_labels(
                token, example.get(ENTITIES), attribute_key=tag_spec.tag_name
            )
            _tags.append(tag_spec.tags_to_ids[_tag])

    # transpose to have seq_len x 1
    return Features(
        np.array([_tags]).T, FEATURE_TYPE_SEQUENCE, tag_spec.tag_name, TAG_ID_ORIGIN
    )


def _surface_attributes(
    features: List[List[Dict[Text, List["Features"]]]],
    featurizers: Optional[List[Text]] = None,
) -> Dict[Text, List[List[List["Features"]]]]:
    """Restructure the input.

    "features" can, for example, be a dictionary of attributes (INTENT,
    TEXT, ACTION_NAME, ACTION_TEXT, ENTITIES, SLOTS, FORM) to a list of features for
    all dialogue turns in all training trackers.
    For NLU training it would just be a dictionary of attributes (either INTENT or
    RESPONSE, TEXT, and potentially ENTITIES) to a list of features for all training
    examples.

    The incoming "features" contain a dictionary as inner most value. This method
    surfaces this dictionary, so that it becomes the outer most value.

    Args:
        features: a dictionary of attributes to a list of features for all
            examples in the training data
        featurizers: the featurizers to consider

    Returns:
        A dictionary of attributes to a list of features for all examples.
    """
    # collect all attributes
    attributes = set(
        attribute
        for list_of_attribute_to_features in features
        for attribute_to_features in list_of_attribute_to_features
        for attribute in attribute_to_features.keys()
    )

    output = defaultdict(list)

    for list_of_attribute_to_features in features:
        intermediate_features = defaultdict(list)
        for attribute_to_features in list_of_attribute_to_features:
            for attribute in attributes:
                features = attribute_to_features.get(attribute)
                if featurizers:
                    features = _filter_features(features, featurizers)

                # if attribute is not present in the example, populate it with None
                intermediate_features[attribute].append(features)

        for key, value in intermediate_features.items():
            output[key].append(value)

    return output


def _filter_features(
    features: Optional[List["Features"]], featurizers: List[Text]
) -> Optional[List["Features"]]:
    """Filter the given features.

    Return only those features that are coming from one of the given featurizers.

    Args:
        features: list of features
        featurizers: names of featurizers to consider

    Returns:
        The filtered list of features.
    """
    if features is None or not featurizers:
        return features

    # it might be that the list of features also contains some tag_ids
    # the origin of the tag_ids is set to TAG_ID_ORIGIN
    # add TAG_ID_ORIGIN to the list of featurizers to make sure that we keep the
    # tag_ids
    featurizers.append(TAG_ID_ORIGIN)

    # filter the features
    return [f for f in features if f.origin in featurizers]


def _create_fake_features(
    all_features: List[List[List["Features"]]],
) -> List["Features"]:
    """Computes default feature values.

    All given features should have the same type, e.g. dense or sparse.

    Args:
        all_features: list containing all feature values encountered in the dataset
        for an attribute.

    Returns:
        The default features
    """
    example_features = next(
        iter(
            [
                list_of_features
                for list_of_list_of_features in all_features
                for list_of_features in list_of_list_of_features
                if list_of_features is not None
            ]
        )
    )

    # create fake_features for Nones
    fake_features = []
    for _features in example_features:
        new_features = copy.deepcopy(_features)
        if _features.is_dense():
            new_features.features = np.zeros(
                (0, _features.features.shape[-1]), _features.features.dtype
            )
        if _features.is_sparse():
            new_features.features = scipy.sparse.coo_matrix(
                (0, _features.features.shape[-1]), _features.features.dtype
            )
        fake_features.append(new_features)

    return fake_features


def convert_to_data_format(
    features: Union[
        List[List[Dict[Text, List["Features"]]]], List[Dict[Text, List["Features"]]]
    ],
    fake_features: Optional[Dict[Text, List["Features"]]] = None,
    consider_dialogue_dimension: bool = True,
    featurizers: Optional[List[Text]] = None,
) -> Tuple[Data, Optional[Dict[Text, List["Features"]]]]:
    """Converts the input into "Data" format.

    "features" can, for example, be a dictionary of attributes (INTENT,
    TEXT, ACTION_NAME, ACTION_TEXT, ENTITIES, SLOTS, FORM) to a list of features for
    all dialogue turns in all training trackers.
    For NLU training it would just be a dictionary of attributes (either INTENT or
    RESPONSE, TEXT, and potentially ENTITIES) to a list of features for all training
    examples.

    The "Data" format corresponds to Dict[Text, Dict[Text, List[FeatureArray]]]. It's
    a dictionary of attributes (e.g. TEXT) to a dictionary of secondary attributes
    (e.g. SEQUENCE or SENTENCE) to the list of actual features.

    Args:
        features: a dictionary of attributes to a list of features for all
            examples in the training data
        fake_features: Contains default feature values for attributes
        consider_dialogue_dimension: If set to false the dialogue dimension will be
            removed from the resulting sequence features.
        featurizers: the featurizers to consider

    Returns:
        Input in "Data" format and fake features
    """
    training = False
    if not fake_features:
        training = True
        fake_features = defaultdict(list)

    # unify format of incoming features
    if isinstance(features[0], Dict):
        features = [[dicts] for dicts in features]

    attribute_to_features = _surface_attributes(features, featurizers)

    attribute_data = {}

    # During prediction we need to iterate over the fake features attributes to

    # have all keys in the resulting model data
    if training:
        attributes = list(attribute_to_features.keys())
    else:
        attributes = list(fake_features.keys())

    # In case an attribute is not present during prediction, replace it with
    # None values that will then be replaced by fake features
    dialogue_length = 1
    num_examples = 1
    for _features in attribute_to_features.values():
        num_examples = max(num_examples, len(_features))
        dialogue_length = max(dialogue_length, len(_features[0]))
    absent_features = [[None] * dialogue_length] * num_examples

    for attribute in attributes:
        attribute_data[attribute] = _feature_arrays_for_attribute(
            attribute,
            absent_features,
            attribute_to_features,
            training,
            fake_features,
            consider_dialogue_dimension,
        )

    # ensure that all attributes are in the same order
    attribute_data = OrderedDict(sorted(attribute_data.items()))

    return attribute_data, fake_features


def _feature_arrays_for_attribute(
    attribute: Text,
    absent_features: List[Any],
    attribute_to_features: Dict[Text, List[List[List["Features"]]]],
    training: bool,
    fake_features: Dict[Text, List["Features"]],
    consider_dialogue_dimension: bool,
) -> Dict[Text, List[FeatureArray]]:
    """Create the features for the given attribute from the all examples features.

    Args:
        attribute: the attribute of Message to be featurized
        absent_features: list of Nones, used as features if `attribute_to_features`
            does not contain the `attribute`
        attribute_to_features: features for every example
        training: boolean indicating whether we are currently in training or not
        fake_features: zero features
        consider_dialogue_dimension: If set to false the dialogue dimension will be
          removed from the resulting sequence features.

    Returns:
        A dictionary of feature type to actual features for the given attribute.
    """
    features = (
        attribute_to_features[attribute]
        if attribute in attribute_to_features
        else absent_features
    )

    # in case some features for a specific attribute are
    # missing, replace them with a feature vector of zeros
    if training:
        fake_features[attribute] = _create_fake_features(features)

    (attribute_masks, _dense_features, _sparse_features) = _extract_features(
        features, fake_features[attribute], attribute
    )

    sparse_features = {}
    dense_features = {}

    for key, values in _sparse_features.items():
        if consider_dialogue_dimension:
            sparse_features[key] = FeatureArray(
                np.array(values), number_of_dimensions=4
            )
        else:
            sparse_features[key] = FeatureArray(
                np.array([v[0] for v in values]), number_of_dimensions=3
            )

    for key, values in _dense_features.items():
        if consider_dialogue_dimension:
            dense_features[key] = FeatureArray(np.array(values), number_of_dimensions=4)
        else:
            dense_features[key] = FeatureArray(
                np.array([v[0] for v in values]), number_of_dimensions=3
            )

    attribute_to_feature_arrays = {
        MASK: [FeatureArray(np.array(attribute_masks), number_of_dimensions=3)]
    }

    feature_types = set()
    feature_types.update(list(dense_features.keys()))
    feature_types.update(list(sparse_features.keys()))

    for feature_type in feature_types:
        attribute_to_feature_arrays[feature_type] = []
        if feature_type in sparse_features:
            attribute_to_feature_arrays[feature_type].append(
                sparse_features[feature_type]
            )
        if feature_type in dense_features:
            attribute_to_feature_arrays[feature_type].append(
                dense_features[feature_type]
            )

    return attribute_to_feature_arrays


def _extract_features(
    features: List[List[List["Features"]]],
    fake_features: List["Features"],
    attribute: Text,
) -> Tuple[
    List[np.ndarray],
    Dict[Text, List[List["Features"]]],
    Dict[Text, List[List["Features"]]],
]:
    """Create masks for all attributes of the given features and split the features
    into sparse and dense features.

    Args:
        features: all features
        fake_features: list of zero features

    Returns:
        - a list of attribute masks
        - a map of attribute to dense features
        - a map of attribute to sparse features
    """
    sparse_features = defaultdict(list)
    dense_features = defaultdict(list)
    attribute_masks = []

    for list_of_list_of_features in features:
        dialogue_sparse_features = defaultdict(list)
        dialogue_dense_features = defaultdict(list)

        # create a mask for every state
        # to capture which turn has which input
        attribute_mask = np.ones(len(list_of_list_of_features), np.float32)

        for i, list_of_features in enumerate(list_of_list_of_features):

            if list_of_features is None:
                # use zero features and set mask to zero
                attribute_mask[i] = 0
                list_of_features = fake_features

            for features in list_of_features:
                # in case of ENTITIES, if the attribute type matches either 'entity',
                # 'role', or 'group' the features correspond to the tag ids of that
                # entity type in order to distinguish later on between the different
                # tag ids, we use the entity type as key
                if attribute == ENTITIES and features.attribute in [
                    ENTITY_ATTRIBUTE_TYPE,
                    ENTITY_ATTRIBUTE_GROUP,
                    ENTITY_ATTRIBUTE_ROLE,
                ]:
                    key = features.attribute
                else:
                    key = features.type

                # all features should have the same types
                if features.is_sparse():
                    dialogue_sparse_features[key].append(features.features)
                else:
                    dialogue_dense_features[key].append(features.features)

        for key, value in dialogue_sparse_features.items():
            sparse_features[key].append(value)
        for key, value in dialogue_dense_features.items():
            dense_features[key].append(value)

        # add additional dimension to attribute mask
        # to get a vector of shape (dialogue length x 1),
        # the batch dim will be added later
        attribute_mask = np.expand_dims(attribute_mask, -1)
        attribute_masks.append(attribute_mask)

    return attribute_masks, dense_features, sparse_features
