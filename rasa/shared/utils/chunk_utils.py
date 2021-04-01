import logging
import numpy as np
from typing import Dict, List, Optional, Text, Tuple, Union

import tensorflow as tf
import scipy.sparse

from rasa.shared.utils.io import DEFAULT_ENCODING
from rasa.shared.nlu.training_data.features import Features

DEFAULT_TRAINING_DATA_OUTPUT_PATH = "training_data.yml"

logger = logging.getLogger(__name__)

TF_RECORD_KEY_SEPARATOR = "#"


def encode_features(
    features: List[Features], index: Optional[Text] = None,
) -> Dict[Text, tf.train.Feature]:
    tf_features = {}

    for feature in features:
        key = _construct_tf_record_key(
            feature.attribute, feature.type, feature.origin, feature.is_dense(), index
        )

        if feature.is_dense():
            tf_features[key] = bytes_feature(feature.features)
        else:
            data = feature.features.data.astype(np.int64)
            shape = feature.features.shape
            row = feature.features.row
            column = feature.features.col

            tf_features[f"{key}{TF_RECORD_KEY_SEPARATOR}data"] = int_feature(data)
            tf_features[f"{key}{TF_RECORD_KEY_SEPARATOR}shape"] = int_feature(shape)
            tf_features[f"{key}{TF_RECORD_KEY_SEPARATOR}row"] = int_feature(row)
            tf_features[f"{key}{TF_RECORD_KEY_SEPARATOR}column"] = int_feature(column)

    return tf_features


def _construct_tf_record_key(
    attribute: Text,
    feature_type: Text,
    origin: Text,
    is_dense: bool,
    index: Optional[Text] = None,
) -> Text:
    prefix = (
        f"{attribute}{TF_RECORD_KEY_SEPARATOR}{feature_type}"
        f"{TF_RECORD_KEY_SEPARATOR}{origin}{TF_RECORD_KEY_SEPARATOR}"
    )
    if index is not None:
        prefix = f"{index}{TF_RECORD_KEY_SEPARATOR}{prefix}"

    if is_dense:
        return f"{prefix}dense"
    return f"{prefix}sparse"


def int_feature(array: Union[Tuple[int], np.ndarray]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=array))


def bytes_feature(array: Union[np.ndarray, Text]) -> tf.train.Feature:
    if isinstance(array, np.ndarray):
        value = tf.io.serialize_tensor(array.astype(np.float64))
    else:
        value = bytes(array, encoding=DEFAULT_ENCODING)

    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def decode_features(
    example: tf.train.Example, key: Text, index: Optional[Text] = None
) -> Optional[Features]:
    (
        attribute,
        feature_type,
        origin,
        is_dense,
        extra_info,
    ) = _deconstruct_tf_record_key(key)

    if is_dense:
        return _convert_to_numpy(example, attribute, feature_type, origin, index)
    elif not is_dense and extra_info == "data":
        return _convert_to_sparse_matrix(
            example, attribute, feature_type, origin, index
        )


def _deconstruct_tf_record_key(key: Text) -> Tuple[Text, Text, Text, bool, Text]:
    parts = key.split(TF_RECORD_KEY_SEPARATOR)

    attribute = parts[0]
    feature_type = parts[1]
    origin = parts[2]
    dense = parts[3] == "dense"
    extra_info = parts[4] if not dense else ""

    return attribute, feature_type, origin, dense, extra_info


def _convert_to_numpy(
    example: tf.train.Example,
    attribute: Text,
    feature_type: Text,
    origin: Text,
    index: Optional[Text] = None,
) -> Features:
    key = _construct_tf_record_key(attribute, feature_type, origin, True, index)

    bytes_list = example.features.feature[key].bytes_list.value[0]
    data = tf.io.parse_tensor(bytes_list, out_type=tf.float64).numpy()

    return Features(data, feature_type, attribute, origin)


def _convert_to_sparse_matrix(
    example: tf.train.Example,
    attribute: Text,
    feature_type: Text,
    origin: Text,
    index: Optional[Text] = None,
) -> Features:
    prefix = _construct_tf_record_key(attribute, feature_type, origin, False, index)

    shape = example.features.feature[
        f"{prefix}{TF_RECORD_KEY_SEPARATOR}shape"
    ].int64_list.value
    data = example.features.feature[
        f"{prefix}{TF_RECORD_KEY_SEPARATOR}data"
    ].int64_list.value
    row = example.features.feature[
        f"{prefix}{TF_RECORD_KEY_SEPARATOR}row"
    ].int64_list.value
    column = example.features.feature[
        f"{prefix}{TF_RECORD_KEY_SEPARATOR}column"
    ].int64_list.value

    return Features(
        scipy.sparse.coo_matrix((data, (row, column)), shape),
        feature_type,
        attribute,
        origin,
    )
