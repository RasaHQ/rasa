# import copy
# from typing import Union, List

import pytest
import tensorflow as tf

# import scipy.sparse
# import numpy as np

from rasa.utils.tensorflow.rasa_layers import (
    ConcatenateSparseDenseFeatures,
    RasaFeatureCombiningLayer,
    RasaSequenceLayer,
)
from rasa.utils.tensorflow.model_data import FeatureSignature

from rasa.utils.tensorflow.constants import (
    DENSE_INPUT_DROPOUT,
    SPARSE_INPUT_DROPOUT,
    DROP_RATE,
    DENSE_DIMENSION,
    REGULARIZATION_CONSTANT,
    CONCAT_DIMENSION,
    WEIGHT_SPARSITY,
    DROP_RATE_ATTENTION,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    UNIDIRECTIONAL_ENCODER,
    HIDDEN_LAYERS_SIZES,
    NUM_TRANSFORMER_LAYERS,
    TRANSFORMER_SIZE,
    NUM_HEADS,
)

# data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]]
# {'label':
# DictWrapper({'ids': ListWrapper([FeatureSignature(is_sparse=False, units=1, number_of_dimensions=2)]), 'mask': ListWrapper([FeatureSignature(is_sparse=False, units=1, number_of_dimensions=3)]), 'sentence': ListWrapper([FeatureSignature(is_sparse=True, units=2033, number_of_dimensions=3)]), 'sequence': ListWrapper([FeatureSignature(is_sparse=True, units=2033, number_of_dimensions=3)]), 'sequence_lengths': ListWrapper([FeatureSignature(is_sparse=False, units=4, number_of_dimensions=1)])}),
# 'text': DictWrapper({'mask': ListWrapper([FeatureSignature(is_sparse=False, units=1, number_of_dimensions=3)]), 'sentence': ListWrapper([FeatureSignature(is_sparse=True, units=2109, number_of_dimensions=3)]), 'sequence': ListWrapper([FeatureSignature(is_sparse=True, units=2109, number_of_dimensions=3)]), 'sequence_lengths': ListWrapper([FeatureSignature(is_sparse=False, units=4, number_of_dimensions=1)])})}

attribute_name = "att1"
units_small = 2
units_bigger = 3
units_sparse_to_dense = 10
units_concat = 7
units_hidden_layer = 11
units_transformer = 14
num_transformer_heads = 2
batch_size = 5
max_seq_length = 3

feature_signature_sparse = FeatureSignature(
    is_sparse=True, units=units_small, number_of_dimensions=3
)

feature_signature_dense = FeatureSignature(
    is_sparse=False, units=units_small, number_of_dimensions=3
)
feature_dense_seq_3d = tf.ones([batch_size, max_seq_length, units_small])
feature_dense_sent_3d = tf.ones([batch_size, 1, units_small])

feature_signature_dense_bigger = FeatureSignature(
    is_sparse=False, units=units_bigger, number_of_dimensions=3
)

attribute_signature_basic = {
    "sequence": [feature_signature_dense, feature_signature_sparse],
    "sentence": [feature_signature_dense],
}


model_config_basic = {
    DENSE_INPUT_DROPOUT: False,
    SPARSE_INPUT_DROPOUT: False,
    DROP_RATE: 0.5,
    DENSE_DIMENSION: {attribute_name: units_sparse_to_dense},
    REGULARIZATION_CONSTANT: 0.001,
    CONCAT_DIMENSION: {attribute_name: units_concat},
    WEIGHT_SPARSITY: 0.5,
    HIDDEN_LAYERS_SIZES: {attribute_name: [units_hidden_layer]},
    NUM_TRANSFORMER_LAYERS: 0,
    TRANSFORMER_SIZE: None,
    UNIDIRECTIONAL_ENCODER: None,
}

model_config_basic_no_hidden_layers = dict(
    model_config_basic, **{HIDDEN_LAYERS_SIZES: {attribute_name: []}}
)

model_config_transformer = dict(
    model_config_basic,
    **{
        DROP_RATE_ATTENTION: 0.5,
        KEY_RELATIVE_ATTENTION: True,
        VALUE_RELATIVE_ATTENTION: True,
        MAX_RELATIVE_POSITION: 10,
        UNIDIRECTIONAL_ENCODER: False,
        NUM_TRANSFORMER_LAYERS: {attribute_name: 2},
        TRANSFORMER_SIZE: {attribute_name: units_transformer},
        NUM_HEADS: num_transformer_heads,
    },
)


@pytest.mark.parametrize(
    "layer_class, model_config, layer_args, expected_output_units",
    [
        (
            ConcatenateSparseDenseFeatures,
            model_config_basic,
            {
                "feature_type": "arbitrary",
                "feature_type_signature": [
                    feature_signature_sparse,
                    feature_signature_sparse,
                    feature_signature_dense,
                    feature_signature_dense_bigger,
                ],
            },
            2 * units_sparse_to_dense + units_small + units_bigger,
        ),
        (
            ConcatenateSparseDenseFeatures,
            model_config_basic,
            {
                "feature_type": "arbitrary",
                "feature_type_signature": [feature_signature_sparse],
            },
            units_sparse_to_dense,
        ),
        (
            RasaFeatureCombiningLayer,
            model_config_basic,
            {"attribute_signature": attribute_signature_basic},
            units_concat,
        ),
        (
            RasaFeatureCombiningLayer,
            model_config_basic,
            {
                "attribute_signature": {
                    "sequence": [],
                    "sentence": [feature_signature_dense],
                },
            },
            units_small,
        ),
        (
            RasaSequenceLayer,
            model_config_transformer,
            {"attribute_signature": attribute_signature_basic},
            units_transformer,
        ),
        (
            RasaSequenceLayer,
            model_config_basic,
            {"attribute_signature": attribute_signature_basic},
            units_hidden_layer,
        ),
        (
            RasaSequenceLayer,
            model_config_basic_no_hidden_layers,
            {"attribute_signature": attribute_signature_basic},
            units_concat,
        ),
    ],
)
def test_layer_gives_correct_output_units(
    layer_class, model_config, layer_args, expected_output_units
):
    layer = layer_class(**layer_args, config=model_config, attribute=attribute_name)
    assert layer.output_units == expected_output_units


def test_correct_output_shape():
    layer = ConcatenateSparseDenseFeatures(
        attribute=attribute_name,
        feature_type="arbitrary",
        feature_type_signature=[
            # feature_signature_sparse,
            feature_signature_dense,
            # feature_signature_dense_bigger,
        ],
        config=model_config_basic,
    )
    outputs = layer(([feature_dense_seq_3d],))

    assert outputs.shape == (batch_size, max_seq_length, units_small)
