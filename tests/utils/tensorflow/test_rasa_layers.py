# import copy
# from typing import Union, List

import pytest
import tensorflow as tf

# import scipy.sparse
# import numpy as np

from rasa.shared.nlu.constants import TEXT
from rasa.utils.tensorflow.rasa_layers import (
    ConcatenateSparseDenseFeatures,
    RasaFeatureCombiningLayer,
    RasaSequenceLayer,
)
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
    SENTENCE,
    SEQUENCE,
    MASKED_LM,
)
from rasa.utils.tensorflow.exceptions import TFLayerConfigException
from rasa.utils.tensorflow.model_data import FeatureSignature

# data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]]
# {'label':
# DictWrapper({'ids': ListWrapper([FeatureSignature(is_sparse=False, units=1, number_of_dimensions=2)]), 'mask': ListWrapper([FeatureSignature(is_sparse=False, units=1, number_of_dimensions=3)]), 'sentence': ListWrapper([FeatureSignature(is_sparse=True, units=2033, number_of_dimensions=3)]), 'sequence': ListWrapper([FeatureSignature(is_sparse=True, units=2033, number_of_dimensions=3)]), 'sequence_lengths': ListWrapper([FeatureSignature(is_sparse=False, units=4, number_of_dimensions=1)])}),
# 'text': DictWrapper({'mask': ListWrapper([FeatureSignature(is_sparse=False, units=1, number_of_dimensions=3)]), 'sentence': ListWrapper([FeatureSignature(is_sparse=True, units=2109, number_of_dimensions=3)]), 'sequence': ListWrapper([FeatureSignature(is_sparse=True, units=2109, number_of_dimensions=3)]), 'sequence_lengths': ListWrapper([FeatureSignature(is_sparse=False, units=4, number_of_dimensions=1)])})}

attribute_name = TEXT
units_small = 2
units_bigger = 3
units_sparse_to_dense = 10
units_concat = 7
units_hidden_layer = 11
units_transformer = 14
num_transformer_heads = 2
num_transformer_layers = 2
batch_size = 5
max_seq_length = 3

feature_signature_sparse = FeatureSignature(
    is_sparse=True, units=units_small, number_of_dimensions=3
)
feature_sparse_seq_3d = tf.sparse.from_dense(
    tf.ones((batch_size, max_seq_length, units_small))
)
feature_sparse_sent_3d = tf.sparse.from_dense(tf.ones((batch_size, 1, units_small)))

feature_signature_dense = FeatureSignature(
    is_sparse=False, units=units_small, number_of_dimensions=3
)
feature_dense_seq_3d = tf.ones((batch_size, max_seq_length, units_small))
feature_dense_sent_3d = tf.ones((batch_size, 1, units_small))

feature_signature_dense_bigger = FeatureSignature(
    is_sparse=False, units=units_bigger, number_of_dimensions=3
)
feature_dense_seq_3d_bigger = tf.ones((batch_size, max_seq_length, units_bigger))
feature_dense_sent_3d_bigger = tf.ones((batch_size, 1, units_bigger))

sequence_lengths = tf.ones((batch_size,)) * max_seq_length
sequence_lengths_empty = tf.ones((batch_size,)) * 0


attribute_signature_basic = {
    SEQUENCE: [feature_signature_dense, feature_signature_sparse],
    SENTENCE: [feature_signature_dense],
}
attribute_features_basic = (
    [feature_dense_seq_3d, feature_sparse_seq_3d],
    [feature_dense_sent_3d],
    sequence_lengths,
)

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
    MASKED_LM: False,
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
        NUM_TRANSFORMER_LAYERS: {attribute_name: num_transformer_layers},
        TRANSFORMER_SIZE: {attribute_name: units_transformer},
        NUM_HEADS: num_transformer_heads,
    },
)

model_config_transformer_mlm = dict(model_config_transformer, **{MASKED_LM: True},)


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


@pytest.mark.parametrize(
    "layer_class, model_config, layer_args, layer_inputs, expected_output_shapes",
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
            (
                [
                    feature_sparse_seq_3d,
                    feature_sparse_seq_3d,
                    feature_dense_seq_3d,
                    feature_dense_seq_3d_bigger,
                ],
            ),
            (
                batch_size,
                max_seq_length,
                2 * units_sparse_to_dense + units_small + units_bigger,
            ),
        ),
        (
            ConcatenateSparseDenseFeatures,
            model_config_basic,
            {
                "feature_type": "arbitrary",
                "feature_type_signature": [feature_signature_sparse],
            },
            ([feature_sparse_sent_3d],),
            (batch_size, 1, units_sparse_to_dense),
        ),
        (
            RasaFeatureCombiningLayer,
            model_config_basic,
            {"attribute_signature": attribute_signature_basic},
            attribute_features_basic,
            [
                (batch_size, max_seq_length + 1, units_concat),
                (batch_size, max_seq_length + 1, 1),
            ],
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
            ([], [feature_dense_sent_3d], sequence_lengths_empty),
            [(batch_size, 1, units_small), (batch_size, 1, 1)],
        ),
        (
            RasaSequenceLayer,
            model_config_transformer_mlm,
            {"attribute_signature": attribute_signature_basic},
            attribute_features_basic,
            [
                (batch_size, max_seq_length + 1, units_transformer),
                (batch_size, max_seq_length + 1, units_hidden_layer),
                (batch_size, max_seq_length + 1, 1),
                (batch_size, max_seq_length + 1, units_small),
                (batch_size, max_seq_length + 1, 1),
                (
                    num_transformer_layers,
                    batch_size,
                    num_transformer_heads,
                    max_seq_length + 1,
                    max_seq_length + 1,
                ),
            ],
        ),
        (
            RasaSequenceLayer,
            model_config_basic,
            {"attribute_signature": attribute_signature_basic},
            attribute_features_basic,
            [
                (batch_size, max_seq_length + 1, units_hidden_layer),
                (batch_size, max_seq_length + 1, units_hidden_layer),
                (batch_size, max_seq_length + 1, 1),
                (0,),
                (0,),
                (0,),
            ],
        ),
        (
            RasaSequenceLayer,
            model_config_basic_no_hidden_layers,
            {"attribute_signature": attribute_signature_basic},
            attribute_features_basic,
            [
                (batch_size, max_seq_length + 1, units_concat),
                (batch_size, max_seq_length + 1, units_concat),
                (batch_size, max_seq_length + 1, 1),
                (0,),
                (0,),
                (0,),
            ],
        ),
    ],
)
def test_correct_output_shape(
    layer_class, model_config, layer_args, layer_inputs, expected_output_shapes
):
    layer = layer_class(**layer_args, attribute=attribute_name, config=model_config,)
    outputs = layer(layer_inputs, training=True)

    if isinstance(expected_output_shapes, list):
        for i, expected_shape in enumerate(expected_output_shapes):
            assert outputs[i].shape == expected_shape
    else:
        assert outputs.shape == expected_output_shapes


@pytest.mark.parametrize(
    "layer_class, layer_args",
    [
        (
            ConcatenateSparseDenseFeatures,
            {"feature_type": "arbitrary", "feature_type_signature": []},
        ),
        (
            RasaFeatureCombiningLayer,
            {"attribute_signature": {"sequence": [], "sentence": []}},
        ),
        (
            RasaSequenceLayer,
            {
                "attribute_signature": {
                    "sequence": [],
                    "sentence": [feature_dense_sent_3d],
                }
            },
        ),
    ],
)
def test_raises_exception_when_missing_features(layer_class, layer_args):
    with pytest.raises(TFLayerConfigException):
        layer_class(**layer_args, attribute=attribute_name, config=model_config_basic)


def test_concat_sparse_dense_correct_output_for_dense_input():
    layer = ConcatenateSparseDenseFeatures(
        attribute=attribute_name,
        feature_type=SEQUENCE,
        feature_type_signature=[
            FeatureSignature(is_sparse=False, units=2, number_of_dimensions=3),
            FeatureSignature(is_sparse=False, units=1, number_of_dimensions=3)
        ],
        config=model_config_basic,
    )
    inputs_raw_1 = [
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[1.5, 2.5], [3.5, 4.5], [0.0, 0.0]],
    ]
    inputs_raw_2 = [
        [[10.0], [20.0], [30.0]],
        [[40.0], [50.0], [0.0]],
    ]
    outputs_expected = [
        [[1.0, 2.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]],
        [[1.5, 2.5, 40.0], [3.5, 4.5, 50.0], [0.0, 0.0, 0.0]],
    ]
    inputs = ([tf.convert_to_tensor(inputs_raw_1, dtype=tf.float32), tf.convert_to_tensor(inputs_raw_2, dtype=tf.float32)],)
    outputs = layer(inputs)
    assert (outputs.numpy() == outputs_expected).all()
