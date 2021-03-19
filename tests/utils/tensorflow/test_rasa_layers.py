import pytest
import tensorflow as tf

import numpy as np

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


attribute_name = TEXT
units_1 = 2
units_2 = 3
units_sparse_to_dense = 10
units_concat = 7
units_hidden_layer = 11
units_transformer = 14
num_transformer_heads = 2
num_transformer_layers = 2
batch_size = 5
max_seq_length = 3


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


# Dummy feature signatures and features (full of 1s) for tests that don't check exact
# numerical outputs, only shapes

feature_signature_sparse_1 = FeatureSignature(
    is_sparse=True, units=units_1, number_of_dimensions=3
)
feature_sparse_seq_1 = tf.sparse.from_dense(
    tf.ones((batch_size, max_seq_length, units_1))
)
feature_sparse_sent_1 = tf.sparse.from_dense(tf.ones((batch_size, 1, units_1)))

feature_signature_dense_1 = FeatureSignature(
    is_sparse=False, units=units_1, number_of_dimensions=3
)
feature_dense_seq_1 = tf.ones((batch_size, max_seq_length, units_1))
feature_dense_sent_1 = tf.ones((batch_size, 1, units_1))

feature_signature_dense_2 = FeatureSignature(
    is_sparse=False, units=units_2, number_of_dimensions=3
)
feature_dense_seq_2 = tf.ones((batch_size, max_seq_length, units_2))
feature_dense_sent_2 = tf.ones((batch_size, 1, units_2))

sequence_lengths = tf.ones((batch_size,)) * max_seq_length
sequence_lengths_empty = tf.ones((batch_size,)) * 0

attribute_signature_basic = {
    SEQUENCE: [feature_signature_dense_1, feature_signature_sparse_1],
    SENTENCE: [feature_signature_dense_1],
}
attribute_features_basic = (
    [feature_dense_seq_1, feature_sparse_seq_1],
    [feature_dense_sent_1],
    sequence_lengths,
)


@pytest.mark.parametrize(
    "layer_class, model_config, layer_args, expected_output_units",
    [
        # ConcatenateSparseDense layer with mixed features
        (
            ConcatenateSparseDenseFeatures,
            model_config_basic,
            {
                "feature_type": "arbitrary",
                "feature_type_signature": [
                    feature_signature_sparse_1,
                    feature_signature_sparse_1,
                    feature_signature_dense_1,
                    feature_signature_dense_2,
                ],
            },
            2 * units_sparse_to_dense + units_1 + units_2,
        ),
        # ConcatenateSparseDense layer with only sparse features
        (
            ConcatenateSparseDenseFeatures,
            model_config_basic,
            {
                "feature_type": "arbitrary",
                "feature_type_signature": [feature_signature_sparse_1],
            },
            units_sparse_to_dense,
        ),
        # ConcatenateSparseDense layer with only dense features
        (
            ConcatenateSparseDenseFeatures,
            model_config_basic,
            {
                "feature_type": "arbitrary",
                "feature_type_signature": [feature_signature_dense_1],
            },
            units_1,
        ),
        # FeatureCombining layer with sequence- and sentence-level features, dimension unifying
        (
            RasaFeatureCombiningLayer,
            model_config_basic,
            {"attribute_signature": attribute_signature_basic},
            units_concat,
        ),
        # FeatureCombining layer with sequence- and sentence-level features, no dimension unifying
        (
            RasaFeatureCombiningLayer,
            model_config_basic,
            {
                "attribute_signature": {
                    SEQUENCE: [feature_signature_dense_1],
                    SENTENCE: [feature_signature_dense_1],
                }
            },
            units_1,
        ),
        # FeatureCombining layer with sentence-level features only
        (
            RasaFeatureCombiningLayer,
            model_config_basic,
            {
                "attribute_signature": {
                    "sequence": [],
                    "sentence": [feature_signature_dense_1],
                },
            },
            units_1,
        ),
        # FeatureCombining layer with sequence-level features only
        (
            RasaFeatureCombiningLayer,
            model_config_basic,
            {
                "attribute_signature": {
                    "sequence": [feature_signature_dense_1],
                    "sentence": [],
                },
            },
            units_1,
        ),
        # Sequence layer with mixed features, hidden layers and transformer
        (
            RasaSequenceLayer,
            model_config_transformer,
            {"attribute_signature": attribute_signature_basic},
            units_transformer,
        ),
        # Sequence layer with mixed features, hidden layers, no transformer
        (
            RasaSequenceLayer,
            model_config_basic,
            {"attribute_signature": attribute_signature_basic},
            units_hidden_layer,
        ),
        # Sequence layer with mixed features, no hidden layers, no transformer
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
    "layer_class, model_config, layer_args, layer_inputs, expected_output_shapes_train, expected_output_shapes_test",
    [
        # ConcatenateSparseDense layer with mixed features
        (
            ConcatenateSparseDenseFeatures,
            model_config_basic,
            {
                "feature_type": "arbitrary",
                "feature_type_signature": [
                    feature_signature_sparse_1,
                    feature_signature_sparse_1,
                    feature_signature_dense_1,
                    feature_signature_dense_2,
                ],
            },
            (
                [
                    feature_sparse_seq_1,
                    feature_sparse_seq_1,
                    feature_dense_seq_1,
                    feature_dense_seq_2,
                ],
            ),
            (
                batch_size,
                max_seq_length,
                2 * units_sparse_to_dense + units_1 + units_2,
            ),
            "same_as_train",  # means that test-time shapes are same as train-time ones
        ),
        # ConcatenateSparseDense layer with only sparse features
        (
            ConcatenateSparseDenseFeatures,
            model_config_basic,
            {
                "feature_type": "arbitrary",
                "feature_type_signature": [feature_signature_sparse_1],
            },
            ([feature_sparse_sent_1],),
            (batch_size, 1, units_sparse_to_dense),
            "same_as_train",
        ),
        # ConcatenateSparseDense layer with only dense features
        (
            ConcatenateSparseDenseFeatures,
            model_config_basic,
            {
                "feature_type": "arbitrary",
                "feature_type_signature": [feature_signature_dense_1],
            },
            ([feature_dense_sent_1],),
            (batch_size, 1, units_1),
            "same_as_train",
        ),
        # FeatureCombining layer with sequence- and sentence-level features, dimension unifying
        (
            RasaFeatureCombiningLayer,
            model_config_basic,
            {"attribute_signature": attribute_signature_basic},
            attribute_features_basic,
            [
                (batch_size, max_seq_length + 1, units_concat),
                (batch_size, max_seq_length + 1, 1),
            ],
            "same_as_train",
        ),
        # FeatureCombining layer with sequence- and sentence-level features, no dimension unifying
        (
            RasaFeatureCombiningLayer,
            model_config_basic,
            {
                "attribute_signature": {
                    SEQUENCE: [feature_signature_dense_1],
                    SENTENCE: [feature_signature_dense_1],
                }
            },
            ([feature_dense_seq_1], [feature_dense_sent_1], sequence_lengths,),
            [
                (batch_size, max_seq_length + 1, units_1),
                (batch_size, max_seq_length + 1, 1),
            ],
            "same_as_train",
        ),
        # FeatureCombining layer with sentence-level features only
        (
            RasaFeatureCombiningLayer,
            model_config_basic,
            {
                "attribute_signature": {
                    "sequence": [],
                    "sentence": [feature_signature_dense_1],
                },
            },
            ([], [feature_dense_sent_1], sequence_lengths_empty),
            [(batch_size, 1, units_1), (batch_size, 1, 1)],
            "same_as_train",
        ),
        # FeatureCombining layer with sequence-level features only
        (
            RasaFeatureCombiningLayer,
            model_config_basic,
            {
                "attribute_signature": {
                    "sequence": [feature_signature_dense_1],
                    "sentence": [],
                },
            },
            ([feature_dense_seq_1], [], sequence_lengths),
            [(batch_size, max_seq_length, units_1), (batch_size, max_seq_length, 1),],
            "same_as_train",
        ),
        # Sequence layer with mixed features, hidden layers and transformer, doing MLM
        (
            RasaSequenceLayer,
            model_config_transformer_mlm,
            {"attribute_signature": attribute_signature_basic},
            attribute_features_basic,
            [
                (batch_size, max_seq_length + 1, units_transformer),
                (batch_size, max_seq_length + 1, units_hidden_layer),
                (batch_size, max_seq_length + 1, 1),
                (batch_size, max_seq_length + 1, units_1),
                (batch_size, max_seq_length + 1, 1),
                (
                    num_transformer_layers,
                    batch_size,
                    num_transformer_heads,
                    max_seq_length + 1,
                    max_seq_length + 1,
                ),
            ],
            [
                (batch_size, max_seq_length + 1, units_transformer),
                (batch_size, max_seq_length + 1, units_hidden_layer),
                (batch_size, max_seq_length + 1, 1),
                (0,),
                (0,),
                (
                    num_transformer_layers,
                    batch_size,
                    num_transformer_heads,
                    max_seq_length + 1,
                    max_seq_length + 1,
                ),
            ],
        ),
        # Sequence layer with mixed features, hidden layers, no transformer, no MLM
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
            "same_as_train",
        ),
        # Sequence layer with mixed features, no hidden layers, no transformer, no MLM
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
            "same_as_train",
        ),
        # Sequence layer with only sequence-level sparse features & MLM (to check token_ids shape)
        (
            RasaSequenceLayer,
            model_config_transformer_mlm,
            {
                "attribute_signature": {
                    SEQUENCE: [feature_signature_sparse_1],
                    SENTENCE: [],
                }
            },
            ([feature_sparse_seq_1], [], sequence_lengths,),
            [
                (batch_size, max_seq_length, units_transformer),
                (batch_size, max_seq_length, units_hidden_layer),
                (batch_size, max_seq_length, 1),
                (batch_size, max_seq_length, 2),
                (batch_size, max_seq_length, 1),
                (
                    num_transformer_layers,
                    batch_size,
                    num_transformer_heads,
                    max_seq_length,
                    max_seq_length,
                ),
            ],
            [
                (batch_size, max_seq_length, units_transformer),
                (batch_size, max_seq_length, units_hidden_layer),
                (batch_size, max_seq_length, 1),
                (0,),
                (0,),
                (
                    num_transformer_layers,
                    batch_size,
                    num_transformer_heads,
                    max_seq_length,
                    max_seq_length,
                ),
            ],
        ),
    ],
)
def test_correct_output_shape(
    layer_class,
    model_config,
    layer_args,
    layer_inputs,
    expected_output_shapes_train,
    expected_output_shapes_test,
):
    layer = layer_class(**layer_args, attribute=attribute_name, config=model_config,)

    train_outputs = layer(layer_inputs, training=True)
    if isinstance(expected_output_shapes_train, list):
        for i, expected_shape in enumerate(expected_output_shapes_train):
            assert train_outputs[i].shape == expected_shape
    else:
        assert train_outputs.shape == expected_output_shapes_train

    if expected_output_shapes_test == "same_as_train":
        expected_output_shapes_test = expected_output_shapes_train
    test_outputs = layer(layer_inputs, training=False)
    if isinstance(expected_output_shapes_test, list):
        for i, expected_shape in enumerate(expected_output_shapes_test):
            assert test_outputs[i].shape == expected_shape
    else:
        assert test_outputs.shape == expected_output_shapes_test


@pytest.mark.parametrize(
    "layer_class, layer_args",
    [
        # ConcatenateSparseDense layer breaks on empty feature type signature
        (
            ConcatenateSparseDenseFeatures,
            {"feature_type": "arbitrary", "feature_type_signature": []},
        ),
        # FeatureCombining layer breaks on empty attribute signature
        (
            RasaFeatureCombiningLayer,
            {"attribute_signature": {"sequence": [], "sentence": []}},
        ),
        # Sequence layer breaks on no sequence-level features
        (
            RasaSequenceLayer,
            {
                "attribute_signature": {
                    "sequence": [],
                    "sentence": [feature_dense_sent_1],
                }
            },
        ),
    ],
)
def test_raises_exception_when_missing_features(layer_class, layer_args):
    with pytest.raises(TFLayerConfigException):
        layer_class(**layer_args, attribute=attribute_name, config=model_config_basic)


def test_concat_sparse_dense_raises_exception_when_inconsistent_sparse_features():
    with pytest.raises(TFLayerConfigException):
        ConcatenateSparseDenseFeatures(
            attribute=attribute_name,
            feature_type=SEQUENCE,
            feature_type_signature=[
                FeatureSignature(is_sparse=True, units=2, number_of_dimensions=3),
                FeatureSignature(is_sparse=True, units=1, number_of_dimensions=3),
            ],
            config=model_config_basic,
        )


# Realistic feature signatures and features for checking exact outputs

realistic_feature_signature_dense_1 = FeatureSignature(
    is_sparse=False, units=1, number_of_dimensions=3
)
realistic_feature_dense_seq_1 = tf.convert_to_tensor(
    [[[10.0], [20.0], [30.0]], [[40.0], [50.0], [0.0]],], dtype=tf.float32
)

realistic_feature_signature_dense_2 = FeatureSignature(
    is_sparse=False, units=2, number_of_dimensions=3
)
realistic_feature_dense_seq_2 = tf.convert_to_tensor(
    [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.5, 2.5], [3.5, 4.5], [0.0, 0.0]],],
    dtype=tf.float32,
)

realistic_feature_signature_dense_3 = FeatureSignature(
    is_sparse=False, units=3, number_of_dimensions=3
)
realistic_feature_dense_sent_3 = tf.convert_to_tensor(
    [[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]], dtype=tf.float32,
)

realistic_sequence_lengths = tf.convert_to_tensor([3, 2], dtype=tf.int32)
realistic_sequence_lengths_empty = tf.convert_to_tensor([0, 0], dtype=tf.int32)


def test_concat_sparse_dense_correct_output_for_dense_input():
    layer = ConcatenateSparseDenseFeatures(
        attribute=attribute_name,
        feature_type=SEQUENCE,
        feature_type_signature=[
            realistic_feature_signature_dense_1,
            realistic_feature_signature_dense_2,
        ],
        config=dict(
            model_config_basic,
            # also activate all dropout to check that it has no effect on dense features
            **{SPARSE_INPUT_DROPOUT: True, DENSE_INPUT_DROPOUT: True},
        ),
    )
    outputs_expected = [
        [[10.0, 1.0, 2.0], [20.0, 3.0, 4.0], [30.0, 5.0, 6.0]],
        [[40.0, 1.5, 2.5], [50.0, 3.5, 4.5], [0.0, 0.0, 0.0]],
    ]
    inputs = ([realistic_feature_dense_seq_1, realistic_feature_dense_seq_2],)
    train_outputs = layer(inputs, training=True)
    assert (train_outputs.numpy() == outputs_expected).all()
    test_outputs = layer(inputs, training=False)
    assert (test_outputs.numpy() == outputs_expected).all()


def test_concat_sparse_dense_applies_dropout_to_sparse_input():
    layer_dropout_for_sparse = ConcatenateSparseDenseFeatures(
        attribute=attribute_name,
        feature_type=SEQUENCE,
        feature_type_signature=[feature_signature_sparse_1, feature_signature_sparse_1],
        config=dict(model_config_basic, **{SPARSE_INPUT_DROPOUT: True, DROP_RATE: 1.0}),
    )

    inputs = ([feature_sparse_seq_1, feature_sparse_seq_1],)
    expected_outputs_train = tf.zeros(
        (batch_size, max_seq_length, units_sparse_to_dense * 2)
    )

    train_outputs = layer_dropout_for_sparse(inputs, training=True)
    assert np.allclose(train_outputs.numpy(), expected_outputs_train.numpy())

    # We can't check exact output contents for sparse inputs but during test-time no
    # dropout should be applied, hence the outputs should not be all zeros in this case
    # (unlike at training time).
    test_outputs = layer_dropout_for_sparse(inputs, training=False)
    assert not np.allclose(test_outputs.numpy(), expected_outputs_train.numpy())


def test_concat_sparse_dense_applies_dropout_to_sparse_densified_input():
    layer_dropout_for_sparse_densified = ConcatenateSparseDenseFeatures(
        attribute=attribute_name,
        feature_type=SEQUENCE,
        feature_type_signature=[feature_signature_sparse_1, feature_signature_sparse_1],
        config=dict(
            model_config_basic, **{DENSE_INPUT_DROPOUT: True, DROP_RATE: 0.99999999}
        ),  # keras dropout doesn't accept velues >= 1.0
    )

    inputs = ([feature_sparse_seq_1, feature_sparse_seq_1],)
    expected_outputs_train = tf.zeros(
        (batch_size, max_seq_length, units_sparse_to_dense * 2)
    )

    train_outputs = layer_dropout_for_sparse_densified(inputs, training=True)
    assert np.allclose(train_outputs.numpy(), expected_outputs_train.numpy())

    # We can't check exact output contents for sparse inputs but during test-time no
    # dropout should be applied, hence the outputs should not be all zeros in this case
    # (unlike at training time).
    test_outputs = layer_dropout_for_sparse_densified(inputs, training=False)
    assert not np.allclose(test_outputs.numpy(), expected_outputs_train.numpy())


@pytest.mark.parametrize(
    "attribute_signature, inputs, expected_outputs_train, expected_outputs_test",
    [
        # Both sequence- and sentence-level features, not unifying dimensions before concat
        (
            {
                SEQUENCE: [
                    realistic_feature_signature_dense_1,
                    realistic_feature_signature_dense_2,
                ],
                SENTENCE: [realistic_feature_signature_dense_3],
            },
            (
                [realistic_feature_dense_seq_1, realistic_feature_dense_seq_2],
                [realistic_feature_dense_sent_3],
                realistic_sequence_lengths,
            ),
            (
                np.array(
                    [
                        [
                            [10.0, 1.0, 2.0],
                            [20.0, 3.0, 4.0],
                            [30.0, 5.0, 6.0],
                            [0.1, 0.2, 0.3],
                        ],
                        [
                            [40.0, 1.5, 2.5],
                            [50.0, 3.5, 4.5],
                            [0.4, 0.5, 0.6],
                            [0.0, 0.0, 0.0],
                        ],
                    ],
                    dtype=np.float32,
                ),
                [[[1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [0.0]]],
            ),
            "same_as_train",
        ),
        # Sequence-level features only
        (
            {
                SEQUENCE: [
                    realistic_feature_signature_dense_1,
                    realistic_feature_signature_dense_2,
                ],
                SENTENCE: [],
            },
            (
                [realistic_feature_dense_seq_1, realistic_feature_dense_seq_2],
                [],
                realistic_sequence_lengths,
            ),
            (
                np.array(
                    [
                        [[10.0, 1.0, 2.0], [20.0, 3.0, 4.0], [30.0, 5.0, 6.0]],
                        [[40.0, 1.5, 2.5], [50.0, 3.5, 4.5], [0.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
                [[[1.0], [1.0], [1.0]], [[1.0], [1.0], [0.0]]],
            ),
            "same_as_train",
        ),
        # Sentence-level features only
        (
            {SEQUENCE: [], SENTENCE: [realistic_feature_signature_dense_3]},
            ([], [realistic_feature_dense_sent_3], realistic_sequence_lengths_empty),
            (realistic_feature_dense_sent_3.numpy(), [[[1.0]], [[1.0]]]),
            "same_as_train",
        ),
    ],
)
def test_feature_combining_correct_output(
    attribute_signature, inputs, expected_outputs_train, expected_outputs_test
):
    layer = RasaFeatureCombiningLayer(
        attribute=attribute_name,
        config=model_config_basic,
        attribute_signature=attribute_signature,
    )
    if expected_outputs_test == "same_as_train":
        expected_outputs_test = expected_outputs_train

    train_outputs, train_mask_seq_sent = layer(inputs, training=True)
    assert (train_outputs.numpy() == expected_outputs_train[0]).all()
    assert (train_mask_seq_sent.numpy() == expected_outputs_train[1]).all()

    test_outputs, test_mask_seq_sent = layer(inputs, training=False)
    assert (test_outputs.numpy() == expected_outputs_test[0]).all()
    assert (test_mask_seq_sent.numpy() == expected_outputs_test[1]).all()


@pytest.mark.parametrize(
    "attribute_signature, inputs, expected_outputs_train",
    [
        # Both sequence- and sentence-level features
        (
            {
                SEQUENCE: [
                    realistic_feature_signature_dense_1,
                    realistic_feature_signature_dense_2,
                ],
                SENTENCE: [realistic_feature_signature_dense_3,],
            },
            (
                [realistic_feature_dense_seq_1, realistic_feature_dense_seq_2],
                [realistic_feature_dense_sent_3],
                realistic_sequence_lengths,
            ),
            (
                np.array(
                    [
                        [
                            [10.0, 1.0, 2.0],
                            [20.0, 3.0, 4.0],
                            [30.0, 5.0, 6.0],
                            [0.1, 0.2, 0.3],
                        ],
                        [
                            [40.0, 1.5, 2.5],
                            [50.0, 3.5, 4.5],
                            [0.4, 0.5, 0.6],
                            [0.0, 0.0, 0.0],
                        ],
                    ],
                    dtype=np.float32,
                ),
                [[[1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [0.0]]],
                np.concatenate(
                    (realistic_feature_dense_seq_1, [[[0.0]], [[0.0]]]), axis=1
                ),
            ),
        ),
        # Only sequence-level features
        (
            {
                SEQUENCE: [
                    realistic_feature_signature_dense_1,
                    realistic_feature_signature_dense_2,
                ],
                SENTENCE: [],
            },
            (
                [realistic_feature_dense_seq_1, realistic_feature_dense_seq_2],
                [],
                realistic_sequence_lengths,
            ),
            (
                np.array(
                    [
                        [[10.0, 1.0, 2.0], [20.0, 3.0, 4.0], [30.0, 5.0, 6.0],],
                        [[40.0, 1.5, 2.5], [50.0, 3.5, 4.5], [0.0, 0.0, 0.0],],
                    ],
                    dtype=np.float32,
                ),
                [[[1.0], [1.0], [1.0]], [[1.0], [1.0], [0.0]]],
                realistic_feature_dense_seq_1.numpy(),
            ),
        ),
    ],
)
def test_sequence_layer_correct_output(
    attribute_signature, inputs, expected_outputs_train
):
    layer = RasaSequenceLayer(
        attribute=attribute_name,
        # Use MLM but no transformer and no hidden layers.
        config=dict(model_config_basic_no_hidden_layers, **{MASKED_LM: True}),
        attribute_signature=attribute_signature,
    )

    # Training-time check
    (
        seq_sent_features_expected,
        mask_seq_sent_expected,
        token_ids_expected,
    ) = expected_outputs_train
    (_, seq_sent_features, mask_seq_sent, token_ids, mlm_boolean_mask, _,) = layer(
        inputs, training=True
    )
    assert (seq_sent_features.numpy() == seq_sent_features_expected).all()
    assert (mask_seq_sent.numpy() == mask_seq_sent_expected).all()
    assert (token_ids.numpy() == token_ids_expected).all()
    assert mlm_boolean_mask.dtype == bool
    # no masking at the padded position found in the shorter sequence
    assert not mlm_boolean_mask[-1][-1][0]
    # when sentence-level features are present, also ensure that no masking is done at
    # sentence-level feature positions (determined by sequence lengths)
    if len(attribute_signature[SENTENCE]) > 0:
        assert not mlm_boolean_mask.numpy()[0][realistic_sequence_lengths.numpy()][0]

    # Test-time check
    (seq_sent_features_expected, mask_seq_sent_expected, _,) = expected_outputs_train
    (
        transformer_outputs,
        seq_sent_features,
        mask_seq_sent,
        token_ids,
        mlm_boolean_mask,
        _,
    ) = layer(inputs, training=False)
    # Check that transformer outputs match the combined features, i.e. that MLM wasn't
    # applied
    assert (transformer_outputs.numpy() == seq_sent_features_expected).all()
    assert (seq_sent_features.numpy() == seq_sent_features_expected).all()
    assert (mask_seq_sent.numpy() == mask_seq_sent_expected).all()
    assert token_ids.numpy().size == 0
    assert mlm_boolean_mask.numpy().size == 0
