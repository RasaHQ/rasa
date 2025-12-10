import numpy as np
import pytest
import scipy

from rasa.nlu.classifiers.diet_classifier import FeatureArray
from rasa.utils.tensorflow import TENSORFLOW_AVAILABLE

if not TENSORFLOW_AVAILABLE:
    pytest.skip("TensorFlow is not available", allow_module_level=True)

import tensorflow as tf

from rasa.utils.tensorflow.data_generator import (
    RasaBatchDataGenerator,
    tf_data_generator_from_rasa_data_generator,
)
from rasa.utils.tensorflow.model_data import RasaModelData


def test_tf_data_generator_signature_structure(model_data: RasaModelData):
    """Verifies that the output signature is correctly constructed as a flat tuple."""
    batch_size = 2
    data_generator = RasaBatchDataGenerator(
        model_data,
        batch_size=batch_size,
        epochs=1,
        batch_strategy="balanced",
        shuffle=False,
    )

    tf_dataset = tf_data_generator_from_rasa_data_generator(data_generator)

    # element_spec should be (inputs_tuple, targets_spec)
    assert isinstance(tf_dataset.element_spec, tuple)
    assert len(tf_dataset.element_spec) == 2

    inputs_spec, targets_spec = tf_dataset.element_spec

    # Inputs should be a flat tuple of TypeSpecs
    assert isinstance(inputs_spec, tuple)

    # Count expected specs based on model_data
    # model_data fixture has:
    # "text" -> "sentence" (dense, 3D) -> 1 spec
    # "text" -> "sentence" (sparse, 3D) -> 3 specs (indices, values, shape)
    # "action_text" -> "sequence" (dense, 4D) -> 1 spec
    # "action_text" -> "sequence" (sparse, 4D) -> 3 specs
    # "dialogue" -> "sentence" (dense, 3D) -> 1 spec
    # "label" -> "ids" (dense, 1D) -> 1 spec
    # "entities" -> "tag_ids" (dense, 3D) -> 1 spec
    # Total expected: 1 + 3 + 1 + 3 + 1 + 1 + 1 = 11 specs

    # Note: The order depends on dictionary iteration order, which is insertion ordered
    # in Python 3.7+
    # model_data fixture construction order:
    # text, action_text, dialogue, label, entities

    assert len(inputs_spec) == 11

    # Verify types
    for spec in inputs_spec:
        assert isinstance(spec, tf.TypeSpec)


def test_tf_data_generator_shapes_and_types(model_data: RasaModelData):
    """Verifies that the shapes in the signature allow for variable
    dimensions (None).
    """
    batch_size = 2
    data_generator = RasaBatchDataGenerator(
        model_data,
        batch_size=batch_size,
        epochs=1,
        batch_strategy="balanced",
        shuffle=False,
    )

    tf_dataset = tf_data_generator_from_rasa_data_generator(data_generator)
    inputs_spec, _ = tf_dataset.element_spec

    # Helper to find spec by checking properties (since order is implicit)
    # We know "label" -> "ids" is 1D dense.
    # It should have shape (None,) corresponding to batch size.

    found_1d_spec = False
    for spec in inputs_spec:
        if spec.dtype == tf.float32 and len(spec.shape) == 1:
            # This matches the 1D feature (label ids)
            assert spec.shape[0] is None
            found_1d_spec = True

    assert found_1d_spec, "Did not find expected 1D spec with shape (None,)"

    # Verify sparse decomposition specs
    # Sparse features should have 3 specs: (None, 3) int64, (None,) float32, (3,) int64
    found_sparse_indices = False
    found_sparse_values = False
    found_sparse_shape = False

    for spec in inputs_spec:
        if spec.dtype == tf.int64 and len(spec.shape) == 2 and spec.shape[1] == 3:
            assert spec.shape[0] is None
            found_sparse_indices = True
        if spec.dtype == tf.float32 and len(spec.shape) == 1:
            # This could overlap with 1D dense, but sparse values are also (None,)
            # We check if we found at least one
            found_sparse_values = True
        if spec.dtype == tf.int64 and len(spec.shape) == 1 and spec.shape[0] == 3:
            found_sparse_shape = True

    assert found_sparse_indices
    assert found_sparse_values
    assert found_sparse_shape


def test_tf_data_generator_iteration(model_data: RasaModelData):
    """Verifies that we can iterate over the dataset and get valid tensors."""
    batch_size = 2
    data_generator = RasaBatchDataGenerator(
        model_data,
        batch_size=batch_size,
        epochs=1,
        batch_strategy="balanced",
        shuffle=False,
    )

    tf_dataset = tf_data_generator_from_rasa_data_generator(data_generator)

    iterator = iter(tf_dataset)

    # Fetch first batch
    inputs, targets = next(iterator)

    assert isinstance(inputs, tuple)
    assert len(inputs) == 11  # Matches spec length

    # Check that everything is a Tensor
    for item in inputs:
        assert isinstance(item, tf.Tensor)

    # Check targets
    assert isinstance(targets, tf.Tensor)
    # Targets are always empty (0, 1) from RasaBatchDataGenerator
    assert targets.shape == (0, 1)


def test_tf_data_generator_shuffling():
    """Verifies that the dataset reshuffles data across epochs when shuffle=True."""
    # We use a simple 1D feature so we can easily track order
    features = np.array([[1], [2], [3], [4], [5], [6], [7]], dtype=np.float32)

    model_data = RasaModelData(
        data={"label": {"ids": [FeatureArray(features, number_of_dimensions=2)]}}
    )

    # Batch size 1 to make order tracking trivial
    data_generator = RasaBatchDataGenerator(
        model_data, batch_size=1, epochs=1, batch_strategy="balanced", shuffle=True
    )

    tf_dataset = tf_data_generator_from_rasa_data_generator(data_generator)

    epoch_1_data = []
    for inputs, _ in tf_dataset:
        val = inputs[0].numpy()[0][0]  # Batch 0, Index 0
        epoch_1_data.append(val)

    epoch_2_data = []
    for inputs, _ in tf_dataset:
        val = inputs[0].numpy()[0][0]
        epoch_2_data.append(val)

    # Verify content is the same (just reordered)
    assert (
        sorted(epoch_1_data)
        == sorted(epoch_2_data)
        == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    )
    assert epoch_1_data != epoch_2_data, "Data order should differ between epochs"


def test_tf_data_generator_shuffling_large_dataset():
    """Verifies that the dataset reshuffles data across epochs when shuffle=True."""
    features_large = np.arange(50, dtype=np.float32).reshape(-1, 1)
    model_data_large = RasaModelData(
        data={"label": {"ids": [FeatureArray(features_large, number_of_dimensions=2)]}}
    )

    data_generator_large = RasaBatchDataGenerator(
        model_data_large,
        batch_size=1,
        epochs=1,
        batch_strategy="balanced",
        shuffle=True,
    )

    tf_dataset_large = tf_data_generator_from_rasa_data_generator(data_generator_large)

    epoch_1_data = [inputs[0].numpy()[0][0] for inputs, _ in tf_dataset_large]
    epoch_2_data = [inputs[0].numpy()[0][0] for inputs, _ in tf_dataset_large]

    assert (
        epoch_1_data != epoch_2_data
    ), "Data order should change between epochs when shuffle=True"
    assert sorted(epoch_1_data) == sorted(epoch_2_data)


def test_tf_data_generator_no_shuffling():
    """Verifies that the dataset preserves order when shuffle=False."""
    features = np.array([[1], [2], [3], [4]], dtype=np.float32)

    model_data = RasaModelData(
        data={"label": {"ids": [FeatureArray(features, number_of_dimensions=2)]}}
    )

    data_generator = RasaBatchDataGenerator(
        model_data, batch_size=1, epochs=1, batch_strategy="balanced", shuffle=False
    )

    tf_dataset = tf_data_generator_from_rasa_data_generator(data_generator)

    e1 = [inputs[0].numpy()[0][0] for inputs, _ in tf_dataset]
    e2 = [inputs[0].numpy()[0][0] for inputs, _ in tf_dataset]

    expected = [1.0, 2.0, 3.0, 4.0]
    assert e1 == expected
    assert e2 == expected


def test_example_1_simple_dense_feature_1d():
    """Example 1: Simple Dense Feature (Labels) - 1D Array."""
    # Input: 2 examples, 1D feature
    # Feature "label" -> "ids": shape (2,)
    features = np.array([1, 0], dtype=np.float32)

    model_data = RasaModelData(
        data={"label": {"ids": [FeatureArray(features, number_of_dimensions=1)]}}
    )

    data_generator = RasaBatchDataGenerator(
        model_data, batch_size=2, epochs=1, batch_strategy="balanced", shuffle=False
    )

    tf_dataset = tf_data_generator_from_rasa_data_generator(data_generator)

    # Expected Signature:
    # Inputs: (TensorSpec(shape=(None,), dtype=float32),)
    # Targets: TensorSpec(shape=(None, 1), dtype=float32)

    inputs_spec, targets_spec = tf_dataset.element_spec

    assert len(inputs_spec) == 1
    assert inputs_spec[0].shape.as_list() == [None]

    assert targets_spec.shape.as_list() == [None, 1]

    # Verify data
    iterator = iter(tf_dataset)
    inputs, targets = next(iterator)

    # Inputs is a tuple of 1 tensor
    assert len(inputs) == 1
    assert np.array_equal(inputs[0].numpy(), features)


def test_example_2_sequential_dense_feature_3d():
    """Example 2: Sequential Dense Feature (Text) - 3D Array with variable
    sequence length.
    """
    # Input: 2 examples.
    # Batch 1: Max seq len 3. Shape (2, 3, 10)

    # Create a batch with padding
    # Ex 1: len 3
    # Ex 2: len 2 (padded to 3)
    feature_data = np.zeros((2, 3, 10), dtype=np.float32)
    feature_data[0, :3, :] = 0.1  # Fill 3 steps
    feature_data[1, :2, :] = 0.2  # Fill 2 steps

    model_data = RasaModelData(
        data={
            "text": {"sentence": [FeatureArray(feature_data, number_of_dimensions=3)]}
        }
    )

    data_generator = RasaBatchDataGenerator(
        model_data, batch_size=2, epochs=1, batch_strategy="balanced", shuffle=False
    )

    tf_dataset = tf_data_generator_from_rasa_data_generator(data_generator)

    # Expected Signature:
    # Inputs: (TensorSpec(shape=(None, None, 10), dtype=float32),)
    # Note: dim 1 is None (variable sequence length)

    inputs_spec, targets_spec = tf_dataset.element_spec

    assert len(inputs_spec) == 1
    # Check shape: (None, None, 10)
    assert inputs_spec[0].shape.as_list() == [None, None, 10]

    # Verify data
    iterator = iter(tf_dataset)
    inputs, targets = next(iterator)

    assert inputs[0].shape == (2, 3, 10)


def test_example_3_sparse_feature():
    """Example 3: Sparse Feature (Entities) - Decomposed into 3 tensors."""
    sp_ex1 = scipy.sparse.csr_matrix(
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float32
    )  # Shape (1, 10)
    sp_ex2 = scipy.sparse.csr_matrix(
        [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32
    )  # Shape (1, 10)

    # Input: 2 examples.
    # Sparse matrix of shape (2, 10)
    features = np.array([sp_ex1, sp_ex2])

    model_data = RasaModelData(
        data={"entities": {"tag_ids": [FeatureArray(features, number_of_dimensions=3)]}}
    )

    data_generator = RasaBatchDataGenerator(
        model_data, batch_size=2, epochs=1, batch_strategy="balanced", shuffle=False
    )

    tf_dataset = tf_data_generator_from_rasa_data_generator(data_generator)

    # Expected Signature:
    # Inputs: 3 Tensors (Indices, Values, Shape)
    # Indices: (None, 3) int64
    # Values: (None,) float32
    # Shape: (3,) int64

    inputs_spec, targets_spec = tf_dataset.element_spec

    assert len(inputs_spec) == 3

    # Indices
    assert inputs_spec[0].dtype == tf.int64
    assert inputs_spec[0].shape.as_list() == [None, 3]

    # Values
    assert inputs_spec[1].dtype == tf.float32
    assert inputs_spec[1].shape.as_list() == [None]

    # Shape
    assert inputs_spec[2].dtype == tf.int64
    assert inputs_spec[2].shape.as_list() == [3]

    # Verify data
    iterator = iter(tf_dataset)
    inputs, targets = next(iterator)

    indices, values, shape = inputs

    # Check values (should be 1.0)
    assert np.all(values.numpy() == 1.0)
    # Check shape (should be [2, 1, 10])
    assert np.array_equal(shape.numpy(), [2, 1, 10])


def test_example_4_mixed_input():
    """Example 4: Complex Mixed Input (Text + Entities + Label)."""
    # 1. Text (Dense 3D): (2, 3, 10)
    text_data = np.zeros((2, 3, 10), dtype=np.float32)

    # 2. Entities (Sparse 3D): (2, 1, 10)
    sp_ex1 = scipy.sparse.csr_matrix([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.float32)
    sp_ex2 = scipy.sparse.csr_matrix([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    entity_data = np.array([sp_ex1, sp_ex2])

    # 3. Label (Dense 1D): (2,)
    label_data = np.array([1, 0], dtype=np.float32)

    model_data = RasaModelData(
        data={
            "text": {"sentence": [FeatureArray(text_data, number_of_dimensions=3)]},
            "entities": {
                "tag_ids": [FeatureArray(entity_data, number_of_dimensions=3)]
            },
            "label": {"ids": [FeatureArray(label_data, number_of_dimensions=1)]},
        }
    )

    data_generator = RasaBatchDataGenerator(
        model_data, batch_size=2, epochs=1, batch_strategy="balanced", shuffle=False
    )

    tf_dataset = tf_data_generator_from_rasa_data_generator(data_generator)

    # Expected Signature:
    # Flattened order depends on insertion order in dict (Python 3.7+ preserves it)
    # 1. Text (1 tensor)
    # 2. Entities (3 tensors)
    # 3. Label (1 tensor)
    # Total: 5 tensors

    inputs_spec, targets_spec = tf_dataset.element_spec

    assert len(inputs_spec) == 5

    # Text
    assert inputs_spec[0].shape.as_list() == [None, None, 10]

    # Indices
    assert inputs_spec[1].dtype == tf.int64
    assert inputs_spec[1].shape.as_list() == [None, 3]
    # Values
    assert inputs_spec[2].dtype == tf.float32
    assert inputs_spec[2].shape.as_list() == [None]
    # Shape
    assert inputs_spec[3].dtype == tf.int64
    assert inputs_spec[3].shape.as_list() == [3]

    # Label
    assert inputs_spec[4].shape.as_list() == [None]

    # Verify data
    iterator = iter(tf_dataset)
    inputs, targets = next(iterator)

    assert len(inputs) == 5
