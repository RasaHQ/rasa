from typing import Text, List, Tuple, Union, Optional
import pytest
from _pytest.monkeypatch import MonkeyPatch
import numpy as np
import tensorflow as tf

from rasa.utils.tensorflow.layers import (
    DotProductLoss,
    MultiLabelDotProductLoss,
    RandomlyConnectedDense,
    DenseForSparse,
)
from rasa.utils.tensorflow.constants import INNER, SOFTMAX, LABEL, LABEL_PAD_ID
import rasa.utils.tensorflow.layers_utils as layers_utils
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    ACTION_NAME,
    ACTION_TEXT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)
from rasa.core.constants import DIALOGUE


def test_dot_product_loss_inner_sim():
    layer = DotProductLoss(0, similarity_type=INNER)
    a = tf.constant([[[1.0, 0.0, 2.0]], [[1.0, 0.0, 2.0]]])
    b = tf.constant([[[1.0, 0.0, -2.0]], [[1.0, 0.0, -2.0]]])
    mask = tf.constant([[1.0, 0.0]])
    similarity = layer.sim(a, b, mask=mask).numpy()
    assert np.all(similarity == [[[-3.0], [0.0]]])


def test_multi_label_dot_product_loss_call_shapes():
    num_neg = 1
    layer = MultiLabelDotProductLoss(num_neg)
    batch_inputs_embed = tf.constant([[[0, 1, 2]], [[-2, 0, 2]]], dtype=tf.float32)
    batch_labels_embed = tf.constant(
        [[[0, 0, 1], [1, 0, 0]], [[0, 1, 0], [1, 0, 0]]], dtype=tf.float32
    )
    batch_labels_ids = tf.constant([[[2], [0]], [[1], [0]]], dtype=tf.float32)
    all_labels_embed = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
    all_labels_ids = tf.constant([[0], [1], [2]], dtype=tf.float32)
    mask = None

    loss, accuracy = layer(
        batch_inputs_embed,
        batch_labels_embed,
        batch_labels_ids,
        all_labels_embed,
        all_labels_ids,
        mask,
    )

    assert len(tf.shape(loss)) == 0
    assert len(tf.shape(accuracy)) == 0


@pytest.mark.parametrize(
    "label_ids, num_candidates, expected_pos_label_mask",
    [
        ([[2, LABEL_PAD_ID], [3, 4]], 20, [[1, 0], [1, 1]]),
        ([[2, 1], [3, 4]], 5, [[1, 1], [1, 1]]),
    ],
)
def test_multi_label_dot_product_loss__construct_label_padding_mask(
    label_ids: List[List[int]],
    num_candidates: int,
    expected_pos_label_mask: List[List[int]],
):
    actual_label_mask = MultiLabelDotProductLoss._construct_mask_for_label_padding(
        np.expand_dims(label_ids, -1), num_candidates
    ).numpy()

    pos_label_columns = np.array(label_ids).shape[1]

    # First check if the mask corresponding to guaranteed positive label ids is correct.
    assert np.all(
        actual_label_mask[:, :pos_label_columns]
        == np.array(expected_pos_label_mask).astype(np.float32)
    )

    # Next check if the mask corresponding to sampled candidates is correct.
    assert np.all(
        actual_label_mask[:, pos_label_columns:]
        == np.ones((len(label_ids), num_candidates), dtype=np.float32)
    )


@pytest.mark.parametrize(
    "sim_pos, sim_candidates_il, pos_neg_labels, mask, expected_loss",
    [
        (
            np.array([[2.0, -0.1, -5], [4.2, 5.1, -4.5]]),
            np.array([[-1.1, -3], [2.1, -3.5]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            None,
            1.1991243,
        ),
        (
            np.array([[2.0, -0.1, -5], [4.2, 5.1, -4.5]]),
            np.array([[-1.1, -3], [2.1, -3.5]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array(
                [[1.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float32
            ),
            1.5972487,
        ),
    ],
)
def test_multi_label_dot_product_loss__compute_loss_with_and_without_mask(
    sim_pos: np.ndarray,
    sim_candidates_il: np.ndarray,
    pos_neg_labels: np.ndarray,
    mask: Optional[np.ndarray],
    expected_loss: float,
):
    layer = MultiLabelDotProductLoss(num_candidates=3)

    loss = layer._loss_sigmoid(
        np.expand_dims(sim_pos, 1).astype(np.float32),
        np.expand_dims(sim_candidates_il, 1).astype(np.float32),
        pos_neg_labels.astype(np.float32),
        mask,
    ).numpy()

    assert np.isclose([loss], [expected_loss])


@pytest.mark.parametrize(
    "sim_pos, sim_candidates_il, pos_neg_labels, mask, expected_accuracy",
    [
        (
            np.array([[2.0, -0.1, -5], [4.2, 5.1, -4.5]]),
            np.array([[-1.1, -3], [2.1, -3.5]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            None,
            0.6,
        ),
        (
            np.array([[2.0, -0.1, -5], [4.2, 5.1, -4.5]]),
            np.array([[-1.1, -3], [2.1, -3.5]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            np.array(
                [[1.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 1.0]], dtype=np.float32
            ),
            0.5833334,
        ),
    ],
)
def test_multi_label_dot_product_loss__compute_accuracy_with_and_without_mask(
    sim_pos: np.ndarray,
    sim_candidates_il: np.ndarray,
    pos_neg_labels: np.ndarray,
    mask: Optional[np.ndarray],
    expected_accuracy: float,
):
    layer = MultiLabelDotProductLoss(num_candidates=3)

    accuracy = layer._accuracy(
        np.expand_dims(sim_pos, 1).astype(np.float32),
        np.expand_dims(sim_candidates_il, 1).astype(np.float32),
        pos_neg_labels.astype(np.float32),
        mask,
    ).numpy()

    assert np.isclose([accuracy], [expected_accuracy])


def test_multi_label_dot_product_loss__sample_candidates_with_constant_number_of_labels(
    monkeypatch: MonkeyPatch,
):
    num_candidates = 2
    num_features = 4
    batch_size = 3
    layer = MultiLabelDotProductLoss(
        num_candidates, scale_loss=False, similarity_type=INNER
    )

    # Seven random vectors for inputs and labels
    i0, i1, i2, l0, l1, l2, l3 = np.round(
        np.random.uniform(-100, 100, size=[7, num_features])
    ).tolist()

    # Each example in the batch has one input
    batch_inputs_embed = tf.constant([[i0], [i1], [i2]], dtype=tf.float32)
    # Each input can have multiple labels (here its always the same number of labels,
    # but it doesn't have to be)
    batch_labels_embed = tf.constant([[l0, l1], [l2, l3], [l3, l0]], dtype=tf.float32)
    # We assign the corresponding indices
    batch_labels_ids = tf.constant(
        [[[0], [1]], [[2], [3]], [[3], [0]]], dtype=tf.float32
    )
    # List all the labels and ids in play
    all_labels_embed = tf.constant([l0, l1, l2, l3], dtype=tf.float32)
    all_labels_ids = tf.constant([[0], [1], [2], [3]], dtype=tf.float32)

    # Inside `layer._sample_candidates` random indices will be generated for the
    # candidates. We mock them to have a deterministic output.
    mock_indices = [0, 2, 0, 1, 0, 3]

    def mock_random_indices(*args, **kwargs) -> tf.Tensor:
        return tf.reshape(tf.constant(mock_indices), [batch_size, num_candidates])

    monkeypatch.setattr(layers_utils, "random_indices", mock_random_indices)

    # Now run the function we want to test
    (
        pos_inputs_embed,
        pos_labels_embed,
        candidate_labels_embed,
        pos_neg_labels,
    ) = layer._sample_candidates(
        batch_inputs_embed,
        batch_labels_embed,
        batch_labels_ids,
        all_labels_embed,
        all_labels_ids,
    )
    # The inputs just stay the inputs, up to an extra dimension
    assert np.all(
        pos_inputs_embed.numpy() == tf.expand_dims(batch_inputs_embed, axis=-2).numpy()
    )
    # All positive labels of each batch are in `pos_labels_embed`
    assert np.all(
        pos_labels_embed.numpy() == np.array([[[l0, l1]], [[l2, l3]], [[l3, l0]]])
    )
    # The candidate label embeddings are picked according to the `mock_indices` above.
    # E.g. a 2 coming from `mock_indices` means that `all_labels_embed[2]` is picked,
    # i.e. `l2`.
    assert np.all(
        candidate_labels_embed.numpy() == np.array([[[l0, l2]], [[l0, l1]], [[l0, l3]]])
    )
    # The `pos_neg_labels` contains `1`s wherever the vector in `candidate_labels_embed`
    # of example `i` is actually in the possible lables of example `i`
    assert np.all(
        pos_neg_labels.numpy()
        == np.array(
            [
                [1, 0],  # l0 is an actual positive example in `batch_labels_embed[0]`,
                # whereas l2 is not
                [
                    0,
                    0,
                ],  # Neither l0 nor l3 are positive examples in `batch_labels_embed[1]`
                [
                    1,
                    1,
                ],  # l0 and l3 are both positive examples in `batch_labels_embed[2]`
            ]
        )
    )


def test_multi_label_dot_product_loss__sample_candidates_with_variable_number_of_labels(
    monkeypatch: MonkeyPatch,
):
    num_candidates = 2
    num_features = 4
    batch_size = 3
    layer = MultiLabelDotProductLoss(num_candidates)

    # Seven random vectors for inputs and labels
    i0, i1, i2, l0, l1, l2, l3 = np.round(
        np.random.uniform(-100, 100, size=[7, num_features])
    ).tolist()
    # Label used for padding
    lp = [-1] * num_features

    # Each example in the batch has one input
    batch_inputs_embed = tf.constant([[i0], [i1], [i2]], dtype=tf.float32)
    # Each input can have multiple labels (lp serves as a placeholder)
    batch_labels_embed = tf.constant(
        [[l0, l1, l3], [l2, lp, lp], [l3, l0, lp]], dtype=tf.float32
    )
    # We assign the corresponding indices
    batch_labels_ids = tf.constant(
        [[[0], [1], [3]], [[2], [-1], [-1]], [[3], [0], [-1]]], dtype=tf.float32
    )
    # List all the labels and ids in play
    all_labels_embed = tf.constant([l0, l1, l2, l3], dtype=tf.float32)
    all_labels_ids = tf.constant([[0], [1], [2], [3]], dtype=tf.float32)

    # Inside `layer._sample_candidates` random indices will be generated for the
    # candidates. We mock them to have a deterministic output.
    mock_indices = [0, 2, 0, 1, 3, 1]

    def mock_random_indices(*args, **kwargs) -> tf.Tensor:
        return tf.reshape(tf.constant(mock_indices), [batch_size, num_candidates])

    monkeypatch.setattr(layers_utils, "random_indices", mock_random_indices)

    # Now run the function we want to test
    (
        pos_inputs_embed,
        pos_labels_embed,
        candidate_labels_embed,
        pos_neg_labels,
    ) = layer._sample_candidates(
        batch_inputs_embed,
        batch_labels_embed,
        batch_labels_ids,
        all_labels_embed,
        all_labels_ids,
    )
    # The inputs just stay the inputs, up to an extra dimension
    assert np.all(
        pos_inputs_embed.numpy() == tf.expand_dims(batch_inputs_embed, axis=-2).numpy()
    )
    # All example labels of each batch are in `pos_labels_embed`
    assert np.all(
        pos_labels_embed.numpy()
        == np.array([[[l0, l1, l3]], [[l2, lp, lp]], [[l3, l0, lp]]])
    )
    # The candidate label embeddings are picked according to the `mock_indices` above.
    # E.g. a 2 coming from `mock_indices` means that `all_labels_embed[2]` is picked,
    # i.e. `l2`.
    assert np.all(
        candidate_labels_embed.numpy() == np.array([[[l0, l2]], [[l0, l1]], [[l3, l1]]])
    )
    # The `pos_neg_labels` contains `1`s wherever the vector in `candidate_labels_embed`
    # of example `i` is actually in the possible lables of example `i`
    assert np.all(
        pos_neg_labels.numpy()
        == np.array(
            [
                [1, 0],  # l0 is an actual positive example in `batch_labels_embed[0]`,
                # whereas l2 is not
                [
                    0,
                    0,
                ],  # Neither l0 nor l1 are positive examples in `batch_labels_embed[1]`
                [1, 0],  # l3 is an actual positive example in `batch_labels_embed[2]`,
                # whereas l1 is not
            ]
        )
    )


def test_multi_label_dot_product_loss__loss_sigmoid_is_ln2_when_all_similarities_zero():
    batch_size = 2
    num_candidates = 2
    sim_pos = tf.zeros([batch_size, 1, 1], dtype=tf.float32)
    sim_candidates_il = tf.zeros([batch_size, 1, num_candidates], dtype=tf.float32)
    pos_neg_labels = tf.cast(
        tf.random.uniform([batch_size, num_candidates]) < 0.5, tf.float32
    )

    layer = MultiLabelDotProductLoss(
        num_candidates, scale_loss=False, similarity_type=INNER
    )
    loss = layer._loss_sigmoid(sim_pos, sim_candidates_il, pos_neg_labels)
    assert abs(loss.numpy() - np.math.log(2.0)) < 1e-6


@pytest.mark.parametrize(
    "model_confidence, mock_similarities, expected_confidences",
    [
        # Confidence is always `1.0` since only one option exists and we use softmax
        (SOFTMAX, [[[-3.0], [0.0]]], [[[1.0], [1.0]]])
    ],
)
def test_dot_product_loss_get_similarities_and_confidences_from_embeddings(
    model_confidence: Text,
    mock_similarities: List,
    expected_confidences: List,
    monkeypatch: MonkeyPatch,
):
    def mock_sim(*args, **kwargs) -> tf.Tensor:
        return tf.constant(mock_similarities)

    monkeypatch.setattr(DotProductLoss, "sim", mock_sim)

    similarities, confidences = DotProductLoss(
        1, model_confidence=model_confidence
    ).get_similarities_and_confidences_from_embeddings(
        # Inputs are not used due to mocking of `sim`
        tf.zeros([1]),
        tf.zeros([1]),
        tf.zeros([1]),
    )
    assert np.all(similarities == mock_similarities)
    assert np.all(confidences == expected_confidences)


@pytest.mark.parametrize(
    "inputs, units, expected_output_shape",
    [
        (np.array([[1, 2], [4, 5], [7, 8]]), 4, (3, 4)),
        (np.array([[1, 2], [4, 5], [7, 8]]), 2, (3, 2)),
        (np.array([[1, 2], [4, 5], [7, 8]]), 5, (3, 5)),
        (np.array([[1, 2], [4, 5], [7, 8], [7, 8]]), 5, (4, 5)),
        (np.array([[[1, 2], [4, 5], [7, 8]]]), 4, (1, 3, 4)),
    ],
)
def test_randomly_connected_dense_shape(
    inputs: np.array, units: int, expected_output_shape: Tuple[int, ...]
):
    layer = RandomlyConnectedDense(units=units)
    y = layer(inputs)
    assert y.shape == expected_output_shape


@pytest.mark.parametrize(
    "inputs, units, expected_num_non_zero_outputs",
    [
        (np.array([[1, 2], [4, 5], [7, 8]]), 4, 12),
        (np.array([[1, 2], [4, 5], [7, 8]]), 2, 6),
        (np.array([[1, 2], [4, 5], [7, 8]]), 5, 15),
        (np.array([[1, 2], [4, 5], [7, 8], [7, 8]]), 5, 20),
        (np.array([[[1, 2], [4, 5], [7, 8]]]), 4, 12),
    ],
)
def test_randomly_connected_dense_output_always_dense(
    inputs: np.array, units: int, expected_num_non_zero_outputs: int
):
    layer = RandomlyConnectedDense(density=0.0, units=units, use_bias=False)
    y = layer(inputs)
    num_non_zero_outputs = tf.math.count_nonzero(y).numpy()
    assert num_non_zero_outputs == expected_num_non_zero_outputs


def test_randomly_connected_dense_all_inputs_connected():
    layer = RandomlyConnectedDense(density=0.0, units=2, use_bias=False)
    # Create a unit vector [1, 0, 0, 0, ...]
    x = np.zeros(10)
    x[0] = 1.0
    # For every standard basis vector
    for _ in range(10):
        x = np.roll(x, 1)
        y = layer(np.expand_dims(x, 0))
        assert tf.reduce_sum(y).numpy() != 0.0


@pytest.mark.parametrize(
    "layer_name, expected_feature_type",
    [
        (f"sparse_to_dense.{TEXT}_{FEATURE_TYPE_SENTENCE}", FEATURE_TYPE_SENTENCE),
        (
            f"sparse_to_dense.{LABEL}_{ACTION_TEXT}_{FEATURE_TYPE_SENTENCE}",
            FEATURE_TYPE_SENTENCE,
        ),
        (
            f"sparse_to_dense.{LABEL}_{ACTION_NAME}_{FEATURE_TYPE_SEQUENCE}",
            FEATURE_TYPE_SEQUENCE,
        ),
        (f"some_name.{DIALOGUE}_{FEATURE_TYPE_SEQUENCE}", FEATURE_TYPE_SEQUENCE),
        (f"some_name.{TEXT}_sentenc", None),
        (f"sparse_to_dense.{TEXT}_squence", None),
        ("some_name", None),
    ],
)
def test_dense_for_sparse_get_feature_type(
    layer_name: Text, expected_feature_type: Union[Text, None]
):
    layer = DenseForSparse(name=layer_name, units=10)
    assert layer.get_feature_type() == expected_feature_type


@pytest.mark.parametrize(
    "layer_name, expected_attribute",
    [
        (f"sparse_to_dense.{TEXT}_{FEATURE_TYPE_SEQUENCE}", TEXT),
        (f"sparse_to_dense.{INTENT}_{FEATURE_TYPE_SENTENCE}", INTENT),
        (f"other_name.{LABEL}_{FEATURE_TYPE_SENTENCE}", LABEL),
        (f"other_name.{DIALOGUE}_{FEATURE_TYPE_SENTENCE}", DIALOGUE),
        (f"sparse_to_dense.{ACTION_NAME}_{FEATURE_TYPE_SEQUENCE}", ACTION_NAME),
        (f"other_name.{ACTION_TEXT}_{FEATURE_TYPE_SENTENCE}", ACTION_TEXT),
        (
            f"other_name.{LABEL}_{ACTION_NAME}_{FEATURE_TYPE_SENTENCE}",
            f"{LABEL}_{ACTION_NAME}",
        ),
        (
            f"sparse_to_dense.{LABEL}_{ACTION_TEXT}_{FEATURE_TYPE_SEQUENCE}",
            f"{LABEL}_{ACTION_TEXT}",
        ),
        ("some_name", None),
        ("sparse_to_dense", None),
        (f"sparse_to_dense.{TEXT}", None),
        (f"sparse_to_dense.labl_{FEATURE_TYPE_SEQUENCE}", None),
    ],
)
def test_dense_for_sparse_get_attribute(
    layer_name: Text, expected_attribute: Union[Text, None]
):
    layer = DenseForSparse(name=layer_name, units=10)
    assert layer.get_attribute() == expected_attribute
