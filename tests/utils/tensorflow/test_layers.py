import pytest
from _pytest.monkeypatch import MonkeyPatch
import numpy as np
import tensorflow as tf
from rasa.utils.tensorflow.layers import DotProductLoss, MultiLabelDotProductLoss
from rasa.utils.tensorflow.constants import INNER, CROSS_ENTROPY, SOFTMAX
import rasa.utils.tensorflow.layers_utils as layers_utils


def test_dot_product_loss_inner_sim():
    layer = DotProductLoss(
        0,
        scale_loss=False,
        similarity_type=INNER,
        constrain_similarities=False,
        model_confidence=SOFTMAX,
    )
    a = tf.constant([[[1.0, 0.0, 2.0]], [[1.0, 0.0, 2.0]]])
    b = tf.constant([[[1.0, 0.0, -2.0]], [[1.0, 0.0, -2.0]]])
    mask = tf.constant([[1.0, 0.0]])
    similarity = layer.sim(a, b, mask=mask).numpy()
    assert np.all(similarity == [[[-3.0], [0.0]]])


def test_multi_label_dot_product_loss_call_shapes():
    num_neg = 1
    layer = MultiLabelDotProductLoss(num_neg, scale_loss=False, similarity_type=INNER)
    batch_inputs_embed = tf.constant([[[0, 1, 2]], [[-2, 0, 2]],], dtype=tf.float32)
    batch_labels_embed = tf.constant(
        [[[0, 0, 1], [1, 0, 0]], [[0, 1, 0], [1, 0, 0]],], dtype=tf.float32
    )
    batch_labels_ids = tf.constant([[[2], [1]], [[1], [0]],], dtype=tf.float32)
    all_labels_embed = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1],], dtype=tf.float32)
    all_labels_ids = tf.constant([[0], [1], [2],], dtype=tf.float32)
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


def test_multi_label_dot_product_loss__get_candidate_indices_shape():
    batch_size = 3
    num_candidates = 4
    total_candidates = 5
    layer = MultiLabelDotProductLoss(
        num_candidates, scale_loss=False, similarity_type=INNER
    )
    candidate_ids = layer._get_candidate_indices(batch_size, total_candidates)

    assert np.all(tf.shape(candidate_ids).numpy() == [batch_size, num_candidates])


def test_multi_label_dot_product_loss__get_candidate_values():
    x = tf.reshape(tf.range(3 * 3), [3, 3])
    candidate_ids = tf.constant([[0, 1], [0, 0], [2, 0]])
    candidate_values = MultiLabelDotProductLoss._get_candidate_values(
        x, candidate_ids
    ).numpy()
    expected_candidate_values = np.array(
        [[[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [0, 1, 2]], [[6, 7, 8], [0, 1, 2]]]
    )
    expected_candidate_values = np.expand_dims(expected_candidate_values, axis=1)

    assert np.all(candidate_values == expected_candidate_values)


def test_multi_label_dot_product_loss__sample_candidates_with_constant_number_of_labels(
    monkeypatch: MonkeyPatch,
):
    num_neg = 2
    batch_size = 3
    layer = MultiLabelDotProductLoss(num_neg, scale_loss=False, similarity_type=INNER)

    # Some random input embeddings
    i0 = [0, 0, 0]
    i1 = [1, 1, 1]
    i2 = [2, 2, 2]

    # Some random label embeddings
    l0 = [11, 12, 13]
    l1 = [21, 22, 23]
    l2 = [31, 32, 33]
    l3 = [41, 42, 43]

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
    # Forget about masks for now
    mask = None

    # Inside `layer._sample_candidates` random indices will be generated for the
    # candidates. We mock them to have a deterministic output.
    mock_indices = [0, 2, 0, 1, 0, 3]

    def mock_random_indices(*args, **kwargs) -> tf.Tensor:
        return tf.reshape(tf.constant(mock_indices), [batch_size, num_neg])

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
    # The first example labels of each batch are in `pos_labels_embed`
    assert np.all(pos_labels_embed.numpy() == np.array([[[l0]], [[l2]], [[l3]]]))
    # The candidate label embeddings are picked according to the `mock_indices` above. -- ToDo: J: Pick candidates from ALL labels
    # E.g. a 2 coming from `mock_indices` means that the first positive label (always) of
    # example 2 (`[l3, l0]`) is picked, i.e. `l3`.
    assert np.all(
        candidate_labels_embed.numpy() == np.array([[[l0, l2]], [[l0, l1]], [[l0, l3]]])
    )
    # The `pos_neg_labels` contains `1`s wherever the vector in `candidate_labels_embed` of example `i` is actually in the
    # possible lables of example `i`
    assert np.all(
        pos_neg_labels.numpy()
        == np.array(
            [
                [
                    1,
                    0,
                ],  # l0 is an actual positive example in `batch_labels_embed[0]`, whereas l2 is not
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
    num_neg = 2
    batch_size = 3
    layer = MultiLabelDotProductLoss(num_neg, scale_loss=False, similarity_type=INNER)

    # Some random input embeddings
    i0 = [0, 0, 0]
    i1 = [1, 1, 1]
    i2 = [2, 2, 2]

    # Some random label embeddings
    l0 = [11, 12, 13]
    l1 = [21, 22, 23]
    l2 = [31, 32, 33]
    l3 = [41, 42, 43]

    # Label used for padding
    lp = [-1, -1, -1]

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
    # Forget about masks for now
    mask = None

    # Inside `layer._sample_candidates` random indices will be generated for the
    # candidates. We mock them to have a deterministic output.
    mock_indices = [0, 2, 0, 1, 3, 1]

    def mock_random_indices(*args, **kwargs) -> tf.Tensor:
        return tf.reshape(tf.constant(mock_indices), [batch_size, num_neg])

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
    # The first example labels of each batch are in `pos_labels_embed`
    assert np.all(pos_labels_embed.numpy() == np.array([[[l0]], [[l2]], [[l3]]]))
    # The candidate label embeddings are picked according to the `mock_indices` above. -- ToDo: J: Pick candidates from ALL labels
    # E.g. a 2 coming from `mock_indices` means that the first positive label (always) of
    # example 2 (`[l3, l0, _]`) is picked, i.e. `l3`.
    assert np.all(
        candidate_labels_embed.numpy() == np.array([[[l0, l2]], [[l0, l1]], [[l3, l1]]])
    )
    # The `pos_neg_labels` contains `1`s wherever the vector in `candidate_labels_embed` of example `i` is actually in the
    # possible lables of example `i`
    assert np.all(
        pos_neg_labels.numpy()
        == np.array(
            [
                [
                    1,
                    0,
                ],  # l0 is an actual positive example in `batch_labels_embed[0]`, whereas l2 is not
                [
                    0,
                    0,
                ],  # Neither l0 nor l1 are positive examples in `batch_labels_embed[1]`
                [
                    1,
                    0,
                ],  # l3 is an actual positive example in `batch_labels_embed[2]`, whereas l1 is not
            ]
        )
    )


def test_multi_label_dot_product_loss__loss_sigmoid_is_ln2_when_all_similarities_zero():
    sim_pos = tf.zeros([2, 1, 1], dtype=tf.float32)
    sim_candidates_il = tf.zeros([2, 1, 2], dtype=tf.float32)
    pos_neg_labels = tf.cast(tf.random.uniform([2, 2]) < 0.5, tf.float32)
    mask = None

    layer = MultiLabelDotProductLoss(2, scale_loss=False, similarity_type=INNER)
    loss = layer._loss_sigmoid(sim_pos, sim_candidates_il, pos_neg_labels, mask)
    assert abs(loss.numpy() - np.math.log(2.0)) < 1e-6
