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
    assert np.all(similarity[0][0] == [-3.0])
    assert np.all(similarity[0][1] == [0.0])


def test_multi_label_dot_product_loss_call_shapes():
    num_neg = 1
    layer = MultiLabelDotProductLoss(num_neg, scale_loss=False, similarity_type=INNER)
    batch_inputs_embed = tf.constant([[[0, 1, 2]], [[-2, 0, 2]],], dtype=tf.float32)
    batch_labels_embed = tf.constant(
        [[[0, 0, 1], [1, 0, 0]], [[0, 1, 0], [1, 0, 0]],], dtype=tf.float32
    )
    batch_labels_ids = tf.constant([[[3], [2]], [[2], [1]],], dtype=tf.float32)
    all_labels_embed = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1],], dtype=tf.float32)
    all_labels_ids = tf.constant([[1], [2], [3],], dtype=tf.float32)
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

    i0 = [0, 0, 0]
    i1 = [1, 1, 1]
    i2 = [2, 2, 2]

    l0 = [11, 12, 13]
    l1 = [21, 22, 23]
    l2 = [31, 32, 33]
    l3 = [41, 42, 43]

    batch_inputs_embed = tf.constant([[i0], [i1], [i2]], dtype=tf.float32)
    batch_labels_embed = tf.constant([[l0, l1], [l2, l3], [l3, l0]], dtype=tf.float32)
    batch_labels_ids = tf.constant(
        [[[0], [1]], [[2], [3]], [[3], [0]]], dtype=tf.float32
    )
    all_labels_embed = tf.constant([l0, l1, l2, l3], dtype=tf.float32)
    all_labels_ids = tf.constant([[0], [1], [2], [3]], dtype=tf.float32)
    mask = None

    mock_indices = [0, 2, 0, 1, 0, 1]

    def mock_random_indices(*args, **kwargs) -> tf.Tensor:
        return tf.reshape(tf.constant(mock_indices), [batch_size, num_neg])

    monkeypatch.setattr(layers_utils, "random_indices", mock_random_indices)

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
    # The candidate label embeddings are picked according to the `mock_indices` above. -- ToDo: J: Why are all candidates picked from the "positive list"?
    # E.g. a 2 coming from `mock_indices` means that the first positive label (always) of
    # example 2 (`[l3, l0]`) is picked, i.e. `l3`.
    assert np.all(
        candidate_labels_embed.numpy() == np.array([[[l0, l3]], [[l0, l2]], [[l0, l2]]])
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
                ],  # l0 is an actual positive example in `batch_labels_embed[0]`, whereas l3 is not
                [
                    0,
                    1,
                ],  # l0 is not a positive example in `batch_labels_embed[1]`, whereas l2 is
                [
                    1,
                    0,
                ],  # l0 is an actual positive example in `batch_labels_embed[2]`, whereas l2 is not
            ]
        )
    )
