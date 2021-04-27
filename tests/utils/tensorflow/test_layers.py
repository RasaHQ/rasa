import pytest
import numpy as np
import tensorflow as tf
from rasa.utils.tensorflow.layers import DotProductLoss, MultiLabelDotProductLoss
from rasa.utils.tensorflow.constants import INNER, CROSS_ENTROPY, SOFTMAX


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


def test_multi_label_dot_product_loss__get_candidate_values_shape():
    pass
