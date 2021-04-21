import pytest
import numpy as np
import tensorflow as tf
from rasa.utils.tensorflow.layers import DotProductLoss
from rasa.utils.tensorflow.constants import INNER, CROSS_ENTROPY, SOFTMAX


def test_dot_product_loss_inner_sim():
    loss = DotProductLoss(
        0,
        loss_type=CROSS_ENTROPY,
        mu_pos=0.0,
        mu_neg=0.0,
        use_max_sim_neg=False,
        neg_lambda=0.0,
        scale_loss=False,
        similarity_type=INNER,
        constrain_similarities=False,
        model_confidence=SOFTMAX,
    )
    a = tf.constant([[[1.0, 0.0, 2.0]], [[1.0, 0.0, 2.0]]])
    b = tf.constant([[[1.0, 0.0, -2.0]], [[1.0, 0.0, -2.0]]])
    mask = tf.constant([[1.0, 0.0]])
    similarity = loss.sim(a, b, mask=mask).numpy()
    assert np.all(similarity[0][0] == [-3.0])
    assert np.all(similarity[0][1] == [0.0])
