"""Tests for CRF."""

# original code taken from
# https://github.com/tensorflow/addons/blob/master/tensorflow_addons/text/tests/crf_test.py
# (modified to our neeeds)

import itertools

import pytest
import numpy as np
import tensorflow as tf

from rasa.utils.tensorflow.crf import (
    crf_sequence_score,
    crf_unary_score,
    crf_binary_score,
    crf_log_norm,
    crf_log_likelihood,
)


def calculate_sequence_score(inputs, transition_params, tag_indices, sequence_lengths):
    expected_unary_score = sum(
        inputs[i][tag_indices[i]] for i in range(sequence_lengths)
    )
    expected_binary_score = sum(
        transition_params[tag_indices[i], tag_indices[i + 1]]
        for i in range(sequence_lengths - 1)
    )
    return expected_unary_score + expected_binary_score


def brute_force_decode(sequence_lengths, inputs, transition_params):
    num_words = inputs.shape[0]
    num_tags = inputs.shape[1]

    all_sequence_scores = []
    all_sequences = []

    tag_indices_iterator = itertools.product(range(num_tags), repeat=sequence_lengths)
    inputs = tf.expand_dims(inputs, 0)
    sequence_lengths = tf.expand_dims(sequence_lengths, 0)
    transition_params = tf.constant(transition_params)

    # Compare the dynamic program with brute force computation.
    for tag_indices in tag_indices_iterator:
        tag_indices = list(tag_indices)
        tag_indices.extend([0] * (num_words - sequence_lengths))
        all_sequences.append(tag_indices)
        sequence_score = crf_sequence_score(
            inputs=inputs,
            tag_indices=tf.expand_dims(tag_indices, 0),
            sequence_lengths=sequence_lengths,
            transition_params=transition_params,
        )
        sequence_score = tf.squeeze(sequence_score, [0])
        all_sequence_scores.append(sequence_score)

    expected_max_sequence_index = np.argmax(all_sequence_scores)
    expected_max_sequence = all_sequences[expected_max_sequence_index]
    expected_max_score = all_sequence_scores[expected_max_sequence_index]
    return expected_max_sequence, expected_max_score


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_sequence_score(dtype):
    transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=dtype)
    # Test both the length-1 and regular cases.
    sequence_lengths_list = [
        np.array(3, dtype=np.int32),
        np.array(1, dtype=np.int32),
    ]
    inputs_list = [
        np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=dtype),
        np.array([[4, 5, -3]], dtype=dtype),
    ]
    tag_indices_list = [
        np.array([1, 2, 1, 0], dtype=np.int32),
        np.array([1], dtype=np.int32),
    ]
    for sequence_lengths, inputs, tag_indices in zip(
        sequence_lengths_list, inputs_list, tag_indices_list
    ):
        sequence_score = crf_sequence_score(
            inputs=tf.expand_dims(inputs, 0),
            tag_indices=tf.expand_dims(tag_indices, 0),
            sequence_lengths=tf.expand_dims(sequence_lengths, 0),
            transition_params=tf.constant(transition_params),
        )
        sequence_score = tf.squeeze(sequence_score, [0])

        expected_sequence_score = calculate_sequence_score(
            inputs, transition_params, tag_indices, sequence_lengths
        )
        np.testing.assert_allclose(sequence_score, expected_sequence_score)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_unary_score(dtype):
    inputs = np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=dtype)
    for dtype in (np.int32, np.int64):
        tag_indices = np.array([1, 2, 1, 0], dtype=dtype)
        sequence_lengths = np.array(3, dtype=np.int32)
        unary_score = crf_unary_score(
            tag_indices=tf.expand_dims(tag_indices, 0),
            sequence_lengths=tf.expand_dims(sequence_lengths, 0),
            inputs=tf.expand_dims(inputs, 0),
        )
        unary_score = tf.squeeze(unary_score, [0])
        expected_unary_score = sum(
            inputs[i][tag_indices[i]] for i in range(sequence_lengths)
        )
        np.testing.assert_allclose(unary_score, expected_unary_score)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_binary_score(dtype):
    tag_indices = np.array([1, 2, 1, 0], dtype=np.int32)
    transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=dtype)
    sequence_lengths = np.array(3, dtype=np.int32)
    binary_score = crf_binary_score(
        tag_indices=tf.expand_dims(tag_indices, 0),
        sequence_lengths=tf.expand_dims(sequence_lengths, 0),
        transition_params=tf.constant(transition_params),
    )
    binary_score = tf.squeeze(binary_score, [0])
    expected_binary_score = sum(
        transition_params[tag_indices[i], tag_indices[i + 1]]
        for i in range(sequence_lengths - 1)
    )
    np.testing.assert_allclose(binary_score, expected_binary_score)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_log_norm(dtype):
    transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=dtype)
    # Test both the length-1 and regular cases.
    sequence_lengths_list = [
        np.array(3, dtype=np.int32),
        np.array(1, dtype=np.int64),
    ]
    inputs_list = [
        np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=dtype),
        np.array([[3, -1, 3]], dtype=dtype),
    ]
    tag_indices_list = [
        np.array([1, 2, 1, 0], dtype=np.int32),
        np.array([2], dtype=np.int32),
    ]

    for sequence_lengths, inputs, tag_indices in zip(
        sequence_lengths_list, inputs_list, tag_indices_list
    ):
        num_words = inputs.shape[0]
        num_tags = inputs.shape[1]
        all_sequence_scores = []

        # Compare the dynamic program with brute force computation.
        for tag_indices in itertools.product(range(num_tags), repeat=sequence_lengths):
            tag_indices = list(tag_indices)
            tag_indices.extend([0] * (num_words - sequence_lengths))
            all_sequence_scores.append(
                crf_sequence_score(
                    inputs=tf.expand_dims(inputs, 0),
                    tag_indices=tf.expand_dims(tag_indices, 0),
                    sequence_lengths=tf.expand_dims(sequence_lengths, 0),
                    transition_params=tf.constant(transition_params),
                )
            )

        brute_force_log_norm = tf.reduce_logsumexp(all_sequence_scores)
        log_norm = crf_log_norm(
            inputs=tf.expand_dims(inputs, 0),
            sequence_lengths=tf.expand_dims(sequence_lengths, 0),
            transition_params=tf.constant(transition_params),
        )
        log_norm = tf.squeeze(log_norm, [0])

        np.testing.assert_allclose(log_norm, brute_force_log_norm)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_log_norm_zero_seq_length(dtype):
    """Test `crf_log_norm` when `sequence_lengths` contains one or more
    zeros."""
    inputs = tf.constant(np.ones([2, 10, 5], dtype=dtype))
    transition_params = tf.constant(np.ones([5, 5], dtype=dtype))
    sequence_lengths = tf.constant(np.zeros([2], dtype=np.int32))
    expected_log_norm = np.zeros([2], dtype=dtype)
    log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)
    np.testing.assert_allclose(log_norm, expected_log_norm)


@pytest.mark.parametrize("dtype", [np.float32])
def test_crf_log_likelihood(dtype):
    inputs = np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=dtype)
    transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=dtype)
    sequence_lengths = np.array(3, dtype=np.int32)

    num_words = inputs.shape[0]
    num_tags = inputs.shape[1]
    all_sequence_log_likelihoods = []

    # Make sure all probabilities sum to 1.
    for tag_indices in itertools.product(range(num_tags), repeat=sequence_lengths):
        tag_indices = list(tag_indices)
        tag_indices.extend([0] * (num_words - sequence_lengths))
        sequence_log_likelihood, _ = crf_log_likelihood(
            inputs=tf.expand_dims(inputs, 0),
            tag_indices=tf.expand_dims(tag_indices, 0),
            sequence_lengths=tf.expand_dims(sequence_lengths, 0),
            transition_params=tf.constant(transition_params),
        )
        all_sequence_log_likelihoods.append(sequence_log_likelihood)
    total_log_likelihood = tf.reduce_logsumexp(all_sequence_log_likelihoods)
    np.testing.assert_allclose(total_log_likelihood, 0.0, rtol=1e-6, atol=1e-6)

    # check if `transition_params = None` raises an error
    crf_log_likelihood(
        inputs=tf.expand_dims(inputs, 0),
        tag_indices=tf.expand_dims(tag_indices, 0),
        sequence_lengths=tf.expand_dims(sequence_lengths, 0),
    )


def test_different_dtype():
    inputs = np.ones([16, 20, 5], dtype=np.float32)
    tags = tf.convert_to_tensor(np.ones([16, 20], dtype=np.int64))
    seq_lens = np.ones([16], dtype=np.int64) * 20

    loss, _ = crf_log_likelihood(
        inputs=inputs, tag_indices=tags, sequence_lengths=seq_lens
    )
