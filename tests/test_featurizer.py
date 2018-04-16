from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.featurizers import TrackerFeaturizer, \
    BinarySingleStateFeaturizer, ProbabilisticSingleStateFeaturizer
import numpy as np


def test_fail_to_load_non_existent_featurizer():
    assert TrackerFeaturizer.load("non_existent_class") is None


# TODO featurizers changed quite a lot, testing SingleStateFeaturizer
def test_binary_featurizer_handles_on_non_existing_features():
    f = BinarySingleStateFeaturizer()
    f.input_state_map = {"a": 0, "b": 3, "c": 2, "d": 1}
    f.num_features = len(f.input_state_map)
    encoded = f.encode({"a": 1.0, "b": 1.0, "c": 0.0, "e": 1.0})
    assert (encoded == np.array([1, 0, 0, 1])).all()


def test_binary_featurizer_uses_correct_dtype_int():
    f = BinarySingleStateFeaturizer()
    f.input_state_map = {"a": 0, "b": 3, "c": 2, "d": 1}
    f.num_features = len(f.input_state_map)
    encoded = f.encode({"a": 1.0, "b": 1.0, "c": 0.0})
    assert encoded.dtype == np.int32


def test_binary_featurizer_uses_correct_dtype_float():
    f = BinarySingleStateFeaturizer()
    f.input_state_map = {"a": 0, "b": 3, "c": 2, "d": 1}
    f.num_features = len(f.input_state_map)
    encoded = f.encode({"a": 1.0, "b": 0.2, "c": 0.0})
    assert encoded.dtype == np.float64


def test_probabilistic_featurizer_handles_on_non_existing_features():
    f = ProbabilisticSingleStateFeaturizer()
    f.input_state_map = {"a": 0, "b": 3, "c": 2, "d": 1}
    f.num_features = len(f.input_state_map)
    encoded = f.encode({"a": 1.0, "b": 0.2, "c": 0.0, "e": 1.0})
    assert (encoded == np.array([1, 0, 0, 0.2])).all()


def test_probabilistic_featurizer_handles_intent_probs():
    f = ProbabilisticSingleStateFeaturizer()
    f.input_state_map = {"intent_a": 0, "b": 3, "intent_c": 2, "d": 1}
    f.num_features = len(f.input_state_map)
    encoded = f.encode({"intent_a": 0.5, "b": 0.2, "intent_c": 1.0})
    assert (encoded == np.array([0.5, 0, 1.0, 0.2])).all()
