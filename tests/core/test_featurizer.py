import logging
from typing import Text
from unittest.mock import Mock
import sys
import asyncio

from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.featurizers.single_state_featurizer import (
    BinarySingleStateFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
    SingleStateFeaturizer,
)
import rasa.core
from rasa.train import train_core, train_nlu, train
from rasa.core.domain import Domain
import numpy as np
from rasa.nlu.constants import (
    TEXT,
    INTENT,
    ACTION_NAME,
    ACTION_TEXT,
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
)
import scipy.sparse
from _pytest.monkeypatch import MonkeyPatch
from pathlib import Path
from tests.conftest import DEFAULT_CONFIG_PATH, DEFAULT_NLU_DATA
from tests.core.conftest import (
    DEFAULT_DOMAIN_PATH_WITH_SLOTS,
    DEFAULT_STORIES_FILE,
)


def test_fail_to_load_non_existent_featurizer():
    assert TrackerFeaturizer.load("non_existent_class") is None


def test_binary_featurizer_correctly_encodes_state():
    f = BinarySingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1}
    encoded = f.encode_state(
        {"user": {"intent": "a"}, "prev_action": {"action_name": "d"}}, interpreter=None
    )
    # user input is ignored as prev action is not action_listen;
    assert list(encoded.keys()) == [ACTION_NAME]
    assert (
        encoded[ACTION_NAME][0].features != scipy.sparse.coo_matrix([[0, 1]])
    ).nnz == 0

    encoded = f.encode_state(
        {"user": {"intent": "a"}, "prev_action": {"action_name": "action_listen"}},
        interpreter=None,
    )
    assert list(encoded.keys()) == [INTENT, ACTION_NAME]
    assert (encoded[INTENT][0].features != scipy.sparse.coo_matrix([[1, 0]])).nnz == 0


def test_binary_featurizer_correctly_encodes_non_existing_feature():
    f = BinarySingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1}
    encoded = f.encode_state(
        {"user": {"intent": "e"}, "prev_action": {"action_name": "action_listen"}},
        interpreter=None,
    )
    assert list(encoded.keys()) == [INTENT, ACTION_NAME]
    assert (encoded[INTENT][0].features != scipy.sparse.coo_matrix([[0, 0]])).nnz == 0


def test_binary_featurizer_creates_encoded_all_actions():
    from rasa.core.actions.action import default_action_names

    domain = Domain(
        intents=[],
        entities=[],
        slots=[],
        templates={},
        forms=[],
        action_names=["a", "b", "c", "d"],
    )
    f = BinarySingleStateFeaturizer()
    f.prepare_from_domain(domain)
    encoded_actions = f.encode_all_actions(domain, None)
    assert len(encoded_actions) == len(domain.action_names)
    assert all(
        [
            ACTION_NAME in encoded_action and ACTION_TEXT not in encoded_action
            for encoded_action in encoded_actions
        ]
    )


def test_binary_featurizer_uses_dtype_float():
    f = BinarySingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"e": 0, "d": 1}
    f._default_feature_states[ENTITIES] = {"c": 0}
    encoded = f.encode_state(
        {
            "user": {"intent": "a", "entities": ["c"]},
            "prev_action": {"action_name": "d"},
        },
        interpreter=None,
    )
    assert encoded[ACTION_NAME][0].features.dtype == np.float32


def test_single_state_featurizer_correctly_encodes_state(
    monkeypatch: MonkeyPatch, tmp_path: Path, unpacked_trained_moodbot_path: Text
):
    # Skip actual NLU training and return trained interpreter path from fixture
    _train_nlu_with_validated_data = Mock(return_value=unpacked_trained_moodbot_path)

    # Patching is bit more complicated as we have a module `train` and function
    # with the same name 😬
    monkeypatch.setattr(
        sys.modules["rasa.train"],
        "_train_nlu_with_validated_data",
        asyncio.coroutine(_train_nlu_with_validated_data),
    )

    # Mock the actual Core training
    _train_core = Mock()
    monkeypatch.setattr(rasa.core, "train", asyncio.coroutine(_train_core))

    train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_CONFIG_PATH,
        [DEFAULT_STORIES_FILE, DEFAULT_NLU_DATA],
        str(tmp_path),
    )

    _train_core.assert_called_once()
    _, _, kwargs = _train_core.mock_calls[0]

    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"e": 0, "d": 1}
    f._default_feature_states[ENTITIES] = {"c": 0}
    encoded = f.encode_state(
        {
            "user": {"text": "a ball", "entities": ["c"]},
            "prev_action": {"action_name": "action_listen"},
        },
        interpreter=kwargs["interpreter"],
    )
    assert all([attribute in encoded for attribute in [TEXT, ENTITIES, ACTION_NAME]])
    assert encoded[TEXT][0].features.shape[-1] == 300
    assert encoded[ACTION_NAME][0].features.shape[-1] == 2
    assert encoded[ENTITIES][0].features.shape[-1] == 1
