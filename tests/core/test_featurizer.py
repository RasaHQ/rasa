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
from rasa.core.constants import SLOTS, ACTIVE_LOOP
<<<<<<< HEAD
=======
from rasa.core.interpreter import RegexInterpreter
>>>>>>> master
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


<<<<<<< HEAD
def test_single_state_featurizer_correctly_encodes_state_without_interpreter():
    """
    Check that all the attributes are correctly featurized when they should and not featurized when shouldn't;
    These tests are for encoding state without a trained interpreter.
=======
def test_single_state_featurizer_without_interpreter_state_not_with_action_listen():
    """This test are for encoding state without a trained interpreter.
    action_name is not action_listen, so, INTENT, TEXT and ENTITIES should not be featurized
    """
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1, "action_listen": 2}
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}

    encoded = f.encode_state(
        {
            "user": {"intent": "a", "text": "blah blah blah"},
            "prev_action": {"action_name": "d", "action_text": "boom"},
            "active_loop": {"name": "i"},
            "slots": {"g": (1.0,)},
        },
        interpreter=RegexInterpreter(),
    )
    # user input is ignored as prev action is not action_listen
    assert list(encoded.keys()) == [ACTION_NAME, ACTIVE_LOOP, SLOTS]
    assert (
        encoded[ACTION_NAME][0].features != scipy.sparse.coo_matrix([[0, 1, 0]])
    ).nnz == 0
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 1, 0, 0]])
    ).nnz == 0
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[0, 0, 1]])).nnz == 0


def test_single_state_featurizer_without_interpreter_state_with_action_listen():
    """
    This test are for encoding state without a trained interpreter.
    action_name is action_listen, so, INTENT and ENTITIES should be featurized
    while text shouldn't because we don't have an interpreter.
>>>>>>> master
    """
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1, "action_listen": 2}
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}
<<<<<<< HEAD

    encoded = f.encode_state(
        {
            "user": {"intent": "a", "text": "blah blah blah"},
            "prev_action": {"action_name": "d", "action_text": "boom"},
            "active_loop": {"name": "i"},
            "slots": {"g": (1.0,)},
        },
        interpreter=None,
    )
    # user input is ignored as prev action is not action_listen
    assert list(encoded.keys()) == [ACTION_NAME, ACTIVE_LOOP, SLOTS]
    assert (
        encoded[ACTION_NAME][0].features != scipy.sparse.coo_matrix([[0, 1, 0]])
    ).nnz == 0
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 1, 0, 0]])
    ).nnz == 0
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[0, 0, 1]])).nnz == 0

    encoded = f.encode_state(
        {
            "user": {"intent": "a", "text": "blah blah blah"},
            "prev_action": {"action_name": "action_listen", "action_text": "boom"},
            "active_loop": {"name": "k"},
            "slots": {"e": (1.0,)},
        },
        interpreter=None,
    )
    # we featurize all the features except for *_text ones because NLU wasn't trained
    assert list(encoded.keys()) == [INTENT, ACTION_NAME, ACTIVE_LOOP, SLOTS]
    assert (encoded[INTENT][0].features != scipy.sparse.coo_matrix([[1, 0]])).nnz == 0
    assert (
        encoded[ACTION_NAME][0].features != scipy.sparse.coo_matrix([[0, 0, 1]])
    ).nnz == 0
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 0, 0, 1]])
    ).nnz == 0
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[1, 0, 0]])).nnz == 0

    # check that no intent / action_name features are added when the interpreter isn't there and
    # intent / action_name not in input
    encoded = f.encode_state(
        {
            "user": {"text": "blah blah blah"},
            "prev_action": {"action_text": "boom"},
            "active_loop": {"name": "k"},
            "slots": {"e": (1.0,)},
        },
        interpreter=None,
    )
    assert list(encoded.keys()) == [ACTIVE_LOOP, SLOTS]
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 0, 0, 1]])
    ).nnz == 0
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[1, 0, 0]])).nnz == 0

=======

    encoded = f.encode_state(
        {
            "user": {"intent": "a", "text": "blah blah blah"},
            "prev_action": {"action_name": "action_listen", "action_text": "boom"},
            "active_loop": {"name": "k"},
            "slots": {"e": (1.0,)},
        },
        interpreter=RegexInterpreter(),
    )
    # we featurize all the features except for *_text ones because NLU wasn't trained
    assert list(encoded.keys()) == [INTENT, ACTION_NAME, ACTIVE_LOOP, SLOTS]
    assert (encoded[INTENT][0].features != scipy.sparse.coo_matrix([[1, 0]])).nnz == 0
    assert (
        encoded[ACTION_NAME][0].features != scipy.sparse.coo_matrix([[0, 0, 1]])
    ).nnz == 0
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 0, 0, 1]])
    ).nnz == 0
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[1, 0, 0]])).nnz == 0


def test_single_state_featurizer_without_interpreter_state_no_intent_no_action_name():
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1, "action_listen": 2}
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}
    # check that no intent / action_name features are added when the interpreter isn't there and
    # intent / action_name not in input
    encoded = f.encode_state(
        {
            "user": {"text": "blah blah blah"},
            "prev_action": {"action_text": "boom"},
            "active_loop": {"name": "k"},
            "slots": {"e": (1.0,)},
        },
        interpreter=RegexInterpreter(),
    )
    assert list(encoded.keys()) == [ACTIVE_LOOP, SLOTS]
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 0, 0, 1]])
    ).nnz == 0
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[1, 0, 0]])).nnz == 0


def test_single_state_featurizer_correctly_encodes_non_existing_value():
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1}
    encoded = f.encode_state(
        {"user": {"intent": "e"}, "prev_action": {"action_name": "action_listen"}},
        interpreter=RegexInterpreter(),
    )
    assert list(encoded.keys()) == [INTENT, ACTION_NAME]
    assert (encoded[INTENT][0].features != scipy.sparse.coo_matrix([[0, 0]])).nnz == 0


def test_single_state_featurizer_creates_encoded_all_actions():
    from rasa.core.actions.action import default_action_names

    domain = Domain(
        intents=[],
        entities=[],
        slots=[],
        templates={},
        forms=[],
        action_names=["a", "b", "c", "d"],
    )
    f = SingleStateFeaturizer()
    f.prepare_from_domain(domain)
    encoded_actions = f.encode_all_actions(domain, RegexInterpreter())
    assert len(encoded_actions) == len(domain.action_names)
    assert all(
        [
            ACTION_NAME in encoded_action and ACTION_TEXT not in encoded_action
            for encoded_action in encoded_actions
        ]
    )
>>>>>>> master

def test_single_state_featurizer_correctly_encodes_non_existing_value():
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1}
    encoded = f.encode_state(
        {"user": {"intent": "e"}, "prev_action": {"action_name": "action_listen"}},
        interpreter=None,
    )
    assert list(encoded.keys()) == [INTENT, ACTION_NAME]
    assert (encoded[INTENT][0].features != scipy.sparse.coo_matrix([[0, 0]])).nnz == 0

<<<<<<< HEAD
=======
def test_single_state_featurizer_uses_dtype_float():
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"e": 0, "d": 1}
    f._default_feature_states[ENTITIES] = {"c": 0}
    encoded = f.encode_state(
        {
            "user": {"intent": "a", "entities": ["c"]},
            "prev_action": {"action_name": "d"},
        },
        interpreter=RegexInterpreter(),
    )
    assert encoded[ACTION_NAME][0].features.dtype == np.float32
>>>>>>> master

def test_single_state_featurizer_creates_encoded_all_actions():
    from rasa.core.actions.action import default_action_names

<<<<<<< HEAD
    domain = Domain(
        intents=[],
        entities=[],
        slots=[],
        templates={},
        forms=[],
        action_names=["a", "b", "c", "d"],
    )
    f = SingleStateFeaturizer()
    f.prepare_from_domain(domain)
    encoded_actions = f.encode_all_actions(domain, None)
    assert len(encoded_actions) == len(domain.action_names)
    assert all(
        [
            ACTION_NAME in encoded_action and ACTION_TEXT not in encoded_action
            for encoded_action in encoded_actions
        ]
    )
=======
def test_single_state_featurizer_with_interpreter_state_with_action_listen(
    unpacked_trained_moodbot_path: Text,
):
    from rasa.core.agent import Agent
>>>>>>> master

    interpreter = Agent.load(unpacked_trained_moodbot_path).interpreter

<<<<<<< HEAD
def test_single_state_featurizer_uses_dtype_float():
    f = SingleStateFeaturizer()
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


def test_single_state_featurizer_correctly_encodes_state_with_interpreter(
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
    f._default_feature_states[ENTITIES] = {"c": 0}
    f._default_feature_states[ACTION_NAME] = {"e": 0, "d": 1, "action_listen": 2}
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}
    encoded = f.encode_state(
        {
            "user": {"text": "a ball", "intent": "b", "entities": ["c"]},
            "prev_action": {
                "action_name": "action_listen",
                "action_text": "throw a ball",
            },
            "active_loop": {"name": "k"},
            "slots": {"e": (1.0,)},
        },
        interpreter=kwargs["interpreter"],
    )
    # check all the features are encoded and *_text features are encoded by a densefeaturizer
    assert sorted(list(encoded.keys())) == sorted(
        [TEXT, ENTITIES, ACTION_NAME, SLOTS, ACTIVE_LOOP, INTENT, ACTION_TEXT]
    )
    assert encoded[TEXT][0].features.shape[-1] == 300
    assert encoded[ACTION_TEXT][0].features.shape[-1] == 300
    assert (encoded[INTENT][0].features != scipy.sparse.coo_matrix([[0, 1]])).nnz == 0
    assert (
        encoded[ACTION_NAME][0].features != scipy.sparse.coo_matrix([[0, 0, 1]])
    ).nnz == 0
    assert encoded[ENTITIES][0].features.shape[-1] == 1
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[1, 0, 0]])).nnz == 0
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 0, 0, 1]])
    ).nnz == 0

    encoded = f.encode_state(
        {
            "user": {"text": "a ball", "intent": "b", "entities": ["c"]},
            "prev_action": {"action_name": "d", "action_text": "throw a ball"},
            "active_loop": {"name": "k"},
            "slots": {"e": (1.0,)},
        },
        interpreter=kwargs["interpreter"],
    )
=======
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ENTITIES] = {"c": 0}
    f._default_feature_states[ACTION_NAME] = {"e": 0, "d": 1, "action_listen": 2}
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}
    encoded = f.encode_state(
        {
            "user": {"text": "a ball", "intent": "b", "entities": ["c"]},
            "prev_action": {
                "action_name": "action_listen",
                "action_text": "throw a ball",
            },
            "active_loop": {"name": "k"},
            "slots": {"e": (1.0,)},
        },
        interpreter=interpreter,
    )
    # check all the features are encoded and *_text features are encoded by a densefeaturizer
    assert sorted(list(encoded.keys())) == sorted(
        [TEXT, ENTITIES, ACTION_NAME, SLOTS, ACTIVE_LOOP, INTENT, ACTION_TEXT]
    )
    assert encoded[TEXT][0].features.shape[-1] == 300
    assert encoded[ACTION_TEXT][0].features.shape[-1] == 300
    assert (encoded[INTENT][0].features != scipy.sparse.coo_matrix([[0, 1]])).nnz == 0
    assert (
        encoded[ACTION_NAME][0].features != scipy.sparse.coo_matrix([[0, 0, 1]])
    ).nnz == 0
    assert encoded[ENTITIES][0].features.shape[-1] == 1
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[1, 0, 0]])).nnz == 0
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 0, 0, 1]])
    ).nnz == 0


def test_single_state_featurizer_with_interpreter_state_not_with_action_listen(
    unpacked_trained_moodbot_path: Text,
):
    # check that user features are ignored when action_name is not action_listen
    from rasa.core.agent import Agent

    interpreter = Agent.load(unpacked_trained_moodbot_path).interpreter
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ENTITIES] = {"c": 0}
    f._default_feature_states[ACTION_NAME] = {"e": 0, "d": 1, "action_listen": 2}
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}
    encoded = f.encode_state(
        {
            "user": {"text": "a ball", "intent": "b", "entities": ["c"]},
            "prev_action": {"action_name": "d", "action_text": "throw a ball"},
            "active_loop": {"name": "k"},
            "slots": {"e": (1.0,)},
        },
        interpreter=interpreter,
    )
>>>>>>> master
    # check user input is ignored when action is not action_listen
    assert list(encoded.keys()) == [ACTION_TEXT, ACTION_NAME, ACTIVE_LOOP, SLOTS]
    assert encoded[ACTION_TEXT][0].features.shape[-1] == 300
    assert (
        encoded[ACTION_NAME][0].features != scipy.sparse.coo_matrix([[0, 1, 0]])
    ).nnz == 0
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[1, 0, 0]])).nnz == 0
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 0, 0, 1]])
    ).nnz == 0

<<<<<<< HEAD
    encoded = f.encode_state(
        {
            "user": {"text": "a ball", "entities": ["c"]},
            "prev_action": {
                "action_name": "action_listen",
                "action_text": "throw a ball",
            },
            "active_loop": {"name": "k"},
            "slots": {"e": (1.0,)},
        },
        interpreter=kwargs["interpreter"],
    )
    # check all the features are encoded and *_text features are encoded by a densefeaturizer
    # and intent features are not added
    assert sorted(list(encoded.keys())) == sorted(
        [TEXT, ENTITIES, ACTION_NAME, SLOTS, ACTIVE_LOOP, ACTION_TEXT]
    )
    assert encoded[TEXT][0].features.shape[-1] == 300
    assert encoded[ACTION_TEXT][0].features.shape[-1] == 300
    assert (
        encoded[ACTION_NAME][0].features != scipy.sparse.coo_matrix([[0, 0, 1]])
    ).nnz == 0
    assert encoded[ENTITIES][0].features.shape[-1] == 1
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[1, 0, 0]])).nnz == 0
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 0, 0, 1]])
    ).nnz == 0

=======

def test_single_state_featurizer_with_interpreter_state_with_no_action_name(
    unpacked_trained_moodbot_path: Text,
):
    # check that action name features are not added by the featurizer when not
    # present in the state and
    # check user input is ignored when action is not action_listen
    # and action_name is features are not added
    from rasa.core.agent import Agent

    interpreter = Agent.load(unpacked_trained_moodbot_path).interpreter
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ENTITIES] = {"c": 0}
    f._default_feature_states[ACTION_NAME] = {"e": 0, "d": 1, "action_listen": 2}
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}
>>>>>>> master
    encoded = f.encode_state(
        {
            "user": {"text": "a ball", "intent": "b", "entities": ["c"]},
            "prev_action": {"action_text": "throw a ball"},
            "active_loop": {"name": "k"},
            "slots": {"e": (1.0,)},
        },
<<<<<<< HEAD
        interpreter=kwargs["interpreter"],
    )
    # check user input is ignored when action is not action_listen
    # and action_name is features are not added
=======
        interpreter=interpreter,
    )
>>>>>>> master
    assert list(encoded.keys()) == [ACTION_TEXT, ACTIVE_LOOP, SLOTS]
    assert encoded[ACTION_TEXT][0].features.shape[-1] == 300
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[1, 0, 0]])).nnz == 0
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 0, 0, 1]])
    ).nnz == 0
