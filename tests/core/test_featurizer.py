from typing import Text
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.shared.core.domain import Domain
import numpy as np
from rasa.shared.nlu.constants import ACTION_TEXT, ACTION_NAME, ENTITIES, TEXT, INTENT
from rasa.shared.core.constants import ACTIVE_LOOP, SLOTS, ENTITY_LABEL_SEPARATOR
from rasa.shared.nlu.interpreter import RegexInterpreter
import scipy.sparse


def test_fail_to_load_non_existent_featurizer():
    assert TrackerFeaturizer.load("non_existent_class") is None


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
    """
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1, "action_listen": 2}
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}

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
    domain = Domain(
        intents=[],
        entities=[],
        slots=[],
        templates={},
        forms={},
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


def test_single_state_featurizer_with_entity_roles_and_groups(
    unpacked_trained_moodbot_path: Text,
):
    from rasa.core.agent import Agent

    interpreter = Agent.load(unpacked_trained_moodbot_path).interpreter

    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ENTITIES] = {
        "c": 0,
        "d": 1,
        f"d{ENTITY_LABEL_SEPARATOR}e": 2,
    }
    f._default_feature_states[ACTION_NAME] = {"e": 0, "d": 1, "action_listen": 2}
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}
    encoded = f.encode_state(
        {
            "user": {
                "text": "a ball",
                "intent": "b",
                "entities": ["c", f"d{ENTITY_LABEL_SEPARATOR}e"],
            },
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
    assert np.all(encoded[ENTITIES][0].features.toarray() == [1, 0, 1])


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


def test_single_state_featurizer_with_interpreter_state_with_action_listen(
    unpacked_trained_moodbot_path: Text,
):
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
    encoded = f.encode_state(
        {
            "user": {"text": "a ball", "intent": "b", "entities": ["c"]},
            "prev_action": {"action_text": "throw a ball"},
            "active_loop": {"name": "k"},
            "slots": {"e": (1.0,)},
        },
        interpreter=interpreter,
    )
    assert list(encoded.keys()) == [ACTION_TEXT, ACTIVE_LOOP, SLOTS]
    assert encoded[ACTION_TEXT][0].features.shape[-1] == 300
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[1, 0, 0]])).nnz == 0
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 0, 0, 1]])
    ).nnz == 0
