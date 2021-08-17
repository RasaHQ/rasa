from typing import Text
import numpy as np
import re
import scipy.sparse
import pytest

from rasa.shared.nlu.training_data.message import Message
from rasa.architecture_prototype.graph_components import (
    E2ELookupTable,
    StoryToTrainingDataConverter,
)
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.core.domain import Domain, State
from rasa.shared.nlu.constants import (
    ACTION_TEXT,
    ACTION_NAME,
    ENTITIES,
    TEXT,
    INTENT,
    FEATURE_TYPE_SEQUENCE,
    FEATURE_TYPE_SENTENCE,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_TAGS,
)
from rasa.shared.core.constants import (
    ACTIVE_LOOP,
    PREVIOUS_ACTION,
    SLOTS,
    ENTITY_LABEL_SEPARATOR,
    USER,
)
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.shared.core.slots import Slot
from rasa.shared.nlu.training_data.features import Features


def test_without_lookup__state_not_with_action_listen():
    """This test are for encoding state without a trained interpreter.
    action_name is not action_listen, so, INTENT, TEXT and ENTITIES should not be
    featurized.
    """
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1, "action_listen": 2}
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}

    state = {
        "user": {"intent": "a", "text": "blah blah blah"},
        "prev_action": {"action_name": "d", "action_text": "boom"},
        "active_loop": {"name": "i"},
        "slots": {"g": (1.0,)},
    }

    encoded = f.encode_state(state, e2e_features=None)

    # user input is ignored as prev action is not action_listen
    assert list(encoded.keys()) == [ACTION_NAME, ACTIVE_LOOP, SLOTS]
    assert (
        encoded[ACTION_NAME][0].features != scipy.sparse.coo_matrix([[0, 1, 0]])
    ).nnz == 0
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 1, 0, 0]])
    ).nnz == 0
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[0, 0, 1]])).nnz == 0


def test_without_lookup__state_with_action_listen():
    """This test are for encoding state without a trained interpreter.
    action_name is action_listen, so, INTENT and ENTITIES should be featurized
    while text shouldn't because we don't have an interpreter.
    """
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1, "action_listen": 2}
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}

    state = {
        "user": {"intent": "a", "text": "blah blah blah"},
        "prev_action": {"action_name": "action_listen", "action_text": "boom"},
        "active_loop": {"name": "k"},
        "slots": {"e": (1.0,)},
    }
    encoded = f.encode_state(state, e2e_features=None)

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


def test_without_lookup__state_no_intent_no_action_name():
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1, "action_listen": 2}
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}

    state = {
        "user": {"text": "blah blah blah"},
        "prev_action": {"action_text": "boom"},
        "active_loop": {"name": "k"},
        "slots": {"e": (1.0,)},
    }

    # check that no intent / action_name features are added when the interpreter
    # isn't there and
    # intent / action_name not in input
    encoded = f.encode_state(state, e2e_features=None)

    assert list(encoded.keys()) == [ACTIVE_LOOP, SLOTS]
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 0, 0, 1]])
    ).nnz == 0
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[1, 0, 0]])).nnz == 0


def test_with_lookup__correctly_fallsback_to_create_features():

    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1}

    state = {USER: {INTENT: "e"}, PREVIOUS_ACTION: {ACTION_NAME: "action_listen"}}

    # create a lookup table with all relevant entries but no Features
    lookup_table = E2ELookupTable()
    lookup_table.add(Message(data={INTENT: state[USER][INTENT]}))
    lookup_table.add(Message(data={ACTION_NAME: state[PREVIOUS_ACTION][ACTION_NAME]}))

    encoded = f.encode_state(state, e2e_features=lookup_table)

    assert list(encoded.keys()) == [INTENT, ACTION_NAME]
    assert (encoded[INTENT][0].features != scipy.sparse.coo_matrix([[0, 0]])).nnz == 0


def test_prepare_for_training():
    domain = Domain(
        intents=["greet"],
        entities=["name"],
        slots=[Slot("name")],
        responses={},
        forms=[],
        action_names=["utter_greet", "action_check_weather"],
    )

    f = SingleStateFeaturizer()
    f.prepare_for_training(domain)

    assert len(f._default_feature_states[INTENT]) > 1
    assert "greet" in f._default_feature_states[INTENT]
    assert len(f._default_feature_states[ENTITIES]) == 1
    assert f._default_feature_states[ENTITIES]["name"] == 0
    assert len(f._default_feature_states[SLOTS]) == 1
    assert f._default_feature_states[SLOTS]["name_0"] == 0
    assert len(f._default_feature_states[ACTION_NAME]) > 2
    assert "utter_greet" in f._default_feature_states[ACTION_NAME]
    assert "action_check_weather" in f._default_feature_states[ACTION_NAME]
    assert len(f._default_feature_states[ACTIVE_LOOP]) == 0


def test_creates_encoded_all_actions():
    domain = Domain(
        intents=[],
        entities=[],
        slots=[],
        responses={},
        forms={},
        action_names=["a", "b", "c", "d"],
    )

    f = SingleStateFeaturizer()
    f.prepare_for_training(domain)

    lookup_table = E2ELookupTable()
    lookup_table.derive_messages_from_domain_and_add(domain)

    encoded_actions = f.encode_all_actions(domain, e2e_features=lookup_table)

    assert len(encoded_actions) == len(domain.action_names_or_texts)
    assert all(
        [
            ACTION_NAME in encoded_action and ACTION_TEXT not in encoded_action
            for encoded_action in encoded_actions
        ]
    )


def test_single_state_featurizer_with_entity_roles_and_groups():

    # create fake message that has been tokenized and entities have been extracted
    text = "I am flying from London to Paris"
    tokens = [
        Token(text=match.group(), start=match.start())
        for match in re.finditer(r"\S+", text)
    ]
    entity_tags = ["city", f"city{ENTITY_LABEL_SEPARATOR}to"]
    entities = [
        {
            ENTITY_ATTRIBUTE_TYPE: entity_tags[0],
            ENTITY_ATTRIBUTE_VALUE: "London",
            ENTITY_ATTRIBUTE_START: 17,
            ENTITY_ATTRIBUTE_END: 23,
        },
        {
            ENTITY_ATTRIBUTE_TYPE: entity_tags[1],
            ENTITY_ATTRIBUTE_VALUE: "Paris",
            ENTITY_ATTRIBUTE_START: 27,
            ENTITY_ATTRIBUTE_END: 32,
        },
    ]
    message = Message({TEXT: text, TOKENS_NAMES[TEXT]: tokens, ENTITIES: entities,})

    # add message  to lookup table
    lookup_table = E2ELookupTable()
    lookup_table.add(message)

    # instantiate matching domain and single state featurizer
    domain = Domain(
        intents=[],
        entities=entity_tags,
        slots=[],
        responses={},
        forms={},
        action_names=[],
    )
    f = SingleStateFeaturizer()
    f.prepare_for_training(domain)

    # encode!
    encoded = f.encode_entities(
        entity_data={TEXT: text, ENTITIES: entities}, e2e_features=lookup_table,
    )

    # check
    assert len(f.entity_tag_specs) == 1
    tags_to_ids = f.entity_tag_specs[0].tags_to_ids
    for idx, entity_tag in enumerate(entity_tags):
        tags_to_ids[entity_tag] = idx + 1  # hence, city -> 1, city#to -> 2
    assert sorted(list(encoded.keys())) == [ENTITY_TAGS]
    assert np.all(
        encoded[ENTITY_TAGS][0].features == [[0], [0], [0], [0], [1], [0], [2]]
    )


def test_without_lookup__encode_entities_with_bilou_entity_roles_and_groups():

    # instantiate matching domain and single state featurizer
    domain = Domain(
        intents=[],
        entities=["city", f"city{ENTITY_LABEL_SEPARATOR}to"],
        slots=[],
        responses={},
        forms={},
        action_names=[],
    )
    f = SingleStateFeaturizer()
    f.prepare_for_training(domain, bilou_tagging=True)

    # encode without lookup table
    encoded = f.encode_entities(
        {
            TEXT: "I am flying from London to Paris",
            ENTITIES: [
                {
                    ENTITY_ATTRIBUTE_TYPE: "city",
                    ENTITY_ATTRIBUTE_VALUE: "London",
                    ENTITY_ATTRIBUTE_START: 17,
                    ENTITY_ATTRIBUTE_END: 23,
                },
                {
                    ENTITY_ATTRIBUTE_TYPE: f"city{ENTITY_LABEL_SEPARATOR}to",
                    ENTITY_ATTRIBUTE_VALUE: "Paris",
                    ENTITY_ATTRIBUTE_START: 27,
                    ENTITY_ATTRIBUTE_END: 32,
                },
            ],
        },
        e2e_features=None,
        bilou_tagging=True,
    )
    assert sorted(list(encoded.keys())) == sorted([ENTITY_TAGS])
    assert np.all(
        encoded[ENTITY_TAGS][0].features == [[0], [0], [0], [0], [4], [0], [8]]
    )

    # encode without lookup table
    encoded = f.encode_entities(
        {
            TEXT: "I am flying to Saint Petersburg",
            ENTITIES: [
                {
                    ENTITY_ATTRIBUTE_TYPE: "city",
                    ENTITY_ATTRIBUTE_VALUE: "Saint Petersburg",
                    ENTITY_ATTRIBUTE_START: 15,
                    ENTITY_ATTRIBUTE_END: 31,
                },
            ],
        },
        e2e_features=None,
        bilou_tagging=True,
    )
    assert sorted(list(encoded.keys())) == sorted([ENTITY_TAGS])
    assert np.all(encoded[ENTITY_TAGS][0].features == [[0], [0], [0], [0], [1], [3]])


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
        e2e_features=None,
    )

    assert encoded[ACTION_NAME][0].features.dtype == np.float32


"""
@pytest.mark.timeout(
    300, func_only=True
)  # these can take a longer time than the default timeout
def test_single_state_featurizer_with_interpreter_state_with_action_listen(
    unpacked_trained_spacybot_path: Text,
):
    interpreter = Agent.load(unpacked_trained_spacybot_path).interpreter

    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"greet": 0, "inform": 1}
    f._default_feature_states[ENTITIES] = {
        "city": 0,
        "name": 1,
        f"city{ENTITY_LABEL_SEPARATOR}to": 2,
        f"city{ENTITY_LABEL_SEPARATOR}from": 3,
    }
    f._default_feature_states[ACTION_NAME] = {
        "utter_ask_where_to": 0,
        "utter_greet": 1,
        "action_listen": 2,
    }
    # `_0` in slots represent feature dimension
    f._default_feature_states[SLOTS] = {"slot_1_0": 0, "slot_2_0": 1, "slot_3_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {
        "active_loop_1": 0,
        "active_loop_2": 1,
        "active_loop_3": 2,
        "active_loop_4": 3,
    }
    encoded = f.encode_state(
        {
            "user": {
                "text": "I am flying from London to Paris",
                "intent": "inform",
                "entities": ["city", f"city{ENTITY_LABEL_SEPARATOR}to"],
            },
            "prev_action": {
                "action_name": "action_listen",
                "action_text": "throw a ball",
            },
            "active_loop": {"name": "active_loop_4"},
            "slots": {"slot_1": (1.0,)},
        },
        interpreter=interpreter,
    )

    # check all the features are encoded and *_text features are encoded by a
    # dense featurizer
    assert sorted(list(encoded.keys())) == sorted(
        [TEXT, ENTITIES, ACTION_NAME, SLOTS, ACTIVE_LOOP, INTENT, ACTION_TEXT]
    )
    assert encoded[TEXT][0].features.shape[-1] == 300
    assert encoded[ACTION_TEXT][0].features.shape[-1] == 300
    assert (encoded[INTENT][0].features != scipy.sparse.coo_matrix([[0, 1]])).nnz == 0
    assert (
        encoded[ACTION_NAME][0].features != scipy.sparse.coo_matrix([[0, 0, 1]])
    ).nnz == 0
    assert encoded[ENTITIES][0].features.shape[-1] == 4
    assert (encoded[SLOTS][0].features != scipy.sparse.coo_matrix([[1, 0, 0]])).nnz == 0
    assert (
        encoded[ACTIVE_LOOP][0].features != scipy.sparse.coo_matrix([[0, 0, 0, 1]])
    ).nnz == 0


@pytest.mark.timeout(
    300, func_only=True
)  # these can take a longer time than the default timeout
def test_single_state_featurizer_with_interpreter_state_not_with_action_listen(
    unpacked_trained_spacybot_path: Text,
):
    # check that user features are ignored when action_name is not action_listen
    from rasa.core.agent import Agent

    interpreter = Agent.load(unpacked_trained_spacybot_path).interpreter
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


@pytest.mark.timeout(
    300, func_only=True
)  # these can take a longer time than the default timeout
def test_single_state_featurizer_with_interpreter_state_with_no_action_name(
    unpacked_trained_spacybot_path: Text,
):
    # check that action name features are not added by the featurizer when not
    # present in the state and
    # check user input is ignored when action is not action_listen
    # and action_name is features are not added
    from rasa.core.agent import Agent

    interpreter = Agent.load(unpacked_trained_spacybot_path).interpreter

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


def test_state_features_for_attribute_raises_on_not_supported_attribute():
    f = SingleStateFeaturizer()

    with pytest.raises(ValueError):
        f._state_features_for_attribute({}, "not-supported-attribute")


def test_to_sparse_sentence_features():
    features = [
        Features(
            scipy.sparse.csr_matrix(np.random.randint(5, size=(5, 10))),
            FEATURE_TYPE_SEQUENCE,
            TEXT,
            "some-featurizer",
        )
    ]

    sentence_features = SingleStateFeaturizer._to_sparse_sentence_features(features)

    assert len(sentence_features) == 1
    assert FEATURE_TYPE_SENTENCE == sentence_features[0].type
    assert features[0].origin == sentence_features[0].origin
    assert features[0].attribute == sentence_features[0].attribute
    assert sentence_features[0].features.shape == (1, 10)


@pytest.mark.timeout(
    300, func_only=True
)  # these can take a longer time than the default timeout
def test_single_state_featurizer_uses_regex_interpreter(
    unpacked_trained_spacybot_path: Text,
):
    from rasa.core.agent import Agent

    domain = Domain(
        intents=[], entities=[], slots=[], responses={}, forms=[], action_names=[],
    )
    f = SingleStateFeaturizer()
    # simulate that core was trained separately by passing
    # RegexInterpreter to prepare_for_training
    f.prepare_for_training(domain, RegexInterpreter())
    # simulate that nlu and core models were manually combined for prediction
    # by passing trained interpreter to encode_all_actions
    interpreter = Agent.load(unpacked_trained_spacybot_path).interpreter
    features = f._extract_state_features({TEXT: "some text"}, interpreter)
    # RegexInterpreter cannot create features for text, therefore since featurizer
    # was trained without nlu, features for text should be empty
    assert not features

"""
