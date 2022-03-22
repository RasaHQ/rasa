from typing import Text
import numpy as np
import re
import scipy.sparse
import pytest


from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.constants import TOKENS_NAMES
from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
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
from rasa.shared.core.domain import Domain
from rasa.shared.core.slots import TextSlot
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTIVE_LOOP,
    PREVIOUS_ACTION,
    SLOTS,
    ENTITY_LABEL_SEPARATOR,
    USER,
)
from rasa.utils.tensorflow.constants import SENTENCE, SEQUENCE


#
# internals
#


def test_state_features_for_attribute__raises_on_not_supported_attribute():
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


def test_create_features__dtype_float():
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"e": 0, "d": 1}
    f._default_feature_states[ENTITIES] = {"c": 0}

    encoded = f._create_features({ACTION_NAME: "d"}, attribute=ACTION_NAME)
    assert len(encoded) == 1  # cause for some reason this is a list
    assert encoded[0].features.dtype == np.float32


#
# preparation
#


def test_prepare_for_training():
    domain = Domain(
        intents=["greet"],
        entities=["name"],
        slots=[TextSlot("name", mappings=[{}])],
        responses={},
        forms={},
        action_names=["utter_greet", "action_check_weather"],
        data={},
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


#
# encode actions
# (always needs lookup table build from domain)
#


def test_encode_all_labels__encoded_all_action_names_and_texts():
    # ... where "labels" means actions...
    domain = Domain(
        intents=[],
        entities=[],
        slots=[],
        responses={},
        forms={},
        action_names=["a", "b", "c", "d"],
        data={},
    )

    f = SingleStateFeaturizer()
    f.prepare_for_training(domain)

    precomputations = MessageContainerForCoreFeaturization()
    precomputations.derive_messages_from_domain_and_add(domain)

    encoded_actions = f.encode_all_labels(domain, precomputations=precomputations)

    assert len(encoded_actions) == len(domain.action_names_or_texts)
    assert all(
        [
            ACTION_NAME in encoded_action and ACTION_TEXT not in encoded_action
            for encoded_action in encoded_actions
        ]
    )


#
# encode state withOUT lookup table
#


def sparse_equals_dense(
    sparse_matrix: scipy.sparse.spmatrix, dense_matrix: np.ndarray
) -> bool:
    return np.all(sparse_matrix.todense() == dense_matrix)


@pytest.mark.parametrize("action_name", [None, "NOT_action_listen", ACTION_LISTEN_NAME])
def test_encode_state__without_lookup(action_name: Text):
    """Tests that `encode_state` creates features for every attribute.
    In particular, that this is done even when there is no lookup table.
    If there is no action_listen in the  state, then no features should be created for
    the user sub-state.
    """
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {
        "c": 0,
        "d": 1,
        "NOT_action_listen": 2,
        ACTION_LISTEN_NAME: 3,
    }
    f._default_feature_states[SLOTS] = {"e_0": 0, "f_0": 1, "g_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {"h": 0, "i": 1, "j": 2, "k": 3}

    state = {
        USER: {INTENT: "a", TEXT: "blah blah blah"},
        PREVIOUS_ACTION: {ACTION_TEXT: "boom"},
        ACTIVE_LOOP: {"name": "i"},
        SLOTS: {"g": (1.0,)},
    }
    if action_name is not None:
        state[PREVIOUS_ACTION][ACTION_NAME] = action_name

    encoded = f.encode_state(state, precomputations=None)

    # this differs depending on whether action name is ACTION_LISTEN_NAME or "d"
    expected_attributes = [ACTIVE_LOOP, SLOTS]
    if action_name == ACTION_LISTEN_NAME:
        expected_attributes += [INTENT]
    if action_name is not None:
        expected_attributes += [ACTION_NAME]
    assert set(encoded.keys()) == set(expected_attributes)

    # the encoding of action_name of course depends on the sub-state
    if action_name is not None:
        if action_name == "NOT_action_listen":
            action_name_encoding = [0, 0, 1, 0]
        else:
            action_name_encoding = [0, 0, 0, 1]
        assert sparse_equals_dense(
            encoded[ACTION_NAME][0].features, np.array([action_name_encoding])
        )

    # the intent / user substate is only featurized if action_listen is
    # with_action_listen
    if action_name == ACTION_LISTEN_NAME:
        assert sparse_equals_dense(encoded[INTENT][0].features, np.array([[1, 0]]))

    # this is always the same
    assert sparse_equals_dense(
        encoded[ACTIVE_LOOP][0].features, np.array([[0, 1, 0, 0]])
    )
    assert sparse_equals_dense(encoded[SLOTS][0].features, np.array([[0, 0, 1]]))


#
# encode state WITH lookup table
#


def dummy_features(
    fill_value: int, units: int, attribute: Text, type: Text, is_sparse: bool
) -> Features:
    """Create some dummy `Features` with the desired properties."""
    matrix = np.full(shape=(1, units), fill_value=fill_value)
    if is_sparse:
        matrix = scipy.sparse.coo_matrix(matrix)
    return Features(
        features=matrix, attribute=attribute, feature_type=type, origin="whatever"
    )


@pytest.mark.parametrize("with_action_listen", [True, False])
def test_encode_state__with_lookup__creates_features_for_intent_and_action_name(
    with_action_listen: bool,
):
    """Tests that features for intent and action name are created if needed.
    Especially tests that this is the case even though no features are present in the
    given lookup table for this intent and action_name.
    However, if no `action_listen` is in the given sub-state, then the user sub-state
    should not be featurized (hence, no features for intent) should be created.
    """

    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"a": 0, "b": 1}
    f._default_feature_states[ACTION_NAME] = {"c": 0, "d": 1, ACTION_LISTEN_NAME: 2}

    # create state
    action_name = ACTION_LISTEN_NAME if with_action_listen else "c"
    state = {USER: {INTENT: "e"}, PREVIOUS_ACTION: {ACTION_NAME: action_name}}

    # create a lookup table with all relevant entries **but no Features**
    precomputations = MessageContainerForCoreFeaturization()
    precomputations.add(Message(data={INTENT: state[USER][INTENT]}))
    precomputations.add(
        Message(data={ACTION_NAME: state[PREVIOUS_ACTION][ACTION_NAME]})
    )

    # encode!
    encoded = f.encode_state(state, precomputations=precomputations)

    if with_action_listen:
        assert set(encoded.keys()) == set([INTENT, ACTION_NAME])
        assert (
            encoded[INTENT][0].features != scipy.sparse.coo_matrix([[0, 0]])
        ).nnz == 0
    else:
        assert set(encoded.keys()) == set([ACTION_NAME])


@pytest.mark.parametrize("action_name", [None, "NOT_action_listen", ACTION_LISTEN_NAME])
def test_encode_state__with_lookup__looksup_or_creates_features(action_name: Text):
    """Tests that features from table are combined or created from scratch.
    If the given action name is ...
    - ACTION_LISTEN_NAME then the user substate and the action name are encoded
    - some "other" action, then the user-substate is not encoed but the action name is
    - set to "None", then we remove the action name from the user substate and as a
      result there should be no encoding for the action name and for the user substate
    """
    f = SingleStateFeaturizer()
    f._default_feature_states[INTENT] = {"greet": 0, "inform": 1}
    f._default_feature_states[ENTITIES] = {
        "city": 0,
        "name": 1,
        f"city{ENTITY_LABEL_SEPARATOR}to": 2,
        f"city{ENTITY_LABEL_SEPARATOR}from": 3,
    }
    f._default_feature_states[ACTION_NAME] = {
        "NOT_action_listen": 0,
        "utter_greet": 1,
        ACTION_LISTEN_NAME: 2,
    }
    # `_0` in slots represent feature dimension
    f._default_feature_states[SLOTS] = {"slot_1_0": 0, "slot_2_0": 1, "slot_3_0": 2}
    f._default_feature_states[ACTIVE_LOOP] = {
        "active_loop_1": 0,
        "active_loop_2": 1,
        "active_loop_3": 2,
        "active_loop_4": 3,
    }

    # create state
    text = "I am flying from London to Paris"
    tokens = [
        Token(text=match.group(), start=match.start())
        for match in re.finditer(r"\S+", text)
    ]
    entity_name_list = ["city", f"city{ENTITY_LABEL_SEPARATOR}to"]
    action_text = "throw a ball"
    intent = "inform"
    state = {
        USER: {TEXT: text, INTENT: intent, ENTITIES: entity_name_list},
        PREVIOUS_ACTION: {ACTION_NAME: action_name, ACTION_TEXT: action_text},
        ACTIVE_LOOP: {"name": "active_loop_4"},
        SLOTS: {"slot_1": (1.0,)},
    }
    if action_name is None:
        del state[PREVIOUS_ACTION][ACTION_NAME]

    # Build lookup table with all relevant information - and dummy features for all
    # dense featurizable attributes.
    # Note that we don't need to add the `ENTITIES` to the message including `TEXT`
    # here because `encode_state` won't featurize the entities using the lookup table
    # (only `encode_entities` does that).
    units = 300
    precomputations = MessageContainerForCoreFeaturization()
    precomputations.add_all(
        [
            Message(
                data={TEXT: text, TOKENS_NAMES[TEXT]: tokens},
                features=[
                    dummy_features(
                        fill_value=11,
                        units=units,
                        attribute=TEXT,
                        type=SENTENCE,
                        is_sparse=True,
                    ),
                    dummy_features(
                        fill_value=12,
                        units=units,
                        attribute=TEXT,
                        type=SEQUENCE,
                        is_sparse=False,
                    ),
                    # Note: sparse sequence feature is last here
                    dummy_features(
                        fill_value=13,
                        units=units,
                        attribute=TEXT,
                        type=SEQUENCE,
                        is_sparse=True,
                    ),
                ],
            ),
            Message(data={INTENT: intent}),
            Message(
                data={ACTION_TEXT: action_text},
                features=[
                    dummy_features(
                        fill_value=1,
                        units=units,
                        attribute=ACTION_TEXT,
                        type=SEQUENCE,
                        is_sparse=True,
                    )
                ],
            ),
        ]
    )
    if action_name is not None:
        precomputations.add(Message(data={ACTION_NAME: action_name}))

    # encode the state
    encoded = f.encode_state(state, precomputations=precomputations)

    # check all the features are encoded and *_text features are encoded by a
    # dense featurizer
    expected_attributes = [SLOTS, ACTIVE_LOOP, ACTION_TEXT]
    if action_name is not None:  # i.e. we did not remove it from the state
        expected_attributes += [ACTION_NAME]
    if action_name == ACTION_LISTEN_NAME:
        expected_attributes += [TEXT, ENTITIES, INTENT]
    assert set(encoded.keys()) == set(expected_attributes)

    # Remember, sparse sequence features come first (and `.features` denotes the matrix
    # not a `Features` object)
    if action_name == ACTION_LISTEN_NAME:
        assert encoded[TEXT][0].features.shape[-1] == units
        assert encoded[TEXT][0].is_sparse()
        assert encoded[ENTITIES][0].features.shape[-1] == 4
        assert sparse_equals_dense(encoded[INTENT][0].features, np.array([[0, 1]]))
    assert encoded[ACTION_TEXT][0].features.shape[-1] == units
    assert encoded[ACTION_TEXT][0].is_sparse()
    if action_name is not None:
        if action_name == "NOT_action_listen":
            action_name_encoding = [1, 0, 0]
        else:  # action_listen
            action_name_encoding = [0, 0, 1]
        assert sparse_equals_dense(
            encoded[ACTION_NAME][0].features, np.array([action_name_encoding])
        )
    else:
        assert ACTION_NAME not in encoded
    assert sparse_equals_dense(encoded[SLOTS][0].features, np.array([[1, 0, 0]]))
    assert sparse_equals_dense(
        encoded[ACTIVE_LOOP][0].features, np.array([[0, 0, 0, 1]])
    )


#
# encode entities
# (always needs lookup table build from tokenized messages incl. entities)
#


def test_encode_entities__with_entity_roles_and_groups():

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
    message = Message({TEXT: text, TOKENS_NAMES[TEXT]: tokens, ENTITIES: entities})

    # create a lookup table that has seen this message
    precomputations = MessageContainerForCoreFeaturization()
    precomputations.add(message)

    # instantiate matching domain and single state featurizer
    domain = Domain(
        intents=[],
        entities=entity_tags,
        slots=[],
        responses={},
        forms={},
        action_names=[],
        data={},
    )
    f = SingleStateFeaturizer()
    f.prepare_for_training(domain)

    # encode!
    encoded = f.encode_entities(
        entity_data={TEXT: text, ENTITIES: entities}, precomputations=precomputations
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


def test_encode_entities__with_bilou_entity_roles_and_groups():

    # Instantiate domain and configure the single state featurizer for this domain.
    # Note that there are 2 entity tags here.
    entity_tags = ["city", f"city{ENTITY_LABEL_SEPARATOR}to"]
    domain = Domain(
        intents=[],
        entities=entity_tags,
        slots=[],
        responses={},
        forms={},
        action_names=[],
        data={},
    )
    f = SingleStateFeaturizer()
    f.prepare_for_training(domain, bilou_tagging=True)

    # (1) example with both entities

    # create message that has been tokenized and where entities have been extracted
    text = "I am flying from London to Paris"
    tokens = [
        Token(text=match.group(), start=match.start())
        for match in re.finditer(r"\S+", text)
    ]
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
    message = Message({TEXT: text, TOKENS_NAMES[TEXT]: tokens, ENTITIES: entities})

    # create a lookup table that has seen this message
    precomputations = MessageContainerForCoreFeaturization()
    precomputations.add(message)

    # encode!
    encoded = f.encode_entities(
        {TEXT: text, ENTITIES: entities},
        precomputations=precomputations,
        bilou_tagging=True,
    )
    assert sorted(list(encoded.keys())) == sorted([ENTITY_TAGS])
    assert np.all(
        encoded[ENTITY_TAGS][0].features == [[0], [0], [0], [0], [4], [0], [8]]
    )

    # (2) example with only the "city" entity

    # create message that has been tokenized and where entities have been extracted
    text = "I am flying to Saint Petersburg"
    tokens = [
        Token(text=match.group(), start=match.start())
        for match in re.finditer(r"\S+", text)
    ]
    entities = [
        {
            ENTITY_ATTRIBUTE_TYPE: "city",
            ENTITY_ATTRIBUTE_VALUE: "Saint Petersburg",
            ENTITY_ATTRIBUTE_START: 15,
            ENTITY_ATTRIBUTE_END: 31,
        }
    ]
    message = Message({TEXT: text, TOKENS_NAMES[TEXT]: tokens, ENTITIES: entities})

    # create a lookup table that has seen this message
    precomputations = MessageContainerForCoreFeaturization()
    precomputations.add(message)

    # encode!
    encoded = f.encode_entities(
        {TEXT: text, ENTITIES: entities},
        precomputations=precomputations,
        bilou_tagging=True,
    )
    assert sorted(list(encoded.keys())) == sorted([ENTITY_TAGS])
    assert np.all(encoded[ENTITY_TAGS][0].features == [[0], [0], [0], [0], [1], [3]])
