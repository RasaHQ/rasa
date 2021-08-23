import pytest
import copy
import numpy as np
import itertools
from typing import List, Text, Optional, Dict


from rasa.core.featurizers.precomputation import (
    CoreFeaturizationCollector,
    CoreFeaturizationPrecomputations,
    CoreFeaturizationPreparer,
)
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import (
    INTENT,
    TEXT,
    ENTITIES,
    ACTION_NAME,
    ACTION_TEXT,
    INTENT_NAME_KEY,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_GROUP,
)
from rasa.shared.core.constants import DEFAULT_ACTION_NAMES
from rasa.shared.core.slots import Slot
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered, ActionExecuted
from rasa.shared.core.training_data.structures import StoryGraph, StoryStep
from rasa.shared.core.trackers import DialogueStateTracker


def _dummy_features(id: int, attribute: Text) -> Features:
    return Features(
        np.full(shape=(1), fill_value=id),
        attribute=attribute,
        feature_type="really-anything",
        origin="",
    )


@pytest.mark.parametrize(
    "wrong_attributes",
    [list(), ["other"]]
    + list(itertools.permutations(CoreFeaturizationPrecomputations.KEY_ATTRIBUTES, 2)),
)
def test_precomputation_add_with_wrong_key_attributes(wrong_attributes: List[Text]):
    sub_state = {attribute: "dummy" for attribute in wrong_attributes}
    with pytest.raises(ValueError, match="Expected exactly one attribute out of"):
        CoreFeaturizationPrecomputations().add(Message(sub_state))


@pytest.mark.parametrize(
    "wrong_combination",
    [
        attribute
        for attribute in CoreFeaturizationPrecomputations.KEY_ATTRIBUTES
        if attribute != TEXT
    ],
)
def test_precomputation_add_with_unexpected_entity_attribute(wrong_combination: Text):
    sub_state = {attribute: "dummy" for attribute in [ENTITIES, wrong_combination]}
    with pytest.raises(ValueError, match="Expected entities information"):
        CoreFeaturizationPrecomputations().add(Message(sub_state))


def test_precomputation_build_with_copies():

    # construct a set of unique substates and messages
    dummy_value = "this-could-be-anything"
    unique_sub_states = [
        {INTENT: "greet"},
        {TEXT: "text", ENTITIES: dummy_value},
        {ACTION_TEXT: "action_text"},
        {ACTION_NAME: "action_name"},
    ]
    unique_messages = [Message(sub_state) for sub_state in unique_sub_states]

    # add some copies
    num_duplicates = 3
    messages = unique_messages * num_duplicates

    # build table
    lookup_table = CoreFeaturizationPrecomputations(handle_collisions=False)
    for message in messages:
        lookup_table.add(message)

    assert len(lookup_table) == len(unique_sub_states)
    # instead of freezing dicts, building sets, and comparing them, just check ...
    for message in lookup_table.values():
        assert message in unique_messages
    for message in unique_messages:
        assert message in lookup_table.values()

    assert lookup_table.num_collisions_ignored == len(unique_sub_states) * 2
    assert lookup_table.num_collisions_resolved == 0


def test_precomputation_build_with_inconsistencies_due_to_missing_annotations():

    # construct a set of unique substates
    dummy_value = "this-could-be-anything"
    unique_sub_states = [
        {INTENT: "greet"},
        {TEXT: "text", ENTITIES: dummy_value},
        {ACTION_TEXT: "action_text"},
        {ACTION_NAME: "action_name"},
    ]

    # add some substats with an additional attribute
    ARBITRARY_NON_KEY = "arbitrary_non_key"
    sub_states = list(unique_sub_states)  # copy :)
    for sub_state in unique_sub_states:
        sub_state_copy = dict(sub_state)
        sub_state_copy[ARBITRARY_NON_KEY] = "anything"
        sub_states.append(sub_state_copy)

    # if we handle the collisions...
    lookup_table = CoreFeaturizationPrecomputations(handle_collisions=True)
    for sub_state in sub_states:
        lookup_table.add(Message(sub_state))
    half_of_added_messages = int(len(sub_states) / 2)
    assert len(lookup_table) == half_of_added_messages
    for message in lookup_table.values():
        assert ARBITRARY_NON_KEY in message.data
    assert lookup_table.num_collisions_ignored == 0
    assert lookup_table.num_collisions_resolved == half_of_added_messages

    # ... and if we don't
    lookup_table2 = CoreFeaturizationPrecomputations(handle_collisions=False)
    for idx in range(0, half_of_added_messages):
        lookup_table2.add(Message(sub_states[idx]))
    for idx in range(half_of_added_messages, 2 * half_of_added_messages):
        with pytest.raises(ValueError, match="Expected added message to be consistent"):
            lookup_table2.add(Message(sub_states[idx]))


def test_precomputation_build_with_inconsistencies_due_to_missing_features():
    # This really should not happen, but if it did...

    # construct a set of unique substates and messages
    dummy_value = "this-could-be-anything"
    unique_sub_states = [
        {INTENT: "greet"},
        {TEXT: "text", ENTITIES: dummy_value},
        {ACTION_TEXT: "action_text"},
        {ACTION_NAME: "action_name"},
    ]

    # add different numbers of features
    dummy_feature = Features(
        np.ndarray((1, 2)),
        attribute="this-is-never-checked-and-can-be-anything",
        feature_type="really-anything",
        origin="",
    )
    messages = []
    for sub_state in unique_sub_states:
        messages.append(Message(data=sub_state, features=[]))
        messages.append(Message(data=sub_state, features=[dummy_feature]))
        messages.append(
            Message(data=sub_state, features=[dummy_feature, dummy_feature])
        )

    # if we do collision handling...
    lookup_table = CoreFeaturizationPrecomputations(handle_collisions=True)
    for message in messages:
        lookup_table.add(message)
    assert len(lookup_table) == len(unique_sub_states)
    for message in lookup_table.values():
        assert message.features is not None
        assert len(message.features) == 2
    assert lookup_table.num_collisions_ignored == 0
    assert lookup_table.num_collisions_resolved == len(unique_sub_states) * 2

    # ... and if we don't
    lookup_table2 = CoreFeaturizationPrecomputations(handle_collisions=False)
    lookup_table2.add(messages[0])
    with pytest.raises(ValueError, match="Expected added message to be consistent"):
        lookup_table2.add(messages[1])


def test_precomputation_build_with_inconsistent_data():
    """Tests whether the lookup table can deal not-too messy training data.
    That is, we assume that someone missed to annotate `ENTITIES`
    """

    dummy_value = "this-could-be-anything"
    unique_sub_states = [
        {INTENT: "greet"},
        {TEXT: "text", ENTITIES: dummy_value},
        {ACTION_TEXT: "action_text"},
        {ACTION_NAME: "action_name"},
    ]

    messages = []
    ARBITRARY_NON_KEY = "arbitrary_non_key"
    dummy_feature = Features(
        np.ndarray((1, 2)),
        attribute="this-is-never-checked-and-can-be-anything",
        feature_type="really-anything",
        origin="",
    )
    for sub_state in unique_sub_states:
        # one message with a feature...
        messages.append(Message(data=sub_state, features=[dummy_feature]))
        # ... and one with extra attribute instead
        sub_state_copy = copy.deepcopy(sub_state)
        sub_state_copy[ARBITRARY_NON_KEY] = dummy_value
        messages.append(Message(data=sub_state, features=[]))

    # if we handle the collisions:
    lookup_table = CoreFeaturizationPrecomputations(handle_collisions=True)
    for message in messages:
        lookup_table.add(message)
    assert len(lookup_table) == len(unique_sub_states)
    for message in lookup_table.values():
        assert message.features is not None
        assert len(message.features) == 1  # because item with features was added first
    assert lookup_table.num_collisions_ignored == len(unique_sub_states)

    # if we do not handle the collisions:
    lookup_table2 = CoreFeaturizationPrecomputations(handle_collisions=False)
    for idx in range(0, len(messages) - 1, 2):
        lookup_table2.add(messages[idx])
        with pytest.raises(ValueError, match="Expected added message to be consistent"):
            lookup_table2.add(messages[idx + 1])


# TODO: one test where an inconsistency is ignored...
def test_precomputation_build_ignores_some_inconsistencies():
    # This is a reminder that not everything inconsistency will be detected.
    # Similarly, differences in the actual Features (values/origin/...) will be ingored.
    ARBITRARY_NON_KEY = "arbitrary_non_key"
    sub_states = [
        {TEXT: "text", ENTITIES: "1"},
        {TEXT: "text", ENTITIES: "2"},
        {INTENT: "text", ARBITRARY_NON_KEY: "1"},
        {INTENT: "text", ARBITRARY_NON_KEY: "2"},
    ]
    lookup_table = CoreFeaturizationPrecomputations(handle_collisions=False)
    for sub_state in sub_states:
        lookup_table.add(Message(sub_state))
    assert len(lookup_table) == int(len(sub_states) / 2)


def test_precomputation_lookup_features():

    OTHER = "other"
    messages = [
        Message(data={TEXT: "A"}, features=[_dummy_features(1, TEXT)]),
        Message(data={INTENT: "B", OTHER: "C"}, features=[_dummy_features(2, OTHER)]),
        Message(data={TEXT: "A2"}, features=[_dummy_features(3, TEXT)]),
        Message(data={INTENT: "B2", OTHER: "C2"}, features=[_dummy_features(4, OTHER)]),
    ]

    table = CoreFeaturizationPrecomputations()
    table.add_all(messages)

    # If we don't specify a list of attributes, the resulting features dictionary will
    # only contain those attributes for which there are features.
    sub_state = {TEXT: "A", INTENT: "B", OTHER: "C"}
    features = table.lookup_features(sub_state=sub_state)
    for attribute, feature_value in [(TEXT, 1), (INTENT, None), (OTHER, 2)]:
        if feature_value is not None:
            assert attribute in features
            assert len(features[attribute]) == 1
            assert feature_value == features[attribute][0].features[0]
        else:
            assert attribute not in features

    # If we query features for `INTENT`, then a key will be there, even if there are
    # no feaetures
    features = table.lookup_features(
        sub_state=sub_state, attributes=list(sub_state.keys())
    )
    assert INTENT in features
    assert len(features[INTENT]) == 0

    # We only get the list of features we want...
    features = table.lookup_features(sub_state, attributes=[OTHER])
    assert TEXT not in features
    assert INTENT not in features
    assert len(features[OTHER]) == 1

    # ... even if there are no features:
    YET_ANOTHER = "another"
    features = table.lookup_features(sub_state, attributes=[YET_ANOTHER])
    assert len(features[YET_ANOTHER]) == 0

    # And we cannot lookup features for sub states that have no key attribute
    with pytest.raises(ValueError, match="Unknown key"):
        table.lookup_features({TEXT: "A-unknwon"})


def test_precomputation_lookup_features_if_lookup_table_is_broken():

    broken_table = CoreFeaturizationPrecomputations()
    broken_table._table = {
        CoreFeaturizationPrecomputations._build_key({TEXT: "A"}): Message(
            data={}, features=[_dummy_features(2, TEXT)]
        ),
        CoreFeaturizationPrecomputations._build_key({INTENT: "B"}): Message(
            data={}, features=[_dummy_features(1, TEXT)]
        ),
    }
    with pytest.raises(
        RuntimeError, match=f"Feature for attribute {TEXT} has already been"
    ):
        broken_table.lookup_features({TEXT: "A", INTENT: "B"})

    # as a reminder that we don't fix every strange edgecase:
    not_broken_but_strange_table = CoreFeaturizationPrecomputations()
    not_broken_but_strange_table._table = {
        CoreFeaturizationPrecomputations._build_key({TEXT: "A"}): Message(data=dict()),
        CoreFeaturizationPrecomputations._build_key({INTENT: "B"}): Message(
            data=dict(), features=[_dummy_features(1, TEXT)]
        ),
    }
    features = not_broken_but_strange_table.lookup_features({TEXT: "A", INTENT: "B"})
    assert TEXT in features and len(features[TEXT]) == 1


def test_precomputation_lookup_message():

    OTHER = "other"
    messages = [
        Message(data={TEXT: "A"}, features=[_dummy_features(1, TEXT)]),
        Message(data={TEXT: "B"}),
        Message(data={TEXT: "B", OTHER: "C"}, features=[_dummy_features(2, OTHER)]),
        Message(data={INTENT: "B"}),
        Message(data={ACTION_TEXT: "B"}),
        Message(data={ACTION_NAME: "B"}),
    ]

    table = CoreFeaturizationPrecomputations(handle_collisions=True)
    table.add_all(messages)

    message = table.lookup_message(user_text="A")
    assert message
    message = table.lookup_message(user_text="B")
    assert OTHER in message.data.keys()
    with pytest.raises(ValueError, match=f"Expected a message with key \\('{TEXT}"):
        table.lookup_message(user_text="not included")


def test_precomputation_derive_messages_from_events_and_add():
    events = [
        UserUttered(intent={INTENT_NAME_KEY: "greet"}),
        ActionExecuted(action_name="utter_greet"),
        ActionExecuted(action_name="utter_greet"),
    ]
    lookup_table = CoreFeaturizationPrecomputations(handle_collisions=False)
    lookup_table.derive_messages_from_events_and_add(events)
    # should create: one intent-only user substate and one prev_action substate
    assert len(lookup_table) == 2

    events = [
        UserUttered(text="text", intent={INTENT_NAME_KEY: "greet"}),
        ActionExecuted(action_name="utter_greet"),
    ]
    lookup_table = CoreFeaturizationPrecomputations(handle_collisions=False)
    lookup_table.derive_messages_from_events_and_add(events)
    # should create: one intent-only user substate, one text-only user substate one
    # prev_action substate
    assert len(lookup_table) == 3


def test_precomputation_derive_messages_from_domain_and_add():
    action_names = ["a", "b"]
    # action texts, response keys, forms, and action_names must be unique or the
    # domain will complain about it
    action_texts = ["a2", "b2"]
    responses = {"a3": "a2", "b3": "b2"}
    forms = {"a4": "a4"}
    # however, intent names can be anything
    intents = ["a", "b"]
    domain = Domain(
        intents=intents,
        action_names=action_names,
        action_texts=action_texts,
        responses=responses,
        entities=["e_a", "e_b", "e_c"],
        slots=[Slot(name="s")],
        forms=forms,
    )
    lookup_table = CoreFeaturizationPrecomputations(handle_collisions=True)
    lookup_table.derive_messages_from_domain_and_add(domain)
    # Note that we cannot just sum the above because the `domain` will e.g. combine the
    # `action_texts` with the `responses` to obtain the `domain.action_texts`
    assert len(lookup_table) == (
        len(domain.intent_properties)
        + len(domain.user_actions)
        + len(domain.action_texts)
        + len(DEFAULT_ACTION_NAMES)
        + len(domain.form_names)
    )


def _create_entity(
    value: Text, type: Text, role: Optional[Text] = None, group: Optional[Text] = None
) -> Dict[Text, Text]:
    entity = {}
    entity[ENTITY_ATTRIBUTE_VALUE] = value
    entity[ENTITY_ATTRIBUTE_TYPE] = type
    entity[ENTITY_ATTRIBUTE_ROLE] = role
    entity[ENTITY_ATTRIBUTE_GROUP] = group
    return entity


def test_preparation_for_training():
    """Tests whether story graph and domain information ends up in training data.
    Note: We do not test for completeness/shape of the result here. These aspects are
    defined by the same principles as the lookup table. Hence, these aspects should
    be tested in the lookup table tests, whereas here we just probe whether the lookup
    table is applied correctly.
    """
    # create domain and story graph
    intent_in_domain_only = "only-appears-in-domain"
    event_text_prefix = "only-appears-in-events"
    domain = Domain(
        intents=["greet", "inform", intent_in_domain_only],
        entities=["entity_name"],
        slots=[],
        responses=dict(),
        action_names=["action_listen", "utter_greet"],
        forms=dict(),
        action_texts=["Hi how are you?"],
    )
    text_that_contains_entities = f"{event_text_prefix}-1"
    events = [
        ActionExecuted(action_name="action_listen"),
        UserUttered(
            text=text_that_contains_entities,
            intent={"intent_name": "greet"},
            entities=[_create_entity(value="Bot", type="entity_name")],
        ),
        ActionExecuted(action_name="utter_greet", action_text="Hi how are you?"),
        ActionExecuted(action_name="action_listen"),
        UserUttered(text=f"{event_text_prefix}-2", intent={"intent_name": "inform"}),
        ActionExecuted(action_name="action_listen"),
    ]
    story_graph = StoryGraph([StoryStep(events=events)])

    # convert!
    training_data = CoreFeaturizationPreparer.prepare_for_training(
        domain=domain, story_graph=story_graph
    )

    # check information from events is contained - by checking the TEXTs
    training_texts = sorted(
        message.data.get(TEXT)
        for message in training_data.training_examples
        if TEXT in message.data
    )
    assert all(
        [
            (
                text.startswith(event_text_prefix)
                or text == CoreFeaturizationPreparer.WORKAROUND_TEXT
            )
            for text in training_texts
        ]
    )
    assert CoreFeaturizationPreparer.WORKAROUND_TEXT in training_texts
    # check that entities are accessible via the TEXT
    messages = [
        message
        for message in training_data.training_examples
        if message.data.get(TEXT) == text_that_contains_entities
    ]
    assert len(messages) == 1
    message = messages[0]
    assert message.data.get(ENTITIES)
    # check information from domain is contained - by checking the INTENTs
    training_intents = set(
        message.get(INTENT) for message in training_data.training_examples
    )
    assert intent_in_domain_only in training_intents


def test_preparation_for_inference():
    """Tests whether information from tracker events end up in the messages.
    Note: We do not test for completeness/shape of the result here. These aspects are
    defined by the same principles as the lookup table. Hence, these aspects should
    be tested in the lookup table tests, whereas here we just probe whether the lookup
    table is applied correctly.
    """
    # create tracker
    event_text_prefix = "this-is-a-text-prefix"
    text_that_contains_entities = f"{event_text_prefix}-1"
    events = [
        UserUttered(
            text=text_that_contains_entities,
            intent={"intent_name": "greet"},
            entities=[_create_entity(value="Bot", type="entity_name")],
        ),
        ActionExecuted(action_name="utter_greet", action_text="Hi how are you?"),
        ActionExecuted(action_name="action_listen"),
        UserUttered(text=f"{event_text_prefix}-2", intent={"intent_name": "inform"}),
    ]
    tracker = DialogueStateTracker.from_events(sender_id="arbitrary", evts=events)

    # convert!
    messages = CoreFeaturizationPreparer.prepare_for_inference(tracker)

    # check information from events is contained - by checking the TEXTs
    message_texts = sorted(
        message.data.get(TEXT) for message in messages if TEXT in message.data
    )
    assert all([(text.startswith(event_text_prefix)) for text in message_texts])
    assert CoreFeaturizationPreparer.WORKAROUND_TEXT not in message_texts
    # check that ENTITIES are accessible via the TEXT
    messages_for_chosen_text = [
        message
        for message in messages
        if message.data.get(TEXT) == text_that_contains_entities
    ]
    assert len(messages_for_chosen_text) == 1
    message = messages_for_chosen_text[0]
    assert message.data.get(ENTITIES)


@pytest.mark.parametrize(
    "messages_with_unique_lookup_key",
    [
        [
            Message(data={TEXT: "A"}, features=[_dummy_features(1, TEXT)]),
            Message(data={ACTION_TEXT: "B"}),
        ],
        [],
    ],
)
def test_collection(messages_with_unique_lookup_key: List[Message]):

    messages = messages_with_unique_lookup_key

    # pass as training data
    training_data = TrainingData(training_examples=messages)
    precomputations = CoreFeaturizationCollector.collect(training_data)
    assert len(precomputations) == len(messages)

    # pass the list of messages directly
    precomputations = CoreFeaturizationCollector.collect(messages)
    assert len(precomputations) == len(messages)


def test_collection_fails():
    """The collection expects messages that have a unique lookup key.

    This is because they (should) have been passed through the preparation stage which
    will have constructed messages with this property.
    """
    messages = [
        Message(data={TEXT: "A", ACTION_TEXT: "B"}, features=[_dummy_features(1, TEXT)])
    ]
    training_data = TrainingData(training_examples=messages)

    with pytest.raises(ValueError):
        CoreFeaturizationCollector.collect(training_data)
    with pytest.raises(ValueError):
        CoreFeaturizationCollector.collect(messages)
