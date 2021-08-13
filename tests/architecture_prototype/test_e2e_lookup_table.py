import pytest
import copy
import numpy as np
import itertools
from typing import List, Text

from rasa.shared.nlu.training_data.features import Features
from rasa.architecture_prototype.graph_components import E2ELookupTable
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import INTENT, TEXT, ENTITIES, ACTION_NAME, ACTION_TEXT


def dummy_features(id: int, attribute: Text) -> Features:
    return Features(
        np.full(shape=(1), fill_value=id),
        attribute=attribute,
        feature_type="really-anything",
        origin="",
    )


@pytest.mark.parametrize(
    "wrong_attributes",
    [list(), ["other"]]
    + list(itertools.permutations(E2ELookupTable.KEY_ATTRIBUTES, 2)),
)
def test_add_with_wrong_key_attributes(wrong_attributes: List[Text]):

    sub_state = {attribute: "dummy" for attribute in wrong_attributes}
    with pytest.raises(ValueError, match="Expected exactly one attribute out of"):
        E2ELookupTable().add(Message(sub_state))


@pytest.mark.parametrize(
    "wrong_combination",
    [attribute for attribute in E2ELookupTable.KEY_ATTRIBUTES if attribute != TEXT],
)
def test_add_with_unexpected_entity_attribute(wrong_combination: Text):
    sub_state = {attribute: "dummy" for attribute in [ENTITIES, wrong_combination]}
    with pytest.raises(ValueError, match="Expected entities information"):
        E2ELookupTable().add(Message(sub_state))


def test_build_with_copies():

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
    lookup_table = E2ELookupTable(handle_collisions=False)
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


def test_build_with_inconsistencies_due_to_missing_annotations():

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
    lookup_table = E2ELookupTable(handle_collisions=True)
    for sub_state in sub_states:
        lookup_table.add(Message(sub_state))
    half_of_added_messages = int(len(sub_states) / 2)
    assert len(lookup_table) == half_of_added_messages
    for message in lookup_table.values():
        assert ARBITRARY_NON_KEY in message.data
    assert lookup_table.num_collisions_ignored == 0
    assert lookup_table.num_collisions_resolved == half_of_added_messages

    # ... and if we don't
    lookup_table2 = E2ELookupTable(handle_collisions=False)
    for idx in range(0, half_of_added_messages):
        lookup_table2.add(Message(sub_states[idx]))
    for idx in range(half_of_added_messages, 2 * half_of_added_messages):
        with pytest.raises(ValueError, match="Expected added message to be consistent"):
            lookup_table2.add(Message(sub_states[idx]))


def test_build_with_inconsistencies_due_to_missing_features():
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
    lookup_table = E2ELookupTable(handle_collisions=True)
    for message in messages:
        lookup_table.add(message)
    assert len(lookup_table) == len(unique_sub_states)
    for message in lookup_table.values():
        assert message.features is not None
        assert len(message.features) == 2
    assert lookup_table.num_collisions_ignored == 0
    assert lookup_table.num_collisions_resolved == len(unique_sub_states) * 2

    # ... and if we don't
    lookup_table2 = E2ELookupTable(handle_collisions=False)
    lookup_table2.add(messages[0])
    with pytest.raises(ValueError, match="Expected added message to be consistent"):
        lookup_table2.add(messages[1])


def test_build_with_inconsistent_data():
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
    lookup_table = E2ELookupTable(handle_collisions=True)
    for message in messages:
        lookup_table.add(message)
    assert len(lookup_table) == len(unique_sub_states)
    for message in lookup_table.values():
        assert message.features is not None
        assert len(message.features) == 1  # because item with features was added first
    assert lookup_table.num_collisions_ignored == len(unique_sub_states)

    # if we do not handle the collisions:
    lookup_table2 = E2ELookupTable(handle_collisions=False)
    for idx in range(0, len(messages) - 1, 2):
        lookup_table2.add(messages[idx])
        with pytest.raises(ValueError, match="Expected added message to be consistent"):
            lookup_table2.add(messages[idx + 1])


# TODO: one test where an inconsistency is ignored...
def test_build_ignores_some_inconsistencies():
    # This is a reminder that not everything inconsistency will be detected.
    # Similarly, differences in the actual Features (values/origin/...) will be ingored.
    ARBITRARY_NON_KEY = "arbitrary_non_key"
    sub_states = [
        {TEXT: "text", ENTITIES: "1"},
        {TEXT: "text", ENTITIES: "2"},
        {INTENT: "text", ARBITRARY_NON_KEY: "1"},
        {INTENT: "text", ARBITRARY_NON_KEY: "2"},
    ]
    lookup_table = E2ELookupTable(handle_collisions=False)
    for sub_state in sub_states:
        lookup_table.add(Message(sub_state))
    assert len(lookup_table) == int(len(sub_states) / 2)


def test_lookup_features():

    OTHER = "other"
    messages = [
        Message(data={TEXT: "A"}, features=[dummy_features(1, TEXT)]),
        Message(data={INTENT: "B", OTHER: "C"}, features=[dummy_features(2, OTHER)]),
        Message(data={TEXT: "A2"}, features=[dummy_features(3, TEXT)]),
        Message(data={INTENT: "B2", OTHER: "C2"}, features=[dummy_features(4, OTHER)]),
    ]

    table = E2ELookupTable()
    table.add_all(messages)

    sub_state = {TEXT: "A", INTENT: "B", OTHER: "C"}
    features = table.lookup_features(sub_state=sub_state)
    for attribute, id in [(TEXT, 1), (INTENT, None), (OTHER, 2)]:
        if id is not None:
            assert attribute in features
            assert len(features[attribute]) == 1
            assert id == features[attribute][0].features[0]
        else:
            # TODO: should this exist but be empty?
            assert attribute not in features

    features = table.lookup_features(sub_state, attributes=[OTHER])
    assert TEXT not in features
    assert INTENT not in features

    YET_ANOTHER = "another"
    features = table.lookup_features(sub_state, attributes=[YET_ANOTHER])
    assert YET_ANOTHER not in features

    with pytest.raises(ValueError, match="Unknown key"):
        table.lookup_features({TEXT: "A-unknwon"})


def test_lookup_features_if_lookup_table_is_broken():

    broken_table = E2ELookupTable()
    broken_table._table = {
        E2ELookupTable._build_key({TEXT: "A"}): Message(
            data={}, features=[dummy_features(2, TEXT)]
        ),
        E2ELookupTable._build_key({INTENT: "B"}): Message(
            data={}, features=[dummy_features(1, TEXT)]
        ),
    }
    with pytest.raises(
        RuntimeError, match=f"Feature for attribute {TEXT} has already been"
    ):
        broken_table.lookup_features({TEXT: "A", INTENT: "B"})

    # as a reminder that we don't fix every strange edgecase:
    not_broken_but_strange_table = E2ELookupTable()
    not_broken_but_strange_table._table = {
        E2ELookupTable._build_key({TEXT: "A"}): Message(data=dict()),
        E2ELookupTable._build_key({INTENT: "B"}): Message(
            data=dict(), features=[dummy_features(1, TEXT)]
        ),
    }
    features = not_broken_but_strange_table.lookup_features({TEXT: "A", INTENT: "B"})
    assert TEXT in features and len(features[TEXT]) == 1


def test_lookup_message():

    OTHER = "other"
    messages = [
        Message(data={TEXT: "A"}, features=[dummy_features(1, TEXT)]),
        Message(data={TEXT: "B"}),
        Message(data={TEXT: "B", OTHER: "C"}, features=[dummy_features(2, OTHER)]),
        Message(data={INTENT: "B"}),
        Message(data={ACTION_TEXT: "B"}),
        Message(data={ACTION_NAME: "B"}),
    ]

    table = E2ELookupTable(handle_collisions=True)
    table.add_all(messages)

    message = table.lookup_message(user_text="A")
    assert message
    message = table.lookup_message(user_text="B")
    assert OTHER in message.data.keys()
    with pytest.raises(ValueError, match=f"Expected a message with key \('{TEXT}"):
        table.lookup_message(user_text="not included")
