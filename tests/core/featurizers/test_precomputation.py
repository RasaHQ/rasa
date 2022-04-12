import pytest
import numpy as np
import itertools
from typing import List, Text, Optional, Dict


from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.core.featurizers.precomputation import (
    CoreFeaturizationCollector,
    MessageContainerForCoreFeaturization,
    CoreFeaturizationInputConverter,
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
from rasa.shared.core.slots import TextSlot
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, UserUttered, ActionExecuted
from rasa.shared.core.training_data.structures import StoryGraph, StoryStep
from rasa.shared.core.trackers import DialogueStateTracker


def _dummy_features(id: int, attribute: Text) -> Features:
    return Features(
        np.full(shape=(1), fill_value=id),
        attribute=attribute,
        feature_type="really-anything",
        origin="",
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


def test_container_messages():
    message_data_list = [{INTENT: "1"}, {INTENT: "2", "other": 3}, {TEXT: "3"}]
    container = MessageContainerForCoreFeaturization()
    container.add_all([Message(data=data) for data in message_data_list])
    assert len(container.messages(INTENT)) == 2
    assert len(container.messages(TEXT)) == 1


def test_container_keys():
    message_data_list = [{INTENT: "1"}, {INTENT: "2"}, {TEXT: "3", "other": 3}]
    container = MessageContainerForCoreFeaturization()
    container.add_all([Message(data=data) for data in message_data_list])
    assert set(container.keys(INTENT)) == {"1", "2"}
    assert set(container.keys(TEXT)) == {"3"}


def test_container_all_messages():
    message_data_list = [{INTENT: "1"}, {INTENT: "2", "other": 3}, {TEXT: "3"}]
    container = MessageContainerForCoreFeaturization()
    container.add_all([Message(data=data) for data in message_data_list])
    assert len(container.all_messages()) == 3


def test_container_fingerprints_differ_for_different_containers():
    container1 = MessageContainerForCoreFeaturization()
    container1.add(Message(data={INTENT: "1"}))
    container2 = MessageContainerForCoreFeaturization()
    container2.add(Message(data={INTENT: "2"}))
    assert container2.fingerprint() != container1.fingerprint()


def test_container_fingerprint_differ_for_containers_with_different_insertion_order():
    # because we use this for training data and order might affect training of
    # e.g. featurizers, we want this to differ
    container1 = MessageContainerForCoreFeaturization()
    container1.add(Message(data={INTENT: "1"}))
    container1.add(Message(data={INTENT: "2"}))
    container2 = MessageContainerForCoreFeaturization()
    container2.add(Message(data={INTENT: "2"}))
    container2.add(Message(data={INTENT: "1"}))
    assert container2.fingerprint() != container1.fingerprint()


@pytest.mark.parametrize(
    "no_or_multiple_key_attributes",
    [list(), ["other"]]
    + list(
        itertools.permutations(MessageContainerForCoreFeaturization.KEY_ATTRIBUTES, 2)
    ),
)
def test_container_add_fails_if_message_has_wrong_attributes(
    no_or_multiple_key_attributes: List[Text],
):
    sub_state = {attribute: "dummy" for attribute in no_or_multiple_key_attributes}
    with pytest.raises(ValueError, match="Expected exactly one attribute out of"):
        MessageContainerForCoreFeaturization().add(Message(sub_state))


def test_container_add_message_copies():
    # construct a set of unique substates and messages
    dummy_value = "this-could-be-anything"
    substates_with_unique_key_attribute = [
        {INTENT: "greet"},
        {TEXT: "text", ENTITIES: dummy_value},
        {TEXT: "other-text"},
        {ACTION_TEXT: "action_text"},
        {ACTION_NAME: "action_name"},
    ]
    unique_messages = [
        Message(sub_state) for sub_state in substates_with_unique_key_attribute
    ]
    # make some copies
    num_copies = 3
    messages = unique_messages * (1 + num_copies)
    # build table
    lookup_table = MessageContainerForCoreFeaturization()
    for message in messages:
        lookup_table.add(message)
    # assert that we have as many entries as unique keys
    assert len(lookup_table) == len(substates_with_unique_key_attribute)
    assert set(lookup_table.all_messages()) == set(unique_messages)
    assert (
        lookup_table.num_collisions_ignored
        == len(substates_with_unique_key_attribute) * num_copies
    )


def test_container_add_does_not_fail_if_message_feature_content_differs():
    # construct a set of unique substates
    dummy_value = "this-could-be-anything"
    substates_with_unique_key_attribute = [
        {INTENT: "greet"},
        {TEXT: "text", ENTITIES: dummy_value},
        {ACTION_TEXT: "action_text"},
        {ACTION_NAME: "action_name"},
    ]
    constant_feature = _dummy_features(id=1, attribute="arbitrary")
    different_feature = _dummy_features(id=1, attribute="arbitrary")
    lookup_table = MessageContainerForCoreFeaturization()
    for sub_state in substates_with_unique_key_attribute:
        lookup_table.add(Message(data=sub_state, features=[constant_feature]))
    length = len(lookup_table)
    # with different feature
    for sub_state in substates_with_unique_key_attribute:
        lookup_table.add(Message(data=sub_state, features=[different_feature]))
        assert len(lookup_table) == length


def test_container_add_fails_if_messages_are_different_but_have_same_key():
    # construct a set of unique substates
    dummy_value = "this-could-be-anything"
    substates_with_unique_key_attribute = [
        {INTENT: "greet"},
        {TEXT: "text", ENTITIES: dummy_value},
        {ACTION_TEXT: "action_text"},
        {ACTION_NAME: "action_name"},
    ]
    constant_feature = _dummy_features(id=1, attribute="arbitrary")
    different_feature = _dummy_features(id=1, attribute="arbitrary")
    # adding the unique messages works fine of course,...
    lookup_table = MessageContainerForCoreFeaturization()
    for sub_state in substates_with_unique_key_attribute:
        lookup_table.add(Message(data=sub_state, features=[constant_feature]))
    # ... but adding any substate with same key but different content doesn't
    new_key = "some-new-key"
    expected_error_message = "Expected added message to be consistent"
    for sub_state in substates_with_unique_key_attribute:
        # with extra attribute
        sub_state_with_extra_attribute = sub_state.copy()
        sub_state_with_extra_attribute[new_key] = "some-value-for-the-new-key"
        with pytest.raises(ValueError, match=expected_error_message):
            lookup_table.add(Message(data=sub_state_with_extra_attribute))
        # with new feature
        with pytest.raises(ValueError, match=expected_error_message):
            lookup_table.add(
                Message(data=sub_state, features=[constant_feature, different_feature])
            )
        # without features
        with pytest.raises(ValueError, match=expected_error_message):
            lookup_table.add(Message(data=sub_state))
        # ... and we could test many more but this should suffice.


def test_container_feature_lookup():
    arbitrary_attribute = "other"
    messages = [
        Message(data={TEXT: "A"}, features=[_dummy_features(1, TEXT)]),
        Message(
            data={INTENT: "B", arbitrary_attribute: "C"},
            features=[_dummy_features(2, arbitrary_attribute)],
        ),
        Message(data={TEXT: "A2"}, features=[_dummy_features(3, TEXT)]),
        Message(
            data={INTENT: "B2", arbitrary_attribute: "C2"},
            features=[_dummy_features(4, arbitrary_attribute)],
        ),
    ]

    table = MessageContainerForCoreFeaturization()
    table.add_all(messages)

    # If we don't specify a list of attributes, the resulting features dictionary will
    # only contain those attributes for which there are features.
    sub_state = {TEXT: "A", INTENT: "B", arbitrary_attribute: "C"}
    features = table.collect_features(sub_state=sub_state)
    for attribute, feature_value in [
        (TEXT, 1),
        (INTENT, None),
        (arbitrary_attribute, 2),
    ]:
        if feature_value is not None:
            assert attribute in features
            assert len(features[attribute]) == 1
            assert feature_value == features[attribute][0].features[0]
        else:
            assert attribute not in features

    # If we query features for `INTENT`, then a key will be there, even if there are
    # no features
    features = table.collect_features(
        sub_state=sub_state, attributes=list(sub_state.keys())
    )
    assert INTENT in features
    assert len(features[INTENT]) == 0

    # We only get the list of features we want...
    features = table.collect_features(sub_state, attributes=[arbitrary_attribute])
    assert TEXT not in features
    assert INTENT not in features
    assert len(features[arbitrary_attribute]) == 1

    # ... even if there are no features:
    YET_ANOTHER = "another"
    features = table.collect_features(sub_state, attributes=[YET_ANOTHER])
    assert len(features[YET_ANOTHER]) == 0


def test_container_feature_lookup_fails_without_key_attribute():
    table = MessageContainerForCoreFeaturization()
    with pytest.raises(ValueError, match="Unknown key"):
        table.collect_features({TEXT: "A-unknown"})


def test_container_feature_lookup_fails_if_different_features_for_same_attribute():
    broken_table = MessageContainerForCoreFeaturization()
    broken_table._table = {
        TEXT: {"A": Message(data={}, features=[_dummy_features(2, TEXT)])},
        INTENT: {"B": Message(data={}, features=[_dummy_features(1, TEXT)])},
    }
    with pytest.raises(
        RuntimeError, match=f"Feature for attribute {TEXT} has already been"
    ):
        broken_table.collect_features({TEXT: "A", INTENT: "B"})


def test_container_feature_lookup_works_if_messages_are_broken_but_consistent():
    not_broken_but_strange_table = MessageContainerForCoreFeaturization()
    not_broken_but_strange_table._table = {
        TEXT: {"A": Message(data=dict())},
        INTENT: {"B": Message(data=dict(), features=[_dummy_features(1, TEXT)])},
    }
    features = not_broken_but_strange_table.collect_features({TEXT: "A", INTENT: "B"})
    assert TEXT in features and len(features[TEXT]) == 1


def test_container_message_lookup():
    # create some messages with unique key attributes
    messages = [
        Message(data={TEXT: "A"}, features=[_dummy_features(1, TEXT)]),
        Message(data={TEXT: "B"}),
        Message(data={INTENT: "B"}),
        Message(data={ACTION_TEXT: "B"}),
        Message(data={ACTION_NAME: "B"}),
    ]
    # add messages to container
    table = MessageContainerForCoreFeaturization()
    table.add_all(messages)
    # lookup messages using existing texts
    message = table.lookup_message(user_text="A")
    assert message
    assert len(message.data) == 1
    assert len(message.features) == 1
    message = table.lookup_message(user_text="B")
    assert message
    assert len(message.data) == 1


def test_container_message_lookup_fails_if_text_cannot_be_looked_up():
    table = MessageContainerForCoreFeaturization()
    with pytest.raises(ValueError, match="Expected a message with key"):
        table.lookup_message(user_text="a text not included in the table")


@pytest.mark.parametrize(
    "events,expected_num_entries",
    [
        (
            [
                UserUttered(intent={INTENT_NAME_KEY: "greet"}),
                ActionExecuted(action_name="utter_greet"),
                ActionExecuted(action_name="utter_greet"),
            ],
            2,
        ),
        (
            [
                UserUttered(text="text", intent={INTENT_NAME_KEY: "greet"}),
                ActionExecuted(action_name="utter_greet"),
            ],
            3,
        ),
    ],
)
def test_container_derive_messages_from_events_and_add(
    events: List[Event], expected_num_entries: int
):
    lookup_table = MessageContainerForCoreFeaturization()
    lookup_table.derive_messages_from_events_and_add(events)
    assert len(lookup_table) == expected_num_entries


def test_container_derive_messages_from_domain_and_add():
    action_names = ["a", "b"]
    # action texts, response keys, forms, and action_names must be unique or the
    # domain will complain about it ...
    action_texts = ["a2", "b2"]
    # ... but the response texts could overlap with e.g action texts
    responses = {"a3": {TEXT: "a2"}, "b3": {TEXT: "b2"}}
    forms = {"a4": "a4"}
    # however, intent names can be anything
    intents = ["a", "b"]
    domain = Domain(
        intents=intents,
        action_names=action_names,
        action_texts=action_texts,
        responses=responses,
        entities=["e_a", "e_b", "e_c"],
        slots=[TextSlot(name="s", mappings=[{}])],
        forms=forms,
        data={},
    )
    lookup_table = MessageContainerForCoreFeaturization()
    lookup_table.derive_messages_from_domain_and_add(domain)
    assert len(lookup_table) == (
        len(domain.intent_properties) + len(domain.action_names_or_texts)
    )


@pytest.fixture
def input_converter(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    return CoreFeaturizationInputConverter.create(
        CoreFeaturizationInputConverter.get_default_config(),
        default_model_storage,
        Resource("CoreFeaturizationInputConverters"),
        default_execution_context,
    )


def test_converter_for_training(input_converter: CoreFeaturizationInputConverter):
    # create domain and story graph
    domain = Domain(
        intents=["greet", "inform", "domain-only-intent"],
        entities=["entity_name"],
        slots=[],
        responses=dict(),
        action_names=["action_listen", "utter_greet"],
        forms=dict(),
        data={},
        action_texts=["Hi how are you?"],
    )
    events = [
        ActionExecuted(action_name="action_listen"),
        UserUttered(
            text="hey this has some entities",
            intent={INTENT_NAME_KEY: "greet"},
            entities=[_create_entity(value="Bot", type="entity_name")],
        ),
        ActionExecuted(action_name="utter_greet", action_text="Hi how are you?"),
        ActionExecuted(action_name="action_listen"),
        UserUttered(
            text="some test with an intent!", intent={INTENT_NAME_KEY: "inform"}
        ),
        ActionExecuted(action_name="action_listen"),
    ]
    story_graph = StoryGraph([StoryStep("name", events=events)])
    # convert!
    training_data = input_converter.convert_for_training(
        domain=domain, story_graph=story_graph
    )
    messages = training_data.training_examples
    # check that messages were created from (story) events as expected
    _check_messages_created_from_events_as_expected(events=events, messages=messages)
    # check that messages were created from domain as expected
    for intent in domain.intent_properties:
        assert Message(data={INTENT: intent}) in messages
    for action_name_or_text in domain.action_names_or_texts:
        if action_name_or_text in domain.action_texts:
            assert Message(data={ACTION_TEXT: action_name_or_text}) in messages
        else:
            assert Message(data={ACTION_NAME: action_name_or_text}) in messages
    # check that each message contains only one attribute, which must be a key attribute
    _check_messages_contain_attribute_which_is_key_attribute(messages=messages)


def _check_messages_created_from_events_as_expected(
    events: List[Event], messages: List[Message]
) -> None:
    for event in events:
        expected = []
        if isinstance(event, UserUttered):
            if event.text is not None:
                expected.append({TEXT: event.text})
            if event.intent_name is not None:
                expected.append({INTENT: event.intent_name})
        if isinstance(event, ActionExecuted):
            if event.action_name is not None:
                expected.append({ACTION_NAME: event.action_name})
            if event.action_text is not None:
                expected.append({ACTION_TEXT: event.action_text})
        for sub_state in expected:
            assert Message(sub_state) in messages


def _check_messages_contain_attribute_which_is_key_attribute(messages: List[Message]):
    for message in messages:
        assert len(message.data) == 1
        assert (
            list(message.data.keys())[0]
            in MessageContainerForCoreFeaturization.KEY_ATTRIBUTES
        )


def test_converter_for_inference(input_converter: CoreFeaturizationInputConverter):
    # create tracker
    events = [
        UserUttered(
            text="some text with entities!",
            intent={INTENT_NAME_KEY: "greet"},
            entities=[_create_entity(value="Bot", type="entity_name")],
        ),
        ActionExecuted(action_name="utter_greet", action_text="Hi how are you?"),
        ActionExecuted(action_name="action_listen"),
        UserUttered(text="some text with intent!", intent={INTENT_NAME_KEY: "inform"}),
    ]
    tracker = DialogueStateTracker.from_events(sender_id="arbitrary", evts=events)
    # convert!
    messages = input_converter.convert_for_inference(tracker)
    # check that messages were created from tracker events as expected
    _check_messages_created_from_events_as_expected(
        events=tracker.events, messages=messages
    )
    # check that each message contains only one attribute, which must be a key attribute
    _check_messages_contain_attribute_which_is_key_attribute(messages=messages)


@pytest.fixture
def collector(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    return CoreFeaturizationCollector.create(
        CoreFeaturizationCollector.get_default_config(),
        default_model_storage,
        Resource("CoreFeaturizationCollector"),
        default_execution_context,
    )


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
def test_collection(
    collector: CoreFeaturizationCollector,
    messages_with_unique_lookup_key: List[Message],
):

    messages = messages_with_unique_lookup_key

    # pass as training data
    training_data = TrainingData(training_examples=messages)
    precomputations = collector.collect(training_data)
    assert len(precomputations) == len(messages)

    # pass the list of messages directly
    precomputations = collector.collect(messages)
    assert len(precomputations) == len(messages)


def test_collection_fails(collector: CoreFeaturizationCollector):
    """The collection expects messages that have a unique lookup key.

    This is because they (should) have been passed through the preparation stage which
    will have constructed messages with this property.
    """
    messages = [
        Message(data={TEXT: "A", ACTION_TEXT: "B"}, features=[_dummy_features(1, TEXT)])
    ]
    training_data = TrainingData(training_examples=messages)

    with pytest.raises(ValueError):
        collector.collect(training_data)
    with pytest.raises(ValueError):
        collector.collect(messages)
