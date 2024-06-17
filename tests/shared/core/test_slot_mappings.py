import uuid
import warnings
from typing import Any, Dict, List, Optional, Text
from unittest.mock import Mock

import pytest
from pytest import MonkeyPatch

from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.constants import SlotMappingType

from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered, ActiveLoop
from rasa.shared.core.slot_mappings import (
    SlotFillingManager,
    SlotMapping,
    extract_slot_value,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    INTENT,
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.utils.yaml import YamlValidationException


@pytest.fixture
def cuisine_domain():
    return Domain.from_yaml(
        f"""
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
    intents:
    - greet
    - inform
    - affirm
    - deny
    entities:
    - cuisine
    slots:
        cuisine:
            type: text
            mappings:
            - type: from_entity
              entity: cuisine
        order_confirmation:
            type: bool
            mappings:
            - type: from_intent
              intent: affirm
              value: true
            - type: from_intent
              intent: deny
              value: false
        """
    )


@pytest.fixture
def sender_id() -> str:
    return uuid.uuid4().hex


@pytest.fixture
def cuisine_tracker(cuisine_domain: Domain, sender_id: str) -> DialogueStateTracker:
    events = [
        UserUttered("Hello", intent={"name": "greet"}),
        BotUttered("How can I help you?"),
    ]
    return DialogueStateTracker.from_events(
        sender_id, evts=events, slots=cuisine_domain.slots
    )


@pytest.mark.parametrize(
    "slot_name, expected", [("GPE_destination", ["Vancouver"]), ("GPE_origin", [])]
)
def test_slot_mapping_entity_is_desired(slot_name: Text, expected: List[Any]):
    domain = Domain.from_file("data/test_domains/travel_form.yml")
    tracker = DialogueStateTracker("test_id", slots=domain.slots)
    event = UserUttered(
        text="I'm travelling to Vancouver.",
        intent={"name": "inform", "confidence": 0.9604260921478271},
        entities=[{"entity": "GPE", "value": "Vancouver", "role": "destination"}],
    )
    tracker.update(event, domain)
    slot_mappings = domain.as_dict().get("slots").get(slot_name).get("mappings")
    actual = SlotMapping.entity_is_desired(slot_mappings[0], tracker)
    assert actual == expected


@pytest.mark.parametrize(
    "user_text, entities, expected",
    [
        (
            "I'd like to order some sushi",
            [{ENTITY_ATTRIBUTE_TYPE: "cuisine", ENTITY_ATTRIBUTE_VALUE: "sushi"}],
            ["sushi"],
        ),
        ("Goodbye", [], []),
        (
            "I'd like to book a table for 2",
            [{ENTITY_ATTRIBUTE_TYPE: "number", ENTITY_ATTRIBUTE_VALUE: "2"}],
            [],
        ),
    ],
)
def test_slot_mapping_entity_is_desired_with_message(
    user_text: str,
    entities: List,
    expected: bool,
    cuisine_domain: Domain,
    cuisine_tracker: DialogueStateTracker,
) -> None:
    mappings_for_cuisine = (
        cuisine_domain.as_dict().get("slots").get("cuisine").get("mappings")
    )
    message = Message(data={TEXT: user_text, ENTITIES: entities})

    assert (
        SlotMapping.entity_is_desired(mappings_for_cuisine[0], cuisine_tracker, message)
        == expected
    )


def test_slot_mapping_intent_is_desired() -> None:
    domain = Domain.from_file("examples/nlu_based/formbot/domain.yml")
    tracker = DialogueStateTracker("sender_id_test", slots=domain.slots)
    event1 = UserUttered(
        text="I'd like to book a restaurant for 2 people.",
        intent={"name": "request_restaurant", "confidence": 0.9604260921478271},
        entities=[{"entity": "number", "value": 2}],
    )
    tracker.update(event1, domain)
    mappings_for_num_people = (
        domain.as_dict().get("slots").get("num_people").get("mappings")
    )
    assert SlotMapping.intent_is_desired(mappings_for_num_people[0], tracker, domain)

    event2 = UserUttered(
        text="Yes, 2 please",
        intent={"name": "affirm", "confidence": 0.9604260921478271},
        entities=[{"entity": "number", "value": 2}],
    )
    tracker.update(event2, domain)
    assert (
        SlotMapping.intent_is_desired(mappings_for_num_people[0], tracker, domain)
        is False
    )

    event3 = UserUttered(
        text="Yes, please",
        intent={"name": "affirm", "confidence": 0.9604260921478271},
        entities=[],
    )
    tracker.update(event3, domain)
    mappings_for_preferences = (
        domain.as_dict().get("slots").get("preferences").get("mappings")
    )
    assert (
        SlotMapping.intent_is_desired(mappings_for_preferences[0], tracker, domain)
        is False
    )


@pytest.mark.parametrize(
    "user_text, intent, expected",
    [
        ("I'd like to order some sushi", "inform", True),
        ("Goodbye", "goodbye", False),
        ("What's the weather like today?", None, False),
        ("I'd like to book a table", "book_restaurant", False),
    ],
)
def test_slot_mapping_intent_is_desired_with_message(
    user_text: str,
    intent: Optional[str],
    expected: bool,
    sender_id: str,
) -> None:
    domain = Domain.from_yaml(
        f"""
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
    intents:
    - greet
    - inform
    - goodbye
    - book_restaurant
    slots:
        cuisine:
            type: text
            mappings:
            - type: from_text
              intent: inform
              not_intent: book_restaurant
        """
    )
    events = [
        UserUttered("Hello", intent={"name": "greet"}),
        BotUttered("How can I help you?"),
    ]
    tracker = DialogueStateTracker.from_events(
        sender_id, evts=events, slots=domain.slots
    )

    mappings_for_cuisine = domain.as_dict().get("slots").get("cuisine").get("mappings")
    message = Message(data={TEXT: user_text, INTENT: {"name": intent}})

    assert (
        SlotMapping.intent_is_desired(mappings_for_cuisine[0], tracker, domain, message)
        is expected
    )


def test_slot_mappings_ignored_intents_during_active_loop():
    domain = Domain.from_yaml(
        """
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
    intents:
    - greet
    - chitchat
    slots:
      cuisine:
        type: text
        mappings:
        - type: from_text
          conditions:
          - active_loop: restaurant_form
    forms:
      restaurant_form:
        ignored_intents:
        - chitchat
        required_slots:
        - cuisine
    """
    )
    tracker = DialogueStateTracker("sender_id", slots=domain.slots)
    event1 = ActiveLoop("restaurant_form")
    event2 = UserUttered(
        text="The weather is sunny today",
        intent={"name": "chitchat", "confidence": 0.9604260921478271},
        entities=[],
    )
    tracker.update_with_events([event1, event2])
    mappings_for_cuisine = domain.as_dict().get("slots").get("cuisine").get("mappings")
    assert (
        SlotMapping.intent_is_desired(mappings_for_cuisine[0], tracker, domain) is False
    )


def test_slot_mappings_invalid_type_raises():
    with pytest.raises(YamlValidationException):
        Domain.from_yaml(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            entities:
            - from_entity
            slots:
              some_slot:
                type: text
                influence_conversation: False
                mappings:
                  type: from_entity
                  entity: some_entity
            """
        )


def test_slot_mappings_check_mapping_validity_from_intent():
    slot_name = "mood"
    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - greet
        - goodbye
        - mood_great
        - mood_unhappy

        slots:
          {slot_name}:
            type: categorical
            values:
             - great
             - sad
            mappings:
             - type: from_intent
               not_intent: mood_unhappy
               value: great
             - type: from_intent
               intent: mood_unhappy
               value: sad
        """
    )
    mappings_for_slot = domain.as_dict().get("slots").get(slot_name).get("mappings")
    assert SlotMapping.check_mapping_validity(
        slot_name=slot_name,
        mapping_type=SlotMappingType.FROM_INTENT,
        mapping=mappings_for_slot[0],
        domain=domain,
    )


@pytest.mark.parametrize(
    "intent, expected",
    [
        (["goodbye", "mood_great", "greet"], True),
        ([], True),
        ("", True),
        ({}, True),
        ("null", True),
    ],
)
def test_slot_mappings_check_mapping_validity_valid_intent_list(
    intent: Text, expected: bool
):
    slot_name = "mood"
    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - greet
        - goodbye
        - mood_great
        - mood_unhappy
        slots:
          {slot_name}:
            type: any
            influence_conversation: false
            mappings:
            - type: from_intent
              value: "testing 123"
              intent: {intent}
        forms:
            test_form:
                required_slots:
                - test_slot
        """
    )
    mappings_for_slot = domain.as_dict().get("slots").get(slot_name).get("mappings")
    assert (
        SlotMapping.check_mapping_validity(
            slot_name=slot_name,
            mapping_type=SlotMappingType.FROM_INTENT,
            mapping=mappings_for_slot[0],
            domain=domain,
        )
        is expected
    )


def test_slot_mappings_check_mapping_validity_invalid_intent_list():
    slot_name = "mood"
    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - greet
        - goodbye
        - mood_great
        - mood_unhappy
        slots:
          {slot_name}:
            type: any
            influence_conversation: false
            mappings:
            - type: from_intent
              value: "testing 123"
              intent:
                - aaaa
                - bbbb
                - cccc
        forms:
            test_form:
                required_slots:
                - test_slot
        """
    )
    mappings_for_slot = domain.as_dict().get("slots").get(slot_name).get("mappings")
    assert not SlotMapping.check_mapping_validity(
        slot_name=slot_name,
        mapping_type=SlotMappingType.FROM_INTENT,
        mapping=mappings_for_slot[0],
        domain=domain,
    )


@pytest.mark.parametrize(
    "slot_name, mapping, expected",
    [
        ("cuisine", {"type": "from_entity", "entity": "cuisine"}, True),
        (
            "order_confirmation",
            {"type": "from_intent", "intent": "affirm", "value": True},
            True,
        ),
    ],
)
def test_slot_filling_manager_is_slot_mapping_valid_true(
    slot_name: str,
    mapping: Dict,
    expected: bool,
    cuisine_domain: Domain,
    cuisine_tracker: DialogueStateTracker,
) -> None:
    slot_filling_manager = SlotFillingManager(cuisine_domain, cuisine_tracker)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            slot_filling_manager.is_slot_mapping_valid(
                slot_name=slot_name,
                mapping_type=SlotMappingType(mapping.get("type")),
                mapping=mapping,
            )
            is expected
        )


@pytest.mark.parametrize(
    "slot_name, mapping, expected, warning",
    [
        (
            "mood",
            {"type": "from_entity", "entity": "mood"},
            False,
            "Slot 'mood' uses a 'from_entity' mapping for a "
            "non-existent entity 'mood'.",
        ),
        (
            "order_confirmation",
            {"type": "from_intent", "intent": "confirm", "value": True},
            False,
            "Slot 'order_confirmation' uses a 'from_intent' mapping for a "
            "non-existent intent 'confirm'.",
        ),
    ],
)
def test_slot_filling_manager_is_slot_mapping_valid_false(
    cuisine_domain: Domain,
    cuisine_tracker: DialogueStateTracker,
    slot_name: str,
    mapping: Dict,
    expected: bool,
    warning: str,
) -> None:
    slot_filling_manager = SlotFillingManager(cuisine_domain, cuisine_tracker)

    with pytest.warns(UserWarning, match=warning):
        assert (
            slot_filling_manager.is_slot_mapping_valid(
                slot_name=slot_name,
                mapping_type=SlotMappingType(mapping.get("type")),
                mapping=mapping,
            )
            is expected
        )


@pytest.mark.parametrize(
    "active_flow, expected",
    [
        ("mood_flow", True),
        ("cuisine_flow", False),
    ],
)
def test_slot_filling_manager_verify_mapping_conditions_active_flow(
    sender_id: str,
    cuisine_domain: Domain,
    active_flow: str,
    expected: bool,
) -> None:
    slot_name = "mood"
    flow_id = "mood_flow"
    mood_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - mood_great
        slots:
          {slot_name}:
            type: any
            mappings:
            - type: from_intent
              intent: mood_great
              value: "great"
              conditions:
              - active_flow: {flow_id}
        """
    )
    domain = cuisine_domain.merge(mood_domain)

    tracker = DialogueStateTracker.from_events(sender_id, evts=[], slots=domain.slots)
    user_frame = UserFlowStackFrame(
        flow_id=active_flow, step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame])
    tracker.update_stack(stack)

    slot_filling_manager = SlotFillingManager(domain, tracker)
    mapping = next((slot.mappings for slot in domain.slots if slot.name == slot_name))
    assert (
        slot_filling_manager._verify_mapping_conditions(mapping[0], slot_name)
        is expected
    )


def test_slot_filling_manager_verify_mapping_conditions_no_active_flow() -> None:
    slot_name = "mood"
    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - mood_great
        slots:
          {slot_name}:
            type: any
            mappings:
            - type: from_intent
              intent: mood_great
              value: "great"
              conditions:
              - active_flow: mood_flow
        """
    )
    tracker = DialogueStateTracker("sender_id", slots=domain.slots)
    slot_filling_manager = SlotFillingManager(domain, tracker)
    mapping = next((slot.mappings for slot in domain.slots if slot.name == slot_name))
    assert (
        slot_filling_manager._verify_mapping_conditions(mapping[0], slot_name) is False
    )


def test_slot_filling_manager_should_fill_slot_invalid_slot_mapping(
    cuisine_domain: Domain,
    cuisine_tracker: DialogueStateTracker,
    monkeypatch: MonkeyPatch,
) -> None:
    slot_name = "mood"
    mapping = {"type": "from_entity", "entity": "mood"}
    mapping_type = SlotMappingType(mapping.get("type"))
    slot_filling_manager = SlotFillingManager(cuisine_domain, cuisine_tracker)

    mock_is_slot_mapping_valid = Mock(return_value=False)
    mock_is_intent_desired = Mock()
    monkeypatch.setattr(
        slot_filling_manager, "is_slot_mapping_valid", mock_is_slot_mapping_valid
    )
    monkeypatch.setattr(
        slot_filling_manager, "is_intent_desired", mock_is_intent_desired
    )

    assert (
        slot_filling_manager.should_fill_slot(slot_name, mapping_type, mapping) is False
    )
    mock_is_slot_mapping_valid.assert_called_once_with(slot_name, mapping_type, mapping)
    mock_is_intent_desired.assert_not_called()


def test_slot_filling_manager_should_fill_slot_intent_not_desired(
    cuisine_domain: Domain,
    cuisine_tracker: DialogueStateTracker,
    monkeypatch: MonkeyPatch,
) -> None:
    slot_name = "cuisine"
    mapping = {"type": "from_entity", "entity": "cuisine"}
    mapping_type = SlotMappingType(mapping.get("type"))
    slot_filling_manager = SlotFillingManager(cuisine_domain, cuisine_tracker)

    mock_is_intent_desired = Mock(return_value=False)
    mock_verify_mapping_conditions = Mock()
    monkeypatch.setattr(
        slot_filling_manager, "is_intent_desired", mock_is_intent_desired
    )
    monkeypatch.setattr(
        slot_filling_manager,
        "_verify_mapping_conditions",
        mock_verify_mapping_conditions,
    )

    assert (
        slot_filling_manager.should_fill_slot(slot_name, mapping_type, mapping) is False
    )
    mock_is_intent_desired.assert_called_once_with(mapping)
    mock_verify_mapping_conditions.assert_not_called()


def test_slot_filling_manager_should_fill_slot_not_matching_mapping_conditions(
    cuisine_domain: Domain,
    cuisine_tracker: DialogueStateTracker,
    monkeypatch: MonkeyPatch,
) -> None:
    slot_name = "cuisine"
    mapping = {"type": "from_entity", "entity": "cuisine"}
    mapping_type = SlotMappingType(mapping.get("type"))
    slot_filling_manager = SlotFillingManager(cuisine_domain, cuisine_tracker)

    mock_verify_mapping_conditions = Mock(return_value=False)
    mock_fails_unique_entity_mapping_check = Mock()
    monkeypatch.setattr(
        slot_filling_manager,
        "_verify_mapping_conditions",
        mock_verify_mapping_conditions,
    )
    monkeypatch.setattr(
        slot_filling_manager,
        "_fails_unique_entity_mapping_check",
        mock_fails_unique_entity_mapping_check,
    )

    assert (
        slot_filling_manager.should_fill_slot(slot_name, mapping_type, mapping) is False
    )
    mock_verify_mapping_conditions.assert_called_once_with(mapping, slot_name)
    mock_fails_unique_entity_mapping_check.assert_not_called()


def test_slot_filling_manager_should_fill_slot_not_fails_unique_entity_mapping_check(
    cuisine_domain: Domain,
    cuisine_tracker: DialogueStateTracker,
    monkeypatch: MonkeyPatch,
) -> None:
    slot_name = "cuisine"
    mapping = {"type": "from_entity", "entity": "cuisine"}
    mapping_type = SlotMappingType(mapping.get("type"))
    slot_filling_manager = SlotFillingManager(cuisine_domain, cuisine_tracker)

    mock_fails_unique_entity_mapping_check = Mock(return_value=True)
    monkeypatch.setattr(
        slot_filling_manager,
        "_fails_unique_entity_mapping_check",
        mock_fails_unique_entity_mapping_check,
    )

    assert (
        slot_filling_manager.should_fill_slot(slot_name, mapping_type, mapping) is False
    )
    mock_fails_unique_entity_mapping_check.assert_called_once_with(slot_name, mapping)


def test_slot_filling_manager_should_fill_slot_valid(
    cuisine_domain: Domain,
    cuisine_tracker: DialogueStateTracker,
    monkeypatch: MonkeyPatch,
) -> None:
    slot_name = "cuisine"
    mapping = {"type": "from_entity", "entity": "cuisine"}
    mapping_type = SlotMappingType(mapping.get("type"))
    slot_filling_manager = SlotFillingManager(cuisine_domain, cuisine_tracker)

    assert (
        slot_filling_manager.should_fill_slot(slot_name, mapping_type, mapping) is True
    )


@pytest.mark.parametrize(
    "data, slot_name, events, expected_value",
    [
        # from_entity
        (
            {
                TEXT: "I'd like to order some sushi",
                ENTITIES: [
                    {ENTITY_ATTRIBUTE_TYPE: "cuisine", ENTITY_ATTRIBUTE_VALUE: "sushi"}
                ],
            },
            "cuisine",
            [],
            "sushi",
        ),
        # from_text
        (
            {TEXT: "test@test.com", INTENT: {"name": "email"}},
            "email_address",
            [],
            "test@test.com",
        ),
        # from_intent
        ({TEXT: "Yes", INTENT: {"name": "affirm"}}, "order_confirmation", [], True),
        # from_trigger_intent
        (
            {TEXT: "Cancel order", INTENT: {"name": "cancel_order"}},
            "cancellation_confirmation",
            [ActiveLoop("cancel_order")],
            True,
        ),
    ],
)
def test_slot_filling_manager_extract_slot_value_from_predefined_mapping_with_message(
    cuisine_domain: Domain,
    sender_id: str,
    data: Dict,
    slot_name: str,
    events: List,
    expected_value: str,
) -> None:
    message = Message(data=data)
    new_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - email
        slots:
            email_address:
                type: text
                mappings:
                - type: from_text
                  intent: email
            cancellation_confirmation:
                type: bool
                mappings:
                - type: from_trigger_intent
                  intent: cancel_order
                  value: true
        forms:
            cancel_order:
                required_slots:
                - cancellation_confirmation
        """
    )
    domain = cuisine_domain.merge(new_domain)
    tracker = DialogueStateTracker.from_events(
        sender_id, evts=events, slots=domain.slots
    )
    slot_filling_manager = SlotFillingManager(domain, tracker, message)

    mappings = next((slot.mappings for slot in domain.slots if slot.name == slot_name))
    mapping_type = SlotMappingType(mappings[0].get("type"))

    value = slot_filling_manager.extract_slot_value_from_predefined_mapping(
        mapping_type, mappings[0]
    )
    assert value == [expected_value]


@pytest.mark.parametrize(
    "slot_name", ["cuisine", "order_confirmation", "membership", "email_address"]
)
def test_extract_slot_value_false(
    cuisine_domain: Domain,
    sender_id: str,
    slot_name: str,
) -> None:
    new_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
            email_address:
                type: text
                mappings:
                - type: from_llm
            membership:
                type: text
                mappings:
                - type: custom
        """
    )
    domain = cuisine_domain.merge(new_domain)
    tracker = DialogueStateTracker(sender_id, slots=domain.slots)
    message = Message(data={TEXT: "Hi"})
    slot_filling_manager = SlotFillingManager(domain, tracker, message)

    slot = next((slot for slot in domain.slots if slot.name == slot_name))

    assert extract_slot_value(slot, slot_filling_manager) == (None, False)


@pytest.mark.parametrize(
    "slot_name, data, expected_value",
    [
        (
            "cuisine",
            {
                TEXT: "I'd like some sushi",
                ENTITIES: [
                    {ENTITY_ATTRIBUTE_TYPE: "cuisine", ENTITY_ATTRIBUTE_VALUE: "sushi"}
                ],
            },
            "sushi",
        ),
        ("order_confirmation", {TEXT: "Confirmed", INTENT: {"name": "affirm"}}, True),
        ("order_confirmation", {TEXT: "No", INTENT: {"name": "deny"}}, False),
        (
            "accounts",
            {
                TEXT: "I have a savings account.",
                ENTITIES: [
                    {
                        ENTITY_ATTRIBUTE_TYPE: "account_type",
                        ENTITY_ATTRIBUTE_VALUE: "savings",
                    }
                ],
            },
            ["savings"],
        ),
        # test that a slot that was already set can be reset
        ("email_address", {INTENT: {"name": "email"}}, None),
    ],
)
def test_extract_slot_value_true(
    cuisine_domain: Domain,
    sender_id: str,
    slot_name: str,
    data: Dict,
    expected_value: Any,
) -> None:
    new_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - email
        entities:
        - account_type
        slots:
            accounts:
                type: list
                mappings:
                - type: from_entity
                  entity: account_type
            email_address:
                type: text
                mappings:
                - type: from_text
                  intent: email
        """
    )
    domain = cuisine_domain.merge(new_domain)
    tracker = DialogueStateTracker.from_events(
        sender_id, evts=[SlotSet("email_address", "test@test.com")], slots=domain.slots
    )
    message = Message(data=data)
    slot_filling_manager = SlotFillingManager(domain, tracker, message)

    slot = next((slot for slot in domain.slots if slot.name == slot_name))

    assert extract_slot_value(slot, slot_filling_manager) == (expected_value, True)
