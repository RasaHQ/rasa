from typing import Text

import pytest
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.constants import SlotMappingType

from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered, ActiveLoop
from rasa.shared.core.slot_mappings import SlotMapping
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils.validation import YamlValidationException


@pytest.mark.parametrize(
    "slot_name, expected", [("GPE_destination", True), ("GPE_origin", False)]
)
def test_slot_mapping_entity_is_desired(slot_name: Text, expected: bool):
    domain = Domain.from_file("data/test_domains/travel_form.yml")
    tracker = DialogueStateTracker("test_id", slots=domain.slots)
    event = UserUttered(
        text="I'm travelling to Vancouver.",
        intent={"name": "inform", "confidence": 0.9604260921478271},
        entities=[{"entity": "GPE", "value": "Vancouver", "role": "destination"}],
    )
    tracker.update(event, domain)
    slot_mappings = domain.as_dict().get("slots").get(slot_name).get("mappings")
    assert SlotMapping.entity_is_desired(slot_mappings[0], tracker) is expected


def test_slot_mapping_intent_is_desired(domain: Domain):
    domain = Domain.from_file("examples/formbot/domain.yml")
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
    tracker.update_with_events([event1, event2], domain)
    mappings_for_cuisine = domain.as_dict().get("slots").get("cuisine").get("mappings")
    assert (
        SlotMapping.intent_is_desired(mappings_for_cuisine[0], tracker, domain) is False
    )


def test_missing_slot_mappings_raises():
    with pytest.raises(YamlValidationException):
        Domain.from_yaml(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            slots:
              some_slot:
                type: text
                influence_conversation: False
            """
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
