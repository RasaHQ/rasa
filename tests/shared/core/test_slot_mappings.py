import pytest

from rasa.shared.core.domain import Domain, InvalidDomain, KEY_SLOTS
from rasa.shared.core.events import UserUttered, ActiveLoop
from rasa.shared.core.slot_mappings import SlotMapping
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.mark.parametrize(
    "slot_name, expected", [("GPE_destination", True), ("GPE_origin", False)]
)
def test_slot_mapping_entity_is_desired(slot_name, expected):
    domain = Domain.from_file("data/test_domains/travel_form.yml")
    tracker = DialogueStateTracker("test_id", slots=domain.slots)
    event = UserUttered(
        text="I'm travelling to Vancouver.",
        intent={"id": 1, "name": "inform", "confidence": 0.9604260921478271},
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
        intent={
            "id": 1,
            "name": "request_restaurant",
            "confidence": 0.9604260921478271,
        },
        entities=[{"entity": "number", "value": 2}],
    )
    tracker.update(event1, domain)
    mappings_for_num_people = (
        domain.as_dict().get("slots").get("num_people").get("mappings")
    )
    assert SlotMapping.intent_is_desired(mappings_for_num_people[0], tracker, domain)

    event2 = UserUttered(
        text="Yes, 2 please",
        intent={"id": 2, "name": "affirm", "confidence": 0.9604260921478271},
        entities=[{"entity": "number", "value": 2}],
    )
    tracker.update(event2, domain)
    assert (
        SlotMapping.intent_is_desired(mappings_for_num_people[0], tracker, domain)
        is False
    )

    event3 = UserUttered(
        text="Yes, please",
        intent={"id": 3, "name": "affirm", "confidence": 0.9604260921478271},
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
    version: "2.0"
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
        intent={"id": 4, "name": "chitchat", "confidence": 0.9604260921478271},
        entities=[],
    )
    tracker.update_with_events([event1, event2], domain)
    mappings_for_cuisine = domain.as_dict().get("slots").get("cuisine").get("mappings")
    assert (
        SlotMapping.intent_is_desired(mappings_for_cuisine[0], tracker, domain) is False
    )


@pytest.mark.parametrize(
    "domain_as_dict",
    [
        # missing mappings parameter
        {KEY_SLOTS: {"some_slot": {"type": "text", "influence_conversation": False,}}},
        # mappings is not of type list
        {KEY_SLOTS: {"some_slot": {"type": "text", "mappings": {},}}},
        # form does not exist in domain forms
        {
            KEY_SLOTS: {
                "some_slot": {
                    "type": "text",
                    "mappings": [
                        {
                            "type": "from_text",
                            "conditions": [{"active_loop": "some_form"}],
                        }
                    ],
                }
            }
        },
    ],
)
def test_validate_slot_mappings_raises(domain_as_dict):
    with pytest.raises(InvalidDomain):
        Domain.from_dict(domain_as_dict)
