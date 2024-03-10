from typing import Dict, List, Text, Any

import logging
import pytest
from _pytest.logging import LogCaptureFixture

from rasa.core.nlg.response import TemplatedNaturalLanguageGenerator
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.domain import Domain
from rasa.shared.core.slots import TextSlot, AnySlot, CategoricalSlot, BooleanSlot
from rasa.shared.core.trackers import DialogueStateTracker


async def test_nlg_conditional_response_variations_with_no_slots():
    responses = {
        "utter_test": [
            {
                "text": "Conditional OS Response A",
                "condition": [{"type": "slot", "name": "slot test", "value": "A"}],
                "channel": "os",
            },
            {
                "text": "Conditional Response A",
                "condition": [{"type": "slot", "name": "slot test", "value": "A"}],
            },
            {
                "text": "Conditional Response B",
                "condition": [{"type": "slot", "name": "slot test", "value": "B"}],
            },
            {"text": "Default response"},
        ]
    }
    t = TemplatedNaturalLanguageGenerator(responses=responses)
    no_slots_tracker = DialogueStateTracker(sender_id="nlg_test_default", slots=None)
    default_response = await t.generate(
        utter_action="utter_test", tracker=no_slots_tracker, output_channel=""
    )

    assert default_response.get("text") == "Default response"


async def test_nlg_when_multiple_conditions_satisfied():
    responses = {
        "utter_action": [
            {
                "text": "example A",
                "condition": [{"type": "slot", "name": "test", "value": "A"}],
            },
            {
                "text": "example B",
                "condition": [{"type": "slot", "name": "test_another", "value": "B"}],
            },
            {
                "text": "non matching example 1",
                "condition": [
                    {"type": "slot", "name": "test_third_slot", "value": "C"}
                ],
            },
            {
                "text": "non matching example 2",
                "condition": [{"type": "slot", "name": "test", "value": "D"}],
            },
        ]
    }

    t = TemplatedNaturalLanguageGenerator(responses=responses)
    slot_a = TextSlot(
        name="test", mappings=[{}], initial_value="A", influence_conversation=False
    )
    slot_b = TextSlot(
        name="test_another",
        mappings=[{}],
        initial_value="B",
        influence_conversation=False,
    )
    tracker = DialogueStateTracker(sender_id="test_nlg", slots=[slot_a, slot_b])
    resp = await t.generate(
        utter_action="utter_action", tracker=tracker, output_channel=""
    )
    assert resp.get("text") in ["example A", "example B"]


@pytest.fixture(scope="session")
def test_slots() -> List[object]:
    slot_a = CategoricalSlot(
        name="test",
        mappings=[{"type": "from_text", "value": ["cold", "hot"]}],
        initial_value="Cold",
        influence_conversation=False,
    )
    slot_b = BooleanSlot(
        name="test2",
        mappings=[{}],
        initial_value=False,
        influence_conversation=False,
    )
    slot_c = CategoricalSlot(
        name="test3",
        mappings=[{"type": "from_text", "value": ["cold", "hot"]}],
        initial_value="hot",
        influence_conversation=False,
    )
    return [slot_a, slot_b, slot_c]


@pytest.fixture(scope="session")
def test_responses() -> List[Dict[Text, List[Dict[Text, Any]]]]:
    return [
        {
            "utter_action": [
                {
                    "text": "example a",
                    "condition": [{"type": "slot", "name": "test", "value": "cold"}],
                }
            ]
        },
        {
            "utter_action_multiple_conditions": [
                {
                    "text": "example b",
                    "condition": [
                        {"type": "slot", "name": "test", "value": "cold"},
                        {"type": "slot", "name": "test2", "value": False},
                        {"type": "slot", "name": "test3", "value": "hot"},
                    ],
                }
            ]
        },
    ]


async def test_nlg_slot_case_sensitivity(
    test_slots: List[object],
    test_responses: List[Dict[Text, List[Dict[Text, Any]]]],
):
    utter_action = "utter_action"
    t = TemplatedNaturalLanguageGenerator(responses=test_responses[0])
    tracker = DialogueStateTracker(sender_id="test_nlg", slots=[test_slots[0]])
    resp = await t.generate(
        utter_action=utter_action, tracker=tracker, output_channel=""
    )
    assert resp.get("text") == test_responses[0][utter_action][0]["text"]


async def test_matches_filled_slots_multiple_conditions(
    test_slots: List[object],
    test_responses: List[Dict[Text, List[Dict[Text, Any]]]],
):
    utter_action = "utter_action_multiple_conditions"
    t = TemplatedNaturalLanguageGenerator(responses=test_responses[1])
    tracker = DialogueStateTracker(sender_id="test_nlg", slots=test_slots)
    resp = await t.generate(
        utter_action=utter_action,
        tracker=tracker,
        output_channel="",
    )
    assert resp.get("text") == test_responses[1][utter_action][0]["text"]


async def test_matches_filled_slots_multiple_conditions_neg_match_boolean_slot(
    test_slots: List[object], test_responses: List[Dict[Text, List[Dict[Text, Any]]]]
):
    t = TemplatedNaturalLanguageGenerator(responses=test_responses[1])
    test_slots[1].initial_value = True
    tracker = DialogueStateTracker(sender_id="test_nlg", slots=test_slots)
    resp = await t.generate(
        utter_action="utter_action_multiple_conditions",
        tracker=tracker,
        output_channel="",
    )
    assert resp is None


async def test_matches_filled_slots_multiple_conditions_neg_match_in_first_slot(
    test_slots: List[object], test_responses: List[Dict[Text, List[Dict[Text, Any]]]]
):
    t = TemplatedNaturalLanguageGenerator(responses=test_responses[1])
    test_slots[0].initial_value = "junk"
    tracker = DialogueStateTracker(sender_id="test_nlg", slots=test_slots)
    resp = await t.generate(
        utter_action="utter_action_multiple_conditions",
        tracker=tracker,
        output_channel="",
    )
    assert resp is None


async def test_matches_filled_slots_multiple_conditions_neg_match_in_last_slot(
    test_slots: List[object], test_responses: List[Dict[Text, List[Dict[Text, Any]]]]
):
    t = TemplatedNaturalLanguageGenerator(responses=test_responses[1])
    test_slots[2].initial_value = "junk"
    tracker = DialogueStateTracker(sender_id="test_nlg", slots=test_slots)
    resp = await t.generate(
        utter_action="utter_action_multiple_conditions",
        tracker=tracker,
        output_channel="",
    )
    assert resp is None


@pytest.mark.parametrize(
    ("slot_name", "slot_value", "response_variation"),
    (("test", "A", "example one A"), ("test", "B", "example two B")),
)
async def test_nlg_conditional_response_variations_with_interpolated_slots(
    slot_name: Text, slot_value: Any, response_variation: Text
):
    responses = {
        "utter_action": [
            {
                "text": "example one {test}",
                "condition": [{"type": "slot", "name": "test", "value": "A"}],
            },
            {
                "text": "example two {test}",
                "condition": [{"type": "slot", "name": "test", "value": "B"}],
            },
        ]
    }
    t = TemplatedNaturalLanguageGenerator(responses=responses)
    slot = TextSlot(
        name=slot_name,
        mappings=[{}],
        initial_value=slot_value,
        influence_conversation=False,
    )
    tracker = DialogueStateTracker(sender_id="nlg_interpolated", slots=[slot])

    r = await t.generate(
        utter_action="utter_action", tracker=tracker, output_channel=""
    )
    assert r.get("text") == response_variation


@pytest.mark.parametrize(
    ("slot_name", "slot_value", "bot_message"),
    (
        (
            "can_withdraw",
            False,
            "You are not allowed to withdraw any amounts. Please check permission.",
        ),
        (
            "account_type",
            "secondary",
            "Withdrawal was sent for approval to primary account holder.",
        ),
    ),
)
async def test_nlg_conditional_response_variations_with_yaml_single_condition(
    slot_name: Text, slot_value: Any, bot_message: Text
):
    domain = Domain.from_file(
        path="data/test_domains/conditional_response_variations.yml"
    )
    t = TemplatedNaturalLanguageGenerator(responses=domain.responses)

    slot = AnySlot(
        name=slot_name,
        mappings=[{}],
        initial_value=slot_value,
        influence_conversation=False,
    )
    tracker = DialogueStateTracker(sender_id="conversation_id", slots=[slot])

    r = await t.generate(
        utter_action="utter_withdraw", tracker=tracker, output_channel=""
    )
    assert r.get("text") == bot_message


async def test_nlg_conditional_response_variations_with_yaml_multi_constraints():
    domain = Domain.from_file(
        path="data/test_domains/conditional_response_variations.yml"
    )
    t = TemplatedNaturalLanguageGenerator(responses=domain.responses)

    first_slot = CategoricalSlot(
        name="account_type",
        mappings=[{}],
        initial_value="primary",
        influence_conversation=False,
    )
    second_slot = BooleanSlot(
        name="can_withdraw",
        mappings=[{}],
        initial_value=True,
        influence_conversation=False,
    )
    tracker = DialogueStateTracker(
        sender_id="conversation_id", slots=[first_slot, second_slot]
    )
    r = await t.generate(
        utter_action="utter_withdraw", tracker=tracker, output_channel=""
    )
    assert r.get("text") == "Withdrawal has been approved."


async def test_nlg_conditional_response_variations_with_yaml_and_channel():
    domain = Domain.from_file(
        path="data/test_domains/conditional_response_variations.yml"
    )
    t = TemplatedNaturalLanguageGenerator(responses=domain.responses)

    slot = CategoricalSlot(
        name="account_type",
        mappings=[{}],
        initial_value="primary",
        influence_conversation=False,
    )
    tracker = DialogueStateTracker(sender_id="conversation_id", slots=[slot])

    r = await t.generate(
        utter_action="utter_check_balance", tracker=tracker, output_channel="os"
    )
    assert (
        r.get("text") == "As a primary account holder, you can now set-up "
        "your access on mobile app too."
    )

    resp = await t.generate(
        utter_action="utter_check_balance", tracker=tracker, output_channel="app"
    )
    assert resp.get("text") == "Welcome to your app account overview."


@pytest.mark.parametrize(
    ("slot_name", "slot_value", "message"),
    (
        ("test_bool", True, "example boolean"),
        ("test_int", 12, "example integer"),
        ("test_list", [], "example list"),
    ),
)
async def test_nlg_conditional_response_variations_with_diff_slot_types(
    slot_name: Text, slot_value: Any, message: Text
):
    responses = {
        "utter_action": [
            {
                "text": "example boolean",
                "condition": [{"type": "slot", "name": "test_bool", "value": True}],
            },
            {
                "text": "example integer",
                "condition": [{"type": "slot", "name": "test_int", "value": 12}],
            },
            {
                "text": "example list",
                "condition": [{"type": "slot", "name": "test_list", "value": []}],
            },
        ]
    }
    t = TemplatedNaturalLanguageGenerator(responses=responses)
    slot = AnySlot(
        name=slot_name,
        mappings=[{}],
        initial_value=slot_value,
        influence_conversation=False,
    )
    tracker = DialogueStateTracker(sender_id="nlg_tracker", slots=[slot])

    r = await t.generate(
        utter_action="utter_action", tracker=tracker, output_channel=""
    )
    assert r.get("text") == message


async def test_nlg_non_matching_channel():
    domain = Domain.from_yaml(
        """
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
    responses:
        utter_hi:
        - text: "Hello"
        - text: "Hello Slack"
          channel: "slack"
    """
    )
    t = TemplatedNaturalLanguageGenerator(domain.responses)
    tracker = DialogueStateTracker(sender_id="test", slots=[])
    r = await t.generate("utter_hi", tracker, "signal")
    assert r.get("text") == "Hello"


async def test_nlg_conditional_response_variations_with_none_slot():
    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
            utter_action:
            - text: "text A"
              condition:
              - type: slot
                name: account
                value: "A"
        """
    )
    t = TemplatedNaturalLanguageGenerator(domain.responses)
    slot = AnySlot(
        name="account", mappings=[{}], initial_value=None, influence_conversation=False
    )
    tracker = DialogueStateTracker(sender_id="test", slots=[slot])
    r = await t.generate("utter_action", tracker, "")
    assert r is None


async def test_nlg_conditional_response_variations_with_slot_not_a_constraint():
    domain = Domain.from_yaml(
        f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            responses:
                utter_action:
                - text: "text A"
                  condition:
                  - type: slot
                    name: account
                    value: "A"
            """
    )
    t = TemplatedNaturalLanguageGenerator(domain.responses)
    slot = TextSlot(
        name="account", mappings=[{}], initial_value="B", influence_conversation=False
    )
    tracker = DialogueStateTracker(sender_id="test", slots=[slot])
    r = await t.generate("utter_action", tracker, "")
    assert r is None


async def test_nlg_conditional_response_variations_with_null_slot():
    domain = Domain.from_yaml(
        f"""
                version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                responses:
                    utter_action:
                    - text: "text for null"
                      condition:
                      - type: slot
                        name: account
                        value: null
                """
    )
    t = TemplatedNaturalLanguageGenerator(domain.responses)
    slot = AnySlot(
        name="account", mappings=[{}], initial_value=None, influence_conversation=False
    )
    tracker = DialogueStateTracker(sender_id="test", slots=[slot])
    r = await t.generate("utter_action", tracker, "")
    assert r.get("text") == "text for null"

    tracker_no_slots = DialogueStateTracker(sender_id="new_test", slots=[])
    r = await t.generate("utter_action", tracker_no_slots, "")
    assert r.get("text") == "text for null"


async def test_nlg_conditional_response_variations_channel_no_condition_met():
    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
           utter_action:
             - text: "example with channel"
               condition:
                - type: slot
                  name: test
                  value: A
               channel: os
             - text: "default"
        """
    )
    t = TemplatedNaturalLanguageGenerator(domain.responses)
    tracker = DialogueStateTracker(sender_id="test", slots=[])
    r = await t.generate("utter_action", tracker, "os")
    assert r.get("text") == "default"


async def test_nlg_conditional_response_variation_condition_met_channel_mismatch():
    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
           utter_action:
             - text: "example with channel"
               condition:
                - type: slot
                  name: test
                  value: A
               channel: os
             - text: "app default"
               channel: app
        """
    )
    t = TemplatedNaturalLanguageGenerator(domain.responses)
    slot = TextSlot(
        "test", mappings=[{}], initial_value="A", influence_conversation=False
    )
    tracker = DialogueStateTracker(sender_id="test", slots=[slot])
    r = await t.generate("utter_action", tracker, "app")
    assert r.get("text") == "app default"


@pytest.mark.parametrize(
    "slots,channel,expected_response",
    [
        (
            [
                TextSlot(
                    "test",
                    mappings=[{}],
                    initial_value="B",
                    influence_conversation=False,
                )
            ],
            "app",
            "condition example B no channel",
        ),
        (
            [
                TextSlot(
                    "test",
                    mappings=[{}],
                    initial_value="C",
                    influence_conversation=False,
                )
            ],
            "",
            "default",
        ),
        (
            [
                TextSlot(
                    "test",
                    mappings=[{}],
                    initial_value="D",
                    influence_conversation=False,
                )
            ],
            "app",
            "default",
        ),
    ],
)
async def test_nlg_conditional_edgecases(slots, channel, expected_response):
    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
           utter_action:
             - text: "condition example A with channel"
               condition:
                - type: slot
                  name: test
                  value: A
               channel: app

             - text: "condition example C with channel"
               condition:
                - type: slot
                  name: test
                  value: C
               channel: app

             - text: "condition example A no channel"
               condition:
                - type: slot
                  name: test
                  value: A

             - text: "condition example B no channel"
               condition:
                - type: slot
                  name: test
                  value: B

             - text: "default"
        """
    )
    t = TemplatedNaturalLanguageGenerator(domain.responses)
    tracker = DialogueStateTracker(sender_id="test", slots=slots)
    r = await t.generate("utter_action", tracker, channel)
    assert r.get("text") == expected_response


async def test_nlg_conditional_response_variations_condition_logging(
    caplog: LogCaptureFixture,
):
    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
           utter_action:
             - text: "example"
               condition:
                - type: slot
                  name: test_A
                  value: A
                - type: slot
                  name: test_B
                  value: B
             - text: "default"
        """
    )
    t = TemplatedNaturalLanguageGenerator(domain.responses)
    slot_A = TextSlot(
        name="test_A", mappings=[{}], initial_value="A", influence_conversation=False
    )
    slot_B = TextSlot(
        name="test_B", mappings=[{}], initial_value="B", influence_conversation=False
    )
    tracker = DialogueStateTracker(sender_id="test", slots=[slot_A, slot_B])

    with caplog.at_level(logging.DEBUG):
        await t.generate("utter_action", tracker=tracker, output_channel="")

    assert any(
        "Selecting response variation with conditions:" in message
        for message in caplog.messages
    )
    assert any(
        "[condition 1] type: slot | name: test_A | value: A" in message
        for message in caplog.messages
    )
    assert any(
        "[condition 2] type: slot | name: test_B | value: B" in message
        for message in caplog.messages
    )
