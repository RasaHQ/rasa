import uuid
from typing import Any, Dict, Optional, Text

import pytest
from pytest import CaptureFixture, MonkeyPatch

from rasa.core import ContextualResponseRephraser
from rasa.core.actions.action_run_slot_rejections import (
    ActionRunSlotRejections,
    coerce_slot_value,
    utterance_for_slot_type,
)
from rasa.core.channels import OutputChannel
from rasa.core.constants import (
    DOMAIN_GROUND_TRUTH_METADATA_KEY,
    UTTER_SOURCE_METADATA_KEY,
    ACTIVE_FLOW_METADATA_KEY,
    STEP_ID_METADATA_KEY,
)
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.constants import OPENAI_API_KEY_ENV_VAR
from rasa.shared.core.domain import Domain, KEY_RESPONSES_TEXT
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered
from rasa.shared.core.slots import (
    BooleanSlot,
    CategoricalSlot,
    FloatSlot,
    Slot,
    TextSlot,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig


@pytest.fixture
def rejection_test_nlg() -> TemplatedNaturalLanguageGenerator:
    return TemplatedNaturalLanguageGenerator(
        {
            "utter_ask_recurrent_payment_type": [
                {"text": "What type of recurrent payment do you want to setup?"}
            ],
            "utter_invalid_recurrent_payment_type": [
                {"text": "Sorry, you requested an invalid recurrent payment type."}
            ],
            "utter_internal_error_rasa": [{"text": "Sorry, something went wrong."}],
            "utter_ask_payment_amount": [{"text": "What amount do you want to pay?"}],
            "utter_payment_too_high": [
                {"text": "Sorry, the amount is above the maximum £1,000 allowed."}
            ],
            "utter_payment_negative": [
                {"text": "Sorry, the amount cannot be negative."}
            ],
            "utter_categorical_slot_rejection": [
                {"text": "Sorry, you requested an option that is not valid."}
            ],
            "utter_ask_payment_execution_mode": [
                {"text": "When do you want to execute the payment?"},
            ],
            "utter_boolean_slot_rejection": [
                {
                    "text": "Sorry, the option you provided, {{value}}, is not valid.",
                    "metadata": {"template": "jinja"},
                }
            ],
            "utter_float_slot_rejection": [
                {
                    "text": "Sorry, the number you provided, {{value}}, is not valid.",
                    "metadata": {"template": "jinja"},
                }
            ],
        }
    )


@pytest.fixture
def rejection_test_domain() -> Domain:
    return Domain.from_yaml(
        """
        slots:
            recurrent_payment_type:
                type: text
                mappings: []
            payment_recipient:
                type: text
                mappings: []
            payment_amount:
                type: float
                mappings: []
            payment_execution_mode:
                type: categorical
                values:
                    - immediate
                    - future
                mappings:
                - type: custom
            payment_confirmation:
                type: bool
                mappings: []
        responses:
            utter_ask_recurrent_payment_type:
             - text: "What type of recurrent payment do you want to setup?"
            utter_invalid_recurrent_payment_type:
             - text: "Sorry, you requested an invalid recurrent payment type."
               metadata:
                 rephrase: True
            utter_internal_error_rasa:
             - text: "Sorry, something went wrong."
            utter_ask_payment_amount:
             - text: "What amount do you want to pay?"
            utter_payment_too_high:
             - text: "Sorry, the amount is above the maximum £1,000 allowed."
            utter_payment_negative:
             - text: "Sorry, the amount cannot be negative."
            utter_categorical_slot_rejection:
             - text: "Sorry, you requested an option that is not valid."
            utter_ask_payment_confirmation:
             - text: "Do you want to confirm the payment?"
        """
    )


@pytest.fixture
def rejection_test_dialogue_stack() -> DialogueStack:
    return DialogueStack.from_dict(
        [
            {
                "frame_id": "4YL3KDBR",
                "flow_id": "setup_recurrent_payment",
                "step_id": "ask_payment_type",
                "frame_type": "regular",
                "type": "flow",
            },
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "recurrent_payment_type",
                "utter": "utter_ask_recurrent_payment_type",
                "collect_action": "action_ask_recurrent_payment_type",
                "rejections": [
                    {
                        "if": 'not ({"direct debit" "standing order"} '
                        "contains slots.recurrent_payment_type)",
                        "utter": "utter_invalid_recurrent_payment_type",
                    }
                ],
                "type": "pattern_collect_information",
            },
        ]
    )


async def test_action_run_slot_rejections_top_frame_not_collect_information(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "4YL3KDBR",
                "flow_id": "setup_recurrent_payment",
                "step_id": "ask_payment_type",
                "frame_type": "regular",
                "type": "flow",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to setup a new recurrent payment."),
        ],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
        ],
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()

    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events == []


async def test_action_run_slot_rejections_top_frame_none_rejections(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "4YL3KDBR",
                "flow_id": "setup_recurrent_payment",
                "step_id": "ask_payment_recipient",
                "frame_type": "regular",
                "type": "flow",
            },
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_recipient",
                "utter": "utter_ask_payment_recipient",
                "collect_action": "action_ask_payment_recipient",
                "rejections": [],
                "type": "pattern_collect_information",
            },
        ]
    )

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("I want to make a payment."),
        ],
        slots=[
            TextSlot("payment_recipient", mappings=[]),
        ],
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events == []


async def test_action_run_slot_rejections_top_frame_slot_not_been_set(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
    rejection_test_dialogue_stack: DialogueStack,
    capsys: CaptureFixture,
) -> None:
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[UserUttered("i want to setup a new recurrent payment.")],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
        ],
    )
    tracker.update_stack(rejection_test_dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events == []
    out = capsys.readouterr().out
    assert "[debug    ] first.collect.slot.not.set" in out


async def test_action_run_slot_rejections_run_success(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
    rejection_test_dialogue_stack: DialogueStack,
) -> None:
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to setup an international transfer."),
            SlotSet("recurrent_payment_type", "international transfer"),
        ],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
        ],
    )
    tracker.update_stack(rejection_test_dialogue_stack)
    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events == [
        SlotSet("recurrent_payment_type", None),
        BotUttered(
            "Sorry, you requested an invalid recurrent payment type.",
            metadata={
                "utter_action": "utter_invalid_recurrent_payment_type",
                UTTER_SOURCE_METADATA_KEY: "TemplatedNaturalLanguageGenerator",
                ACTIVE_FLOW_METADATA_KEY: "setup_recurrent_payment",
                STEP_ID_METADATA_KEY: "ask_payment_type",
            },
        ),
    ]


@pytest.mark.parametrize(
    "predicate",
    [None, "slots.recurrent_payment_type in {'direct debit', 'standing order'}"],
)
async def test_action_run_slot_rejections_internal_error(
    predicate: Optional[Text],
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
    capsys: CaptureFixture,
) -> None:
    """Test that an invalid or None predicate dispatches an internal error utterance."""
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "4YL3KDBR",
                "flow_id": "setup_recurrent_payment",
                "step_id": "ask_payment_type",
                "frame_type": "regular",
                "type": "flow",
            },
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "recurrent_payment_type",
                "utter": "utter_ask_recurrent_payment_type",
                "collect_action": "action_ask_recurrent_payment_type",
                "rejections": [
                    {
                        "if": predicate,
                        "utter": "utter_invalid_recurrent_payment_type",
                    }
                ],
                "type": "pattern_collect_information",
            },
        ]
    )

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to setup a new recurrent payment."),
            SlotSet("recurrent_payment_type", "international transfer"),
        ],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
        ],
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events[0] == SlotSet("recurrent_payment_type", None)
    assert isinstance(events[1], BotUttered)
    assert events[1].text == "Sorry, something went wrong."
    assert events[1].metadata == {
        "utter_action": "utter_internal_error_rasa",
        UTTER_SOURCE_METADATA_KEY: "TemplatedNaturalLanguageGenerator",
        ACTIVE_FLOW_METADATA_KEY: "setup_recurrent_payment",
        STEP_ID_METADATA_KEY: "ask_payment_type",
    }

    out = capsys.readouterr().out
    assert "[error    ] run.predicate.error" in out
    assert f"predicate={predicate}" in out


async def test_action_run_slot_rejections_collect_missing_utter(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
    capsys: CaptureFixture,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "4YL3KDBR",
                "flow_id": "setup_recurrent_payment",
                "step_id": "ask_payment_type",
                "frame_type": "regular",
                "type": "flow",
            },
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "recurrent_payment_type",
                "utter": "utter_ask_recurrent_payment_type",
                "collect_action": "action_ask_recurrent_payment_type",
                "rejections": [
                    {
                        "if": 'not ({"direct debit" "standing order"} '
                        "contains slots.recurrent_payment_type)",
                        "utter": None,
                    }
                ],
                "type": "pattern_collect_information",
            },
        ]
    )

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to setup a new recurrent payment."),
            SlotSet("recurrent_payment_type", "international transfer"),
        ],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
        ],
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events == [SlotSet("recurrent_payment_type", None)]

    out = capsys.readouterr().out
    assert "[error    ] run.rejection.missing.utter" in out
    assert "utterance=None" in out


async def test_action_run_slot_rejections_not_found_utter(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
    capsys: CaptureFixture,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "4YL3KDBR",
                "flow_id": "setup_recurrent_payment",
                "step_id": "ask_payment_type",
                "frame_type": "regular",
                "type": "flow",
            },
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "recurrent_payment_type",
                "utter": "utter_ask_recurrent_payment_type",
                "collect_action": "action_ask_recurrent_payment_type",
                "rejections": [
                    {
                        "if": 'not ({"direct debit" "standing order"} '
                        "contains slots.recurrent_payment_type)",
                        "utter": "utter_not_found",
                    }
                ],
                "type": "pattern_collect_information",
            },
        ]
    )

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to setup a new recurrent payment."),
            SlotSet("recurrent_payment_type", "international transfer"),
        ],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
        ],
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events == [SlotSet("recurrent_payment_type", None)]

    out = capsys.readouterr().out
    assert "[error    ] run.rejection.failed.finding.utter" in out
    assert "utterance=utter_not_found" in out


async def test_action_run_slot_rejections_pass_multiple_rejection_checks(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
    capsys: CaptureFixture,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "4YL3KDBR",
                "flow_id": "setup_recurrent_payment",
                "step_id": "ask_payment_amount",
                "frame_type": "regular",
                "type": "flow",
            },
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_amount",
                "utter": "utter_ask_payment_amount",
                "collect_action": "action_ask_payment_amount",
                "rejections": [
                    {
                        "if": "slots.payment_amount > 1000",
                        "utter": "utter_payment_too_high",
                    },
                    {
                        "if": "slots.payment_amount < 0",
                        "utter": "utter_payment_negative",
                    },
                ],
                "type": "pattern_collect_information",
            },
        ]
    )

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to transfer £500."),
            SlotSet("payment_amount", 500.0),
        ],
        slots=[
            FloatSlot("payment_amount", mappings=[]),
        ],
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events == []
    assert tracker.get_slot("payment_amount") == 500.0


async def test_action_run_slot_rejections_fails_multiple_rejection_checks(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
    capsys: CaptureFixture,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "4YL3KDBR",
                "flow_id": "setup_recurrent_payment",
                "step_id": "ask_payment_amount",
                "frame_type": "regular",
                "type": "flow",
            },
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_amount",
                "utter": "utter_ask_payment_amount",
                "collect_action": "action_ask_payment_amount",
                "rejections": [
                    {
                        "if": "slots.payment_amount > 1000",
                        "utter": "utter_payment_too_high",
                    },
                    {
                        "if": "slots.payment_amount < 0",
                        "utter": "utter_payment_negative",
                    },
                ],
                "type": "pattern_collect_information",
            },
        ]
    )

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to transfer $-100."),
            SlotSet("payment_amount", -100),
        ],
        slots=[
            FloatSlot("payment_amount", mappings=[]),
        ],
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events == [
        SlotSet("payment_amount", None),
        BotUttered(
            "Sorry, the amount cannot be negative.",
            metadata={
                "utter_action": "utter_payment_negative",
                UTTER_SOURCE_METADATA_KEY: "TemplatedNaturalLanguageGenerator",
                ACTIVE_FLOW_METADATA_KEY: "setup_recurrent_payment",
                STEP_ID_METADATA_KEY: "ask_payment_amount",
            },
        ),
    ]


async def test_invalid_categorical_slot_using_coercion(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_execution_mode",
                "utter": "utter_ask_payment_execution_mode",
                "collect_action": "action_ask_payment_execution_mode",
                "type": "pattern_collect_information",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to make a fast payment"),
            SlotSet("payment_execution_mode", "fast"),
        ],
        slots=rejection_test_domain.slots,
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events == [
        SlotSet("payment_execution_mode", None),
        BotUttered(
            "Sorry, you requested an option that is not valid.",
            metadata={
                "utter_action": "utter_categorical_slot_rejection",
                UTTER_SOURCE_METADATA_KEY: "TemplatedNaturalLanguageGenerator",
                ACTIVE_FLOW_METADATA_KEY: "pattern_collect_information",
                STEP_ID_METADATA_KEY: "start",
            },
        ),
    ]


async def test_valid_categorical_slot(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_execution_mode",
                "utter": "utter_ask_payment_execution_mode",
                "collect_action": "action_ask_payment_execution_mode",
                "type": "pattern_collect_information",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to make an immediate payment"),
            SlotSet("payment_execution_mode", "immediate"),
        ],
        slots=rejection_test_domain.slots,
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events == []


async def test_invalid_boolean_slot_using_coercion(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_confirmation",
                "utter": "utter_ask_payment_confirmation",
                "collect_action": "action_ask_payment_confirmation",
                "type": "pattern_collect_information",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("maybe"),
            SlotSet("payment_confirmation", "maybe"),
        ],
        slots=rejection_test_domain.slots,
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert len(events) == 2
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "payment_confirmation"
    assert events[0].value is None
    assert isinstance(events[1], BotUttered)
    assert events[1].text == "Sorry, the option you provided, maybe, is not valid."
    assert events[1].metadata["utter_action"] == "utter_boolean_slot_rejection"


async def test_valid_boolean_slot_coercion_changes_value(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_confirmation",
                "utter": "utter_ask_payment_confirmation",
                "collect_action": "action_ask_payment_confirmation",
                "type": "pattern_collect_information",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("no"),
            SlotSet("payment_confirmation", "no"),
        ],
        slots=rejection_test_domain.slots,
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert len(events) == 1
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "payment_confirmation"
    assert events[0].value is False


async def test_valid_boolean_slot(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_confirmation",
                "utter": "utter_ask_payment_confirmation",
                "collect_action": "action_ask_payment_confirmation",
                "type": "pattern_collect_information",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("false"),
            SlotSet("payment_confirmation", False),
        ],
        slots=rejection_test_domain.slots,
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )
    assert events == []


async def test_invalid_float_slot_using_coercion(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_amount",
                "utter": "utter_ask_payment_amount",
                "collect_action": "action_ask_payment_amount",
                "type": "pattern_collect_information",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("junk"),
            SlotSet("payment_amount", "junk"),
        ],
        slots=rejection_test_domain.slots,
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert len(events) == 2
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "payment_amount"
    assert events[0].value is None
    assert isinstance(events[1], BotUttered)
    assert events[1].text == "Sorry, the number you provided, junk, is not valid."
    assert events[1].metadata["utter_action"] == "utter_float_slot_rejection"


async def test_valid_float_slot_coercion_changes_value(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_amount",
                "utter": "utter_ask_payment_amount",
                "collect_action": "action_ask_payment_amount",
                "type": "pattern_collect_information",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("40"),
            SlotSet("payment_amount", 40),
        ],
        slots=rejection_test_domain.slots,
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert len(events) == 1
    assert isinstance(events[0], SlotSet)
    assert events[0].key == "payment_amount"
    assert events[0].value == 40.0


async def test_valid_float_slot(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_amount",
                "utter": "utter_ask_payment_amount",
                "collect_action": "action_ask_payment_amount",
                "type": "pattern_collect_information",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("40.5"),
            SlotSet("payment_amount", 40.5),
        ],
        slots=rejection_test_domain.slots,
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )
    assert events == []


async def test_action_run_slot_rejections_with_text_slot(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_recipient",
                "utter": "utter_payment_recipient",
                "collect_action": "action_ask_payment_recipient",
                "type": "pattern_collect_information",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("Joohn"),
            SlotSet("payment_recipient", "Jooohn"),
        ],
        slots=rejection_test_domain.slots,
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )
    assert events == []


async def test_action_run_slot_rejections_with_existing_slot_set_to_none(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_collect_information",
                "step_id": "start",
                "collect": "payment_recipient",
                "utter": "utter_payment_recipient",
                "collect_action": "action_ask_payment_recipient",
                "type": "pattern_collect_information",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("junk"),
            SlotSet("payment_recipient", None),
        ],
        slots=rejection_test_domain.slots,
    )
    tracker.update_stack(dialogue_stack)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )
    assert events == []


@pytest.mark.parametrize(
    "slot_name, slot, slot_value, expected_output",
    [
        ("some_other_slot", FloatSlot("some_float", []), None, None),
        ("some_float", FloatSlot("some_float", []), 40, 40.0),
        ("some_float", FloatSlot("some_float", []), 40.0, 40.0),
        ("some_text", TextSlot("some_text", []), "fourty", "fourty"),
        ("some_bool", BooleanSlot("some_bool", []), "True", True),
        ("some_bool", BooleanSlot("some_bool", []), "false", False),
        ("invalid_float", FloatSlot("invalid_float", []), "40.0.0", None),
        ("invalid_bool", BooleanSlot("invalid_bool", []), "maybe", None),
        (
            "valid_categ",
            CategoricalSlot("valid_categ", [{}], ["option1", "option2"]),
            "option1",
            "option1",
        ),
        (
            "invalid_categ",
            CategoricalSlot("invalid_categ", [{}], ["option1", "option2"]),
            "junk_val",
            None,
        ),
    ],
)
async def test_coerce_slot_value(
    slot_name: str,
    slot: Slot,
    slot_value: Any,
    expected_output: Any,
) -> None:
    """Test that coerce_slot_value coerces the slot value correctly."""
    # Given
    tracker = DialogueStateTracker.from_events("test", evts=[], slots=[slot])
    # When
    coerced_value = coerce_slot_value(slot_value, slot_name, tracker)
    # Then
    assert coerced_value == expected_output


@pytest.mark.parametrize(
    "slot, expected_output",
    [
        (BooleanSlot("some_bool", []), "utter_boolean_slot_rejection"),
        (FloatSlot("some_float", []), "utter_float_slot_rejection"),
        (CategoricalSlot("some_categ", [{}]), "utter_categorical_slot_rejection"),
        (TextSlot("some_text", []), None),
    ],
)
async def test_utterance_for_slot_type(slot: Slot, expected_output: str) -> None:
    assert utterance_for_slot_type(slot) == expected_output


async def test_rephrased_bot_utterance_contains_metadata_keys(
    default_channel: OutputChannel,
    rejection_test_domain: Domain,
    rejection_test_dialogue_stack: DialogueStack,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in action_run_slot_rejections")
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to setup an international transfer."),
            SlotSet("recurrent_payment_type", "international transfer"),
        ],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
        ],
    )
    tracker.update_stack(rejection_test_dialogue_stack)

    message = (
        "The payment type you requested in invalid, I cannot proceed with this request."
    )

    async def mock_rephrase(*args, **kwargs) -> Dict[str, Any]:
        return {KEY_RESPONSES_TEXT: message}

    mock_contextual_rephraser = ContextualResponseRephraser(
        EndpointConfig(), rejection_test_domain
    )
    monkeypatch.setattr(mock_contextual_rephraser, "rephrase", mock_rephrase)

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=mock_contextual_rephraser,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events == [
        SlotSet("recurrent_payment_type", None),
        BotUttered(
            message,
            data={
                "elements": None,
                "quick_replies": None,
                "buttons": None,
                "attachment": None,
                "image": None,
                "custom": None,
            },
            metadata={
                "utter_action": "utter_invalid_recurrent_payment_type",
                UTTER_SOURCE_METADATA_KEY: "ContextualResponseRephraser",
                ACTIVE_FLOW_METADATA_KEY: "setup_recurrent_payment",
                STEP_ID_METADATA_KEY: "ask_payment_type",
                DOMAIN_GROUND_TRUTH_METADATA_KEY: [
                    response["text"]
                    for response in rejection_test_domain.responses.get(
                        "utter_invalid_recurrent_payment_type"
                    )
                ],
            },
        ),
    ]
