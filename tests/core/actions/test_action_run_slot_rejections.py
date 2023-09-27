import uuid
from typing import Optional, Text

import pytest
from pytest import CaptureFixture

from rasa.core.actions.action_run_slot_rejections import ActionRunSlotRejections
from rasa.core.channels import OutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered
from rasa.shared.core.slots import AnySlot, FloatSlot, TextSlot
from rasa.shared.core.trackers import DialogueStateTracker


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
        responses:
            utter_ask_recurrent_payment_type:
             - text: "What type of recurrent payment do you want to setup?"
            utter_invalid_recurrent_payment_type:
             - text: "Sorry, you requested an invalid recurrent payment type."
            utter_internal_error_rasa:
             - text: "Sorry, something went wrong."
            utter_ask_payment_amount:
             - text: "What amount do you want to pay?"
            utter_payment_too_high:
             - text: "Sorry, the amount is above the maximum £1,000 allowed."
            utter_payment_negative:
             - text: "Sorry, the amount cannot be negative."
        """
    )


async def test_action_run_slot_rejections_top_frame_not_collect_information(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
) -> None:
    dialogue_stack = [
        {
            "frame_id": "4YL3KDBR",
            "flow_id": "setup_recurrent_payment",
            "step_id": "ask_payment_type",
            "frame_type": "regular",
            "type": "flow",
        },
    ]
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to setup a new recurrent payment."),
            SlotSet("dialogue_stack", dialogue_stack),
        ],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
            AnySlot("dialogue_stack", mappings=[]),
        ],
    )

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
    dialogue_stack = [
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
            "rejections": [],
            "type": "pattern_collect_information",
        },
    ]

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("I want to make a payment."),
            SlotSet("dialogue_stack", dialogue_stack),
        ],
        slots=[
            TextSlot("payment_recipient", mappings=[]),
            AnySlot("dialogue_stack", mappings=[]),
        ],
    )

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
    capsys: CaptureFixture,
) -> None:
    dialogue_stack = [
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
            "rejections": [
                {
                    "if": 'not ({"direct debit" "standing order"} contains recurrent_payment_type)',  # noqa: E501
                    "utter": "utter_invalid_recurrent_payment_type",
                }
            ],
            "type": "pattern_collect_information",
        },
    ]

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to setup a new recurrent payment."),
            SlotSet("dialogue_stack", dialogue_stack),
        ],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
            AnySlot("dialogue_stack", mappings=[]),
        ],
    )

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
) -> None:
    dialogue_stack = [
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
            "rejections": [
                {
                    "if": 'not ({"direct debit" "standing order"} contains recurrent_payment_type)',  # noqa: E501
                    "utter": "utter_invalid_recurrent_payment_type",
                }
            ],
            "type": "pattern_collect_information",
        },
    ]
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to setup an international transfer."),
            SlotSet("recurrent_payment_type", "international transfer"),
            SlotSet("dialogue_stack", dialogue_stack),
        ],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
            AnySlot("dialogue_stack", mappings=[]),
        ],
    )

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
            metadata={"utter_action": "utter_invalid_recurrent_payment_type"},
        ),
    ]


@pytest.mark.parametrize(
    "predicate", [None, "recurrent_payment_type in {'direct debit', 'standing order'}"]
)
async def test_action_run_slot_rejections_internal_error(
    predicate: Optional[Text],
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
    capsys: CaptureFixture,
) -> None:
    """Test that an invalid or None predicate dispatches an internal error utterance."""
    dialogue_stack = [
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
            "rejections": [
                {
                    "if": predicate,
                    "utter": "utter_invalid_recurrent_payment_type",
                }
            ],
            "type": "pattern_collect_information",
        },
    ]

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to setup a new recurrent payment."),
            SlotSet("recurrent_payment_type", "international transfer"),
            SlotSet("dialogue_stack", dialogue_stack),
        ],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
            AnySlot("dialogue_stack", mappings=[]),
        ],
    )

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
    assert events[1].metadata == {"utter_action": "utter_internal_error_rasa"}

    out = capsys.readouterr().out
    assert "[error    ] run.predicate.error" in out
    assert f"predicate={predicate}" in out


async def test_action_run_slot_rejections_collect_missing_utter(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
    capsys: CaptureFixture,
) -> None:
    dialogue_stack = [
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
            "rejections": [
                {
                    "if": 'not ({"direct debit" "standing order"} contains recurrent_payment_type)',  # noqa: E501
                    "utter": None,
                }
            ],
            "type": "pattern_collect_information",
        },
    ]

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to setup a new recurrent payment."),
            SlotSet("recurrent_payment_type", "international transfer"),
            SlotSet("dialogue_stack", dialogue_stack),
        ],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
            AnySlot("dialogue_stack", mappings=[]),
        ],
    )

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
    dialogue_stack = [
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
            "rejections": [
                {
                    "if": 'not ({"direct debit" "standing order"} contains recurrent_payment_type)',  # noqa: E501
                    "utter": "utter_not_found",
                }
            ],
            "type": "pattern_collect_information",
        },
    ]

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to setup a new recurrent payment."),
            SlotSet("recurrent_payment_type", "international transfer"),
            SlotSet("dialogue_stack", dialogue_stack),
        ],
        slots=[
            TextSlot("recurrent_payment_type", mappings=[]),
            AnySlot("dialogue_stack", mappings=[]),
        ],
    )

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
    dialogue_stack = [
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
            "rejections": [
                {
                    "if": "payment_amount > 1000",
                    "utter": "utter_payment_too_high",
                },
                {
                    "if": "payment_amount < 0",
                    "utter": "utter_payment_negative",
                },
            ],
            "type": "pattern_collect_information",
        },
    ]

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to transfer £500."),
            SlotSet("payment_amount", 500),
            SlotSet("dialogue_stack", dialogue_stack),
        ],
        slots=[
            FloatSlot("payment_amount", mappings=[]),
            AnySlot("dialogue_stack", mappings=[]),
        ],
    )

    action_run_slot_rejections = ActionRunSlotRejections()
    events = await action_run_slot_rejections.run(
        output_channel=default_channel,
        nlg=rejection_test_nlg,
        tracker=tracker,
        domain=rejection_test_domain,
    )

    assert events == []
    assert tracker.get_slot("payment_amount") == 500


async def test_action_run_slot_rejections_fails_multiple_rejection_checks(
    default_channel: OutputChannel,
    rejection_test_nlg: TemplatedNaturalLanguageGenerator,
    rejection_test_domain: Domain,
    capsys: CaptureFixture,
) -> None:
    dialogue_stack = [
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
            "rejections": [
                {
                    "if": "payment_amount > 1000",
                    "utter": "utter_payment_too_high",
                },
                {
                    "if": "payment_amount < 0",
                    "utter": "utter_payment_negative",
                },
            ],
            "type": "pattern_collect_information",
        },
    ]

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            UserUttered("i want to transfer $-100."),
            SlotSet("payment_amount", -100),
            SlotSet("dialogue_stack", dialogue_stack),
        ],
        slots=[
            FloatSlot("payment_amount", mappings=[]),
            AnySlot("dialogue_stack", mappings=[]),
        ],
    )

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
            metadata={"utter_action": "utter_payment_negative"},
        ),
    ]
