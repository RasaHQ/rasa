import uuid

import pytest
from pytest import CaptureFixture

from rasa.core.actions.action_run_slot_rejections import ActionRunSlotRejections
from rasa.core.channels import OutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered
from rasa.shared.core.slots import AnySlot, TextSlot
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
        responses:
            utter_ask_recurrent_payment_type:
             - text: "What type of recurrent payment do you want to setup?"
            utter_invalid_recurrent_payment_type:
             - text: "Sorry, you requested an invalid recurrent payment type."
            utter_internal_error_rasa:
             - text: "Sorry, something went wrong."
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
            "flow_id": "pattern_ask_collect_information",
            "step_id": "start",
            "collect_information": "payment_recipient",
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
            "flow_id": "pattern_ask_collect_information",
            "step_id": "start",
            "collect_information": "recurrent_payment_type",
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
    assert (
        "[debug    ] first.collect.slot.not.set     slot_name=recurrent_payment_type slot_value=None"  # noqa: E501
        in out
    )


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
            "flow_id": "pattern_ask_collect_information",
            "step_id": "start",
            "collect_information": "recurrent_payment_type",
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


async def test_action_run_slot_rejections_top_frame_slot_internal_error(
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
            "flow_id": "pattern_ask_collect_information",
            "step_id": "start",
            "collect_information": "recurrent_payment_type",
            "rejections": [
                {
                    "if": None,
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
    assert "[error    ] collect.predicate.error" in out
    assert "predicate=None" in out


async def test_action_run_slot_rejections_top_frame_slot_missing_utter(
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
            "flow_id": "pattern_ask_collect_information",
            "step_id": "start",
            "collect_information": "recurrent_payment_type",
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
    assert "[error    ] collect.rejection.missing.utter" in out
    assert "utterance=None" in out


async def test_action_run_slot_rejections_top_frame_slot_not_found_utter(
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
            "flow_id": "pattern_ask_collect_information",
            "step_id": "start",
            "collect_information": "recurrent_payment_type",
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
    assert "[debug    ] collect.rejection.failed.finding.utter" in out
    assert "utterance=utter_not_found" in out
