import pytest

from rasa.core.actions.action_clean_stack import ActionCleanStack
from rasa.core.channels import OutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.patterns.clarify import ClarifyPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.code_change import CodeChangeFlowStackFrame
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import FlowStackFrameType
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    BotUttered,
    DialogueStackUpdated,
    FlowStarted,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.fixture
def clean_stack_domain() -> Domain:
    domain_yaml = f"""
    version: '{LATEST_TRAINING_DATA_FORMAT_VERSION}'

    slots:
        additional_information:
            type: text
    responses:
        utter_ask_additional_information:
           - text: "Could you please provide additional information?"
    """
    return Domain.from_yaml(domain_yaml)


async def test_action_clean_stack_run_from_pattern_code_change(
    default_channel: OutputChannel,
    default_nlg: TemplatedNaturalLanguageGenerator,
    clean_stack_domain: Domain,
) -> None:
    tracker = DialogueStateTracker.from_events(
        "test_clean_stack_tracker",
        evts=[UserUttered("I have not received my order yet.")],
        slots=clean_stack_domain.slots,
    )

    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "XYZ12345",
                "flow_id": "order_complaint",
                "step_id": "start",
                "type": "flow",
                "frame_type": "regular",
            },
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_code_change",
                "step_id": "start",
                "type": "pattern_code_change",
            },
        ]
    )

    tracker.update_stack(dialogue_stack)

    action = ActionCleanStack()

    events = await action.run(default_channel, default_nlg, tracker, clean_stack_domain)

    assert len(events) == 1
    assert isinstance(events[0], DialogueStackUpdated)

    tracker.apply_stack_update(events[0].update)

    assert tracker.stack.frames == [
        UserFlowStackFrame(
            frame_id="XYZ12345",
            flow_id="order_complaint",
            step_id="NEXT:END",
            frame_type=FlowStackFrameType.REGULAR,
        ),
        CodeChangeFlowStackFrame(
            frame_id="6Z7PSTRM", flow_id="pattern_code_change", step_id="NEXT:END"
        ),
    ]


async def test_action_clean_stack_run_from_user_flow(
    default_channel: OutputChannel,
    default_nlg: TemplatedNaturalLanguageGenerator,
    clean_stack_domain: Domain,
) -> None:
    tracker = DialogueStateTracker.from_events(
        "test_clean_stack_tracker",
        evts=[
            UserUttered("I have not received my order yet."),
            BotUttered("Would you like to check the status or file a complaint?"),
            UserUttered("Check the status."),
            BotUttered("What is your order number?"),
            UserUttered("901234"),
            BotUttered("Your order will arrive today by 5 PM."),
            FlowStarted(flow_id="end_conversation"),
        ],
        slots=clean_stack_domain.slots,
    )

    dialogue_stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "6Z7PSTRM",
                "flow_id": "pattern_clarification",
                "step_id": "collect_additional_information",
                "type": "pattern_clarification",
                "names": ["order_status", "order_complaint"],
                "clarification_options": "order_status or order_complaint",
            },
            {
                "frame_id": "XYZ12345",
                "flow_id": "pattern_collect_information",
                "step_id": "ask_listen",
                "collect": "additional_information",
                "utter": "utter_ask_additional_information",
                "type": "pattern_collect_information",
                "collect_action": "",
                "rejections": [],
            },
            {
                "frame_id": "ABCD6789",
                "flow_id": "end_conversation",
                "step_id": "clean_stack",
                "type": "flow",
                "frame_type": "link",
            },
        ]
    )

    tracker.update_stack(dialogue_stack)

    action = ActionCleanStack()

    events = await action.run(default_channel, default_nlg, tracker, clean_stack_domain)

    assert len(events) == 1
    assert isinstance(events[0], DialogueStackUpdated)

    tracker.apply_stack_update(events[0].update)

    frames = tracker.stack.frames
    assert len(frames) == 3
    assert isinstance(frames[0], UserFlowStackFrame)
    assert frames[0].flow_id == "end_conversation"
    assert isinstance(frames[1], ClarifyPatternFlowStackFrame)
    assert isinstance(frames[2], CollectInformationPatternFlowStackFrame)
