import uuid

import pytest

from rasa.core.actions.action_handle_digressions import (
    ActionBlockDigressions,
    ActionContinueDigression,
)
from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.patterns.continue_interrupted import (
    ContinueInterruptedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.handle_digressions import (
    FLOW_PATTERN_HANDLE_DIGRESSIONS,
)
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import FlowStackFrameType
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, DialogueStackUpdated, FlowInterrupted
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.fixture
def test_tracker() -> DialogueStateTracker:
    patch_1 = """
        [{"op": "add", "path": "/0", "value":
        {"frame_id": "flow-frame-id", "flow_id": "some-flow-id",
        "step_id": "collect_some_slot", "type": "flow"}}]"""
    patch_2 = f"""
        [{{"op": "add", "path": "/1", "value":
        {{"frame_id": "some-frame-id", "flow_id": "{FLOW_PATTERN_HANDLE_DIGRESSIONS}",
        "step_id": "branching", "type": "{FLOW_PATTERN_HANDLE_DIGRESSIONS}",
        "interrupted_flow_id": "some-flow-id",
        "interrupted_step_id": "collect_some_slot",
        "interrupting_flow_id": "some-interrupting-flow-id",
        "ask_confirm_digressions": [], "block_digressions": []}}}}]"""
    return DialogueStateTracker.from_events(
        uuid.uuid4().hex, [DialogueStackUpdated(patch_1), DialogueStackUpdated(patch_2)]
    )


async def test_action_block_digressions(test_tracker: DialogueStateTracker) -> None:
    action = ActionBlockDigressions()
    channel = CollectingOutputChannel()
    domain = Domain.from_yaml("""
    responses:
        utter_block_digressions:
        - text: "I'm sorry, I can't do that right now."
    """)
    nlg = TemplatedNaturalLanguageGenerator(responses=domain.responses)

    events = await action.run(channel, nlg, test_tracker, domain)

    assert len(events) == 2
    bot_uttered = events[1]
    assert isinstance(bot_uttered, BotUttered)
    assert bot_uttered.text == "I'm sorry, I can't do that right now."

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)
    updated_stack = test_tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 4
    interrupting_flow_frame = updated_stack.frames[0]
    assert isinstance(interrupting_flow_frame, UserFlowStackFrame)
    assert interrupting_flow_frame.flow_id == "some-interrupting-flow-id"

    continue_interrupted_flow_frame = updated_stack.frames[1]
    assert isinstance(
        continue_interrupted_flow_frame, ContinueInterruptedPatternFlowStackFrame
    )
    assert (
        continue_interrupted_flow_frame.previous_flow_name
        == "some-interrupting-flow-id"
    )

    interrupted_flow_id = updated_stack.frames[2]
    assert isinstance(interrupted_flow_id, UserFlowStackFrame)
    assert interrupted_flow_id.flow_id == "some-flow-id"
    assert interrupted_flow_id.step_id == "collect_some_slot"


async def test_action_continue_digressions(test_tracker: DialogueStateTracker) -> None:
    action = ActionContinueDigression()
    channel = CollectingOutputChannel()
    domain = Domain.from_yaml("""
    responses:
        utter_continue_interruption:
        - text: "Let's continue with the chosen topic instead."
    """)
    nlg = TemplatedNaturalLanguageGenerator(responses=domain.responses)

    events = await action.run(channel, nlg, test_tracker, domain)

    assert len(events) == 3
    assert events[0] == FlowInterrupted(
        flow_id="some-flow-id", step_id="collect_some_slot"
    )
    bot_uttered = events[2]
    assert isinstance(bot_uttered, BotUttered)
    assert bot_uttered.text == "Let's continue with the chosen topic instead."

    dialogue_stack_event = events[1]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)
    updated_stack = test_tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 3

    interrupted_flow_id = updated_stack.frames[0]
    assert isinstance(interrupted_flow_id, UserFlowStackFrame)
    assert interrupted_flow_id.flow_id == "some-flow-id"
    assert interrupted_flow_id.step_id == "collect_some_slot"

    interrupting_flow_frame = updated_stack.frames[2]
    assert isinstance(interrupting_flow_frame, UserFlowStackFrame)
    assert interrupting_flow_frame.flow_id == "some-interrupting-flow-id"
    assert interrupting_flow_frame.frame_type == FlowStackFrameType.INTERRUPT
