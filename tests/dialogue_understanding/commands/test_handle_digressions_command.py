from typing import Dict, List, Set

import pytest

from rasa.dialogue_understanding.commands.handle_digressions_command import (
    HandleDigressionsCommand,
)
from rasa.dialogue_understanding.patterns.cannot_handle import (
    CannotHandlePatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.handle_digressions import (
    FLOW_PATTERN_HANDLE_DIGRESSIONS,
)
from rasa.shared.core.events import DialogueStackUpdated
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence
from rasa.shared.core.flows.steps import CollectInformationFlowStep
from rasa.shared.core.flows.utils import ALL_LABEL
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import HANDLE_DIGRESSIONS_COMMAND
from tests.utilities import flows_from_str


def test_command_name() -> None:
    command = HandleDigressionsCommand(flow="flow")
    assert command.command() == HANDLE_DIGRESSIONS_COMMAND


def test_command_from_dict() -> None:
    command = HandleDigressionsCommand.from_dict({"flow": "flow"})
    assert command == HandleDigressionsCommand(flow="flow")


@pytest.mark.parametrize("data", [{}, {"extra": "extra"}])
def test_from_dict_fails_if_missing_flow_parameter(data: Dict) -> None:
    with pytest.raises(ValueError):
        HandleDigressionsCommand.from_dict(data)


def test_run_command_on_tracker_flow_not_in_available_flows():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = HandleDigressionsCommand(flow="flow")

    events = command.run_command_on_tracker(tracker, FlowsList([]), tracker)
    assert len(events) == 1
    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    dialogue_stack_event.apply_to(tracker)
    assert isinstance(tracker.stack.top(), CannotHandlePatternFlowStackFrame)


def test_run_command_on_tracker_empty_stack():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = HandleDigressionsCommand(flow="flow")

    assert (
        command.run_command_on_tracker(tracker, FlowsList([Flow(id="flow")]), tracker)
        == []
    )


def test_run_command_on_tracker_flow_already_on_the_stack_not_at_a_collect_step():
    patch = """
    [{"op": "add", "path": "/0", "value":
    {"frame_id": "flow-frame-id", "flow_id": "flow",
    "step_id": "some-step-id", "type": "flow"}}]
    """
    tracker = DialogueStateTracker.from_events(
        "test", evts=[DialogueStackUpdated(patch)]
    )
    command = HandleDigressionsCommand(flow="flow")

    assert (
        command.run_command_on_tracker(tracker, FlowsList([Flow(id="flow")]), tracker)
        == []
    )


@pytest.mark.parametrize(
    "ask_confirm_digressions, block_digressions, "
    "expected_ask_confirm_digressions, expected_block_digressions",
    [
        ([ALL_LABEL], [], {"interrupting_flow"}, set()),
        ([], [ALL_LABEL], set(), {"interrupting_flow"}),
        (["interrupting_flow"], [], {"interrupting_flow"}, set()),
        ([], ["interrupting_flow"], set(), {"interrupting_flow"}),
        (["other-flow"], ["interrupting_flow"], {"other-flow"}, {"interrupting_flow"}),
        (["interrupting_flow"], ["other-flow"], {"interrupting_flow"}, {"other-flow"}),
    ],
)
def test_run_command_on_tracker_at_a_collect_step(
    ask_confirm_digressions: List[str],
    block_digressions: List[str],
    expected_ask_confirm_digressions: Set[str],
    expected_block_digressions: Set[str],
) -> None:
    patch = """
    [{"op": "add", "path": "/0", "value":
    {"frame_id": "flow-frame-id", "flow_id": "current_flow",
    "step_id": "collect-some-slot", "type": "flow"}}]
    """
    tracker = DialogueStateTracker.from_events(
        "test", evts=[DialogueStackUpdated(patch)]
    )
    command = HandleDigressionsCommand(flow="interrupting_flow")
    flow_steps = FlowStepSequence(
        [
            CollectInformationFlowStep(
                custom_id="collect-some-slot",
                utter="utter_ask_some_slot",
                collect="some_slot",
                collect_action="",
                rejections=[],
                ask_confirm_digressions=ask_confirm_digressions,
                block_digressions=block_digressions,
                idx=0,
                metadata={},
                next=FlowStepLinks.from_json("flow", "END"),
                flow_id="current_flow",
                description="",
            )
        ]
    )
    flows = FlowsList(
        [
            Flow(id="current_flow", step_sequence=flow_steps),
            Flow(id="interrupting_flow"),
            Flow(id="other-flow"),
        ]
    )

    events = command.run_command_on_tracker(tracker, flows, tracker)
    assert len(events) == 1
    dialogue_stack_updated = events[0]
    assert isinstance(dialogue_stack_updated, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_updated.update)
    assert len(updated_stack.frames) == 2

    current_frame = updated_stack.frames[0]
    assert current_frame.flow_id == "current_flow"

    top_frame = updated_stack.frames[1]
    assert top_frame.flow_id == FLOW_PATTERN_HANDLE_DIGRESSIONS
    assert top_frame.interrupted_flow_id == "current_flow"
    assert top_frame.interrupted_step_id == "collect-some-slot"
    assert top_frame.interrupting_flow_id == "interrupting_flow"
    assert top_frame.ask_confirm_digressions == expected_ask_confirm_digressions
    assert top_frame.block_digressions == expected_block_digressions


@pytest.mark.parametrize(
    "ask_confirm_digressions, block_digressions, "
    "expected_ask_confirm_digressions, expected_block_digressions",
    [
        ([ALL_LABEL], [], {"interrupting_flow"}, set()),
        ([], [ALL_LABEL], set(), {"interrupting_flow"}),
        (["interrupting_flow"], [], {"interrupting_flow"}, set()),
        ([], ["interrupting_flow"], set(), {"interrupting_flow"}),
        (["other-flow"], ["interrupting_flow"], {"other-flow"}, {"interrupting_flow"}),
        (["interrupting_flow"], ["other-flow"], {"interrupting_flow"}, {"other-flow"}),
    ],
)
def test_run_command_on_tracker_with_flow_level_digression_properties(
    ask_confirm_digressions: List[str],
    block_digressions: List[str],
    expected_ask_confirm_digressions: Set[str],
    expected_block_digressions: Set[str],
) -> None:
    patch = """
    [{"op": "add", "path": "/0", "value":
    {"frame_id": "flow-frame-id", "flow_id": "current_flow",
    "step_id": "collect-some-slot", "type": "flow"}}]
    """
    tracker = DialogueStateTracker.from_events(
        "test", evts=[DialogueStackUpdated(patch)]
    )
    command = HandleDigressionsCommand(flow="interrupting_flow")
    flow_steps = FlowStepSequence(
        [
            CollectInformationFlowStep(
                custom_id="collect-some-slot",
                utter="utter_ask_some_slot",
                collect="some_slot",
                collect_action="",
                rejections=[],
                ask_confirm_digressions=[],
                block_digressions=[],
                idx=0,
                metadata={},
                next=FlowStepLinks.from_json("flow", "END"),
                flow_id="current_flow",
                description="",
            )
        ]
    )
    flows = FlowsList(
        [
            Flow(
                id="current_flow",
                step_sequence=flow_steps,
                ask_confirm_digressions=ask_confirm_digressions,
                block_digressions=block_digressions,
            ),
            Flow(id="interrupting_flow"),
            Flow(id="other-flow"),
        ]
    )

    events = command.run_command_on_tracker(tracker, flows, tracker)
    assert len(events) == 1
    dialogue_stack_updated = events[0]
    assert isinstance(dialogue_stack_updated, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_updated.update)
    assert len(updated_stack.frames) == 2

    current_frame = updated_stack.frames[0]
    assert current_frame.flow_id == "current_flow"

    top_frame = updated_stack.frames[1]
    assert top_frame.flow_id == FLOW_PATTERN_HANDLE_DIGRESSIONS
    assert top_frame.interrupted_flow_id == "current_flow"
    assert top_frame.interrupted_step_id == "collect-some-slot"
    assert top_frame.interrupting_flow_id == "interrupting_flow"
    assert top_frame.ask_confirm_digressions == expected_ask_confirm_digressions
    assert top_frame.block_digressions == expected_block_digressions


def test_run_command_on_tracker_merged_digression_properties() -> None:
    patch = """
    [{"op": "add", "path": "/0", "value":
    {"frame_id": "flow-frame-id", "flow_id": "current_flow",
    "step_id": "collect-some-slot", "type": "flow"}}]
    """
    tracker = DialogueStateTracker.from_events(
        "test", evts=[DialogueStackUpdated(patch)]
    )
    command = HandleDigressionsCommand(flow="interrupting_flow")
    flows = flows_from_str(
        """
        flows:
            current_flow:
                description: "This is the current flow"
                steps:
                    - id: collect-some-slot
                      collect: some_slot
                      ask_confirm_digressions:
                       - test_flow_digression
                      block_digressions:
                        - interrupting_flow
                ask_confirm_digressions:
                   - test_flow
                block_digressions:
                   - other_flow
            interrupting_flow:
                description: "This flow interrupts the current flow"
                steps:
                 - action: utter_welcome
            other_flow:
                description: "This is another flow"
                steps:
                 - action: utter_goodbye
            test_flow_digressions:
                description: "This flow digresses from the current flow"
                steps:
                 - action: utter_digression_1
            test_flow:
                description: "This flow digresses from the current flow"
                steps:
                 - action: utter_digression_2
    """
    )

    events = command.run_command_on_tracker(tracker, flows, tracker)
    assert len(events) == 1
    dialogue_stack_updated = events[0]
    assert isinstance(dialogue_stack_updated, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_updated.update)
    assert len(updated_stack.frames) == 2

    current_frame = updated_stack.frames[0]
    assert current_frame.flow_id == "current_flow"

    top_frame = updated_stack.frames[1]
    assert top_frame.flow_id == FLOW_PATTERN_HANDLE_DIGRESSIONS
    assert top_frame.interrupted_flow_id == "current_flow"
    assert top_frame.interrupted_step_id == "collect-some-slot"
    assert top_frame.interrupting_flow_id == "interrupting_flow"
    assert top_frame.ask_confirm_digressions == {"test_flow", "test_flow_digression"}
    assert top_frame.block_digressions == {"other_flow", "interrupting_flow"}
