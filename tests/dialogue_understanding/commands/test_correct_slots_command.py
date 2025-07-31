from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import jsonpatch
import pytest

from rasa.dialogue_understanding.commands.correct_slots_command import (
    CorrectedSlot,
    CorrectSlotsCommand,
)
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.correction import (
    CorrectionPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import DialogueStackFrame
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.pattern_frame import PatternFlowStackFrame
from rasa.shared.core.constants import SetSlotExtractor
from rasa.shared.core.events import DialogueStackUpdated, Event, SlotSet
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.flow import Flow
from rasa.shared.core.flows.steps import CollectInformationFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from tests.dialogue_understanding.conftest import update_tracker_with_path_through_flow
from tests.utilities import (
    flows_from_str,
    flows_from_str_including_defaults,
    flows_from_str_with_defaults,
)


@pytest.mark.parametrize(
    "slot_name, slot_value, expected_result, flow_id, collect_steps_config, "
    "stack_frames",
    [
        (
            # slot is in collect step of active flow
            "user_name",
            "John",
            True,
            "test_flow",
            [{"collect": "user_name", "custom_id": "collect_name"}],
            [UserFlowStackFrame(flow_id="test_flow", step_id="collect_name")],
        ),
        (
            # slot is not in any collect step
            "invalid_slot",
            "some_value",
            False,
            "test_flow",
            [
                {"collect": "user_name", "custom_id": "collect_name"},
                {"collect": "email", "custom_id": "collect_email"},
            ],
            [UserFlowStackFrame(flow_id="test_flow", step_id="collect_name")],
        ),
        (
            # flow has no collect steps
            "user_name",
            "John",
            False,
            "test_flow",
            [],
            [UserFlowStackFrame(flow_id="test_flow", step_id="some_step")],
        ),
        (
            # flow not found in list of flows
            "user_name",
            "John",
            False,
            None,
            [{"collect": "user_name", "custom_id": "collect_name"}],
            [UserFlowStackFrame(flow_id="nonexistent_flow", step_id="some_step")],
        ),
        (
            # mixed stack content with valid slot
            "user_name",
            "John",
            True,
            "test_flow",
            [{"collect": "user_name", "custom_id": "collect_name"}],
            [
                PatternFlowStackFrame(flow_id="pattern_flow", step_id="pattern_step"),
                UserFlowStackFrame(flow_id="test_flow", step_id="collect_name"),
            ],
        ),
        (
            # call and link frames with valid slot
            "user_name",
            "John",
            True,
            "test_flow",
            [{"collect": "user_name", "custom_id": "collect_name"}],
            [
                UserFlowStackFrame(
                    flow_id="test_flow",
                    step_id="collect_name",
                    frame_type=FlowStackFrameType.CALL,
                ),
                UserFlowStackFrame(
                    flow_id="another_flow",
                    step_id="some_step",
                    frame_type=FlowStackFrameType.LINK,
                ),
            ],
        ),
    ],
)
def test_should_correct_slot_parametrized(
    slot_name: str,
    slot_value: str,
    expected_result: bool,
    flow_id: Optional[str],
    collect_steps_config: List[Dict[str, Any]],
    stack_frames: List[DialogueStackFrame],
):
    """Parametrized test for should_correct_slot method covering various scenarios."""
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name=slot_name, value=slot_value)]
    )

    # Create flow based on config
    flow = Mock(spec=Flow)
    if flow_id is None:
        flow = None
    else:
        collect_steps = []
        for i, step_config in enumerate(collect_steps_config):
            collect_step = CollectInformationFlowStep(
                collect=step_config["collect"],
                utter=f"utter_ask_{step_config['collect']}",
                collect_action=f"action_ask_{step_config['collect']}",
                rejections=[],
                custom_id=step_config["custom_id"],
                idx=i,
                description=None,
                metadata={},
                next=Mock(),
                flow_id=flow_id,
            )
            collect_steps.append(collect_step)
        flow.get_collect_steps.return_value = collect_steps

    # Create flows list
    flows = Mock(spec=FlowsList)
    flows.flow_by_id.return_value = flow

    # Create stack
    stack = DialogueStack(frames=stack_frames)

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Act
    result = command.should_correct_slot(
        CorrectedSlot(name=slot_name, value=slot_value), tracker, flows
    )

    # Assert
    assert result is expected_result


@pytest.mark.parametrize(
    "slot_name, slot_value, stack_frames",
    [
        (
            # no user flows on the stack
            "user_name",
            "John",
            [PatternFlowStackFrame(flow_id="pattern_flow", step_id="pattern_step")],
        ),
        (
            # stack is empty
            "user_name",
            "John",
            [],
        ),
    ],
)
def test_should_correct_slot_denied_no_flows(
    slot_name: str,
    slot_value: str,
    stack_frames: List[DialogueStackFrame],
):
    """Test that slot correction is denied when no user flows are available."""
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name=slot_name, value=slot_value)]
    )

    # Create flows list
    flows = Mock(spec=FlowsList)

    # Create stack
    if not stack_frames:
        stack = DialogueStack.empty()
    else:
        stack = DialogueStack(frames=stack_frames)

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Act
    result = command.should_correct_slot(
        CorrectedSlot(name=slot_name, value=slot_value), tracker, flows
    )

    # Assert
    assert result is False
    flows.flow_by_id.assert_not_called()


# Keep the original individual tests for backward compatibility and specific edge cases
def test_should_correct_slot_when_slot_is_in_collect_step_of_active_flow():
    """Test that slot correction is allowed when slot is part of a collect
    step in an active flow.
    """
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="user_name", value="John")]
    )

    # Create a flow with a collect step for "user_name"
    flow = Mock(spec=Flow)
    collect_step = CollectInformationFlowStep(
        collect="user_name",
        utter="utter_ask_user_name",
        collect_action="action_ask_user_name",
        rejections=[],
        custom_id="collect_name",
        idx=0,
        description=None,
        metadata={},
        next=Mock(),
        flow_id="test_flow",
    )
    flow.get_collect_steps.return_value = [collect_step]

    # Create flows list with our flow
    flows = Mock(spec=FlowsList)
    flows.flow_by_id.return_value = flow

    # Create stack with the flow
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="test_flow", step_id="collect_name")]
    )

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Act
    result = command.should_correct_slot(
        CorrectedSlot(name="user_name", value="John"), tracker, flows
    )

    # Assert
    assert result is True
    flows.flow_by_id.assert_called_once_with("test_flow")


def test_should_correct_slot_when_slot_is_in_collect_step_of_multiple_flows():
    """Test that slot correction is allowed when slot is part of collect
    steps in multiple flows.
    """
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="email", value="test@example.com")]
    )

    # Create flows with collect steps for "email"
    flow1 = Mock(spec=Flow)
    collect_step1 = CollectInformationFlowStep(
        collect="email",
        utter="utter_ask_email",
        collect_action="action_ask_email",
        rejections=[],
        custom_id="collect_email",
        idx=0,
        description=None,
        metadata={},
        next=Mock(),
        flow_id="flow1",
    )
    flow1.get_collect_steps.return_value = [collect_step1]

    flow2 = Mock(spec=Flow)
    collect_step2 = CollectInformationFlowStep(
        collect="email",
        utter="utter_ask_email",
        collect_action="action_ask_email",
        rejections=[],
        custom_id="collect_email_alt",
        idx=0,
        description=None,
        metadata={},
        next=Mock(),
        flow_id="flow2",
    )
    flow2.get_collect_steps.return_value = [collect_step2]

    # Create flows list
    flows = Mock(spec=FlowsList)
    flows.flow_by_id.side_effect = (
        lambda flow_id: flow1 if flow_id == "flow1" else flow2
    )

    # Create stack with multiple flows
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(flow_id="flow1", step_id="collect_email"),
            UserFlowStackFrame(flow_id="flow2", step_id="collect_email_alt"),
        ]
    )

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Act
    result = command.should_correct_slot(
        CorrectedSlot(name="email", value="test@example.com"), tracker, flows
    )

    # Assert
    assert result is True
    assert flows.flow_by_id.call_count == 1


def test_should_correct_slot_when_slot_is_not_in_any_collect_step():
    """Test that slot correction is denied when slot is not part of
    any collect step.
    """
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="invalid_slot", value="some_value")]
    )

    # Create a flow with collect steps for different slots
    flow = Mock(spec=Flow)
    collect_step1 = CollectInformationFlowStep(
        collect="user_name",
        utter="utter_ask_user_name",
        collect_action="action_ask_user_name",
        rejections=[],
        custom_id="collect_name",
        idx=0,
        description=None,
        metadata={},
        next=Mock(),
        flow_id="test_flow",
    )
    collect_step2 = CollectInformationFlowStep(
        collect="email",
        utter="utter_ask_email",
        collect_action="action_ask_email",
        rejections=[],
        custom_id="collect_email",
        idx=1,
        description=None,
        metadata={},
        next=Mock(),
        flow_id="test_flow",
    )
    flow.get_collect_steps.return_value = [collect_step1, collect_step2]

    # Create flows list
    flows = Mock(spec=FlowsList)
    flows.flow_by_id.return_value = flow

    # Create stack with the flow
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="test_flow", step_id="collect_name")]
    )

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Act
    result = command.should_correct_slot(
        CorrectedSlot(name="invalid_slot", value="some_value"), tracker, flows
    )

    # Assert
    assert result is False


def test_should_correct_slot_when_no_user_flows_on_stack():
    """Test that slot correction is denied when no user flows are on the stack."""
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="user_name", value="John")]
    )

    # Create stack with only pattern frames (no user flows)
    stack = DialogueStack(
        frames=[PatternFlowStackFrame(flow_id="pattern_flow", step_id="pattern_step")]
    )

    # Create flows list
    flows = Mock(spec=FlowsList)

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Act
    result = command.should_correct_slot(
        CorrectedSlot(name="user_name", value="John"), tracker, flows
    )

    # Assert
    assert result is False
    flows.flow_by_id.assert_not_called()


def test_should_correct_slot_when_stack_is_empty():
    """Test that slot correction is denied when stack is empty."""
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="user_name", value="John")]
    )

    # Create empty stack
    stack = DialogueStack.empty()

    # Create flows list
    flows = Mock(spec=FlowsList)

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Act
    result = command.should_correct_slot(
        CorrectedSlot(name="user_name", value="John"), tracker, flows
    )

    # Assert
    assert result is False
    flows.flow_by_id.assert_not_called()


def test_should_correct_slot_when_flow_has_no_collect_steps():
    """Test that slot correction is denied when flow has no collect steps."""
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="user_name", value="John")]
    )

    # Create a flow with no collect steps
    flow = Mock(spec=Flow)
    flow.get_collect_steps.return_value = []

    # Create flows list
    flows = Mock(spec=FlowsList)
    flows.flow_by_id.return_value = flow

    # Create stack with the flow
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="test_flow", step_id="some_step")]
    )

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Act
    result = command.should_correct_slot(
        CorrectedSlot(name="user_name", value="John"), tracker, flows
    )

    # Assert
    assert result is False


def test_should_correct_slot_when_flow_not_found_in_flows_list():
    """Test that slot correction is denied when flow is not found in flows list."""
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="user_name", value="John")]
    )

    # Create flows list that returns None for the flow
    flows = Mock(spec=FlowsList)
    flows.flow_by_id.return_value = None

    # Create stack with a flow
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="nonexistent_flow", step_id="some_step")]
    )

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Act
    result = command.should_correct_slot(
        CorrectedSlot(name="user_name", value="John"), tracker, flows
    )

    # Assert
    assert result is False


def test_should_correct_slot_with_mixed_stack_content():
    """Test that slot correction works correctly with mixed stack content
    (user flows and patterns).
    """
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="user_name", value="John")]
    )

    # Create a flow with collect steps
    flow = Mock(spec=Flow)
    collect_step = CollectInformationFlowStep(
        collect="user_name",
        utter="utter_ask_user_name",
        collect_action="action_ask_user_name",
        rejections=[],
        custom_id="collect_name",
        idx=0,
        description=None,
        metadata={},
        next=Mock(),
        flow_id="test_flow",
    )
    flow.get_collect_steps.return_value = [collect_step]

    # Create flows list
    flows = Mock(spec=FlowsList)
    flows.flow_by_id.return_value = flow

    # Create stack with mixed content (pattern frame and user flow)
    stack = DialogueStack(
        frames=[
            PatternFlowStackFrame(flow_id="pattern_flow", step_id="pattern_step"),
            UserFlowStackFrame(flow_id="test_flow", step_id="collect_name"),
        ]
    )

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Act
    result = command.should_correct_slot(
        CorrectedSlot(name="user_name", value="John"), tracker, flows
    )

    # Assert
    assert result is True
    flows.flow_by_id.assert_called_once_with("test_flow")


def test_should_correct_slot_with_call_and_link_frames():
    """Test that slot correction works with call and link frame types."""
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="user_name", value="John")]
    )

    # Create a flow with collect steps
    flow = Mock(spec=Flow)
    collect_step = CollectInformationFlowStep(
        collect="user_name",
        utter="utter_ask_user_name",
        collect_action="action_ask_user_name",
        rejections=[],
        custom_id="collect_name",
        idx=0,
        description=None,
        metadata={},
        next=Mock(),
        flow_id="test_flow",
    )
    flow.get_collect_steps.return_value = [collect_step]

    # Create flows list
    flows = Mock(spec=FlowsList)
    flows.flow_by_id.return_value = flow

    # Create stack with call and link frames
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="test_flow",
                step_id="collect_name",
                frame_type=FlowStackFrameType.CALL,
            ),
            UserFlowStackFrame(
                flow_id="another_flow",
                step_id="some_step",
                frame_type=FlowStackFrameType.LINK,
            ),
        ]
    )

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Act
    result = command.should_correct_slot(
        CorrectedSlot(name="user_name", value="John"), tracker, flows
    )

    # Assert
    assert result is True
    assert flows.flow_by_id.call_count == 1


def test_run_command_on_tracker_skips_correction_when_slot_not_in_collect_step():
    """Test that run_command_on_tracker skips correction when slot is not
    part of collect steps.
    """
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="invalid_slot", value="some_value")]
    )

    # Create a flow with collect steps for different slots
    flow = Mock(spec=Flow)
    collect_step1 = CollectInformationFlowStep(
        collect="user_name",
        utter="utter_ask_user_name",
        collect_action="action_ask_user_name",
        rejections=[],
        custom_id="collect_name",
        idx=0,
        description=None,
        metadata={},
        next=Mock(),
        flow_id="test_flow",
    )
    collect_step2 = CollectInformationFlowStep(
        collect="email",
        utter="utter_ask_email",
        collect_action="action_ask_email",
        rejections=[],
        custom_id="collect_email",
        idx=1,
        description=None,
        metadata={},
        next=Mock(),
        flow_id="test_flow",
    )
    flow.get_collect_steps.return_value = [collect_step1, collect_step2]

    # Create flows list
    flows = Mock(spec=FlowsList)
    flows.flow_by_id.return_value = flow

    # Create stack with the flow
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="test_flow", step_id="collect_name")]
    )

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Create original tracker
    original_tracker = Mock(spec=DialogueStateTracker)

    # Act
    result = command.run_command_on_tracker(tracker, flows, original_tracker)

    # Assert
    assert result == []  # Should return empty list when correction is skipped
    # Verify that the stack was not modified
    assert len(stack.frames) == 1
    assert stack.frames[0].flow_id == "test_flow"


def test_run_command_on_tracker_skips_correction_when_no_user_flows_on_stack():
    """Test that run_command_on_tracker skips correction when no
    user flows are on stack.
    """
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="user_name", value="John")]
    )

    # Create stack with only pattern frames (no user flows)
    stack = DialogueStack(
        frames=[PatternFlowStackFrame(flow_id="pattern_flow", step_id="pattern_step")]
    )

    # Create flows list
    flows = Mock(spec=FlowsList)

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Create original tracker
    original_tracker = Mock(spec=DialogueStateTracker)

    # Act
    result = command.run_command_on_tracker(tracker, flows, original_tracker)

    # Assert
    assert result == []  # Should return empty list when correction is skipped
    # Verify that the stack was not modified
    assert len(stack.frames) == 1
    assert stack.frames[0].flow_id == "pattern_flow"


def test_run_command_on_tracker_skips_correction_when_stack_is_empty():
    """Test that run_command_on_tracker skips correction when stack is empty."""
    # Arrange
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="user_name", value="John")]
    )

    # Create empty stack
    stack = DialogueStack.empty()

    # Create flows list
    flows = Mock(spec=FlowsList)

    # Create tracker
    tracker = Mock(spec=DialogueStateTracker)
    tracker.stack = stack

    # Create original tracker
    original_tracker = Mock(spec=DialogueStateTracker)

    # Act
    result = command.run_command_on_tracker(tracker, flows, original_tracker)

    # Assert
    assert result == []  # Should return empty list when correction is skipped
    # Verify that the stack remains empty
    assert len(stack.frames) == 0


def test_command_name():
    # names of commands should not change as they are part of persisted
    # trackers
    assert CorrectSlotsCommand.command() == "correct slot"


def test_from_dict():
    assert CorrectSlotsCommand.from_dict(
        {"corrected_slots": [{"name": "foo", "value": "bar"}]}
    ) == CorrectSlotsCommand(corrected_slots=[CorrectedSlot(name="foo", value="bar")])


def test_from_dict_fails_if_missing_name_parameter():
    with pytest.raises(ValueError):
        CorrectSlotsCommand.from_dict({"corrected_slots": [{"value": "bar"}]})


def test_from_dict_fails_if_missing_value_parameter():
    with pytest.raises(ValueError):
        CorrectSlotsCommand.from_dict({"corrected_slots": [{"name": "foo"}]})


def test_from_dict_fails_if_missing_corrected_slots_parameter():
    with pytest.raises(ValueError):
        CorrectSlotsCommand.from_dict({})


def test_run_command_on_tracker_without_flows():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = CorrectSlotsCommand(corrected_slots=[])

    assert command.run_command_on_tracker(tracker, [], tracker) == []


@pytest.fixture
def collect_foo_tracker_update() -> Dict[str, Any]:
    return {
        "frame_id": "HF1UKRND",
        "flow_id": "pattern_collect_information",
        "step_id": "START",
        "collect": "foo",
        "utter": "utter_ask_foo",
        "collect_action": "action_ask_foo",
        "rejections": [],
        "type": "pattern_collect_information",
    }


@pytest.fixture
def collect_bar_tracker_update() -> Dict[str, Any]:
    return {
        "frame_id": "HF1UKRND",
        "flow_id": "pattern_collect_information",
        "step_id": "START",
        "collect": "bar",
        "utter": "utter_ask_bar",
        "collect_action": "action_ask_bar",
        "rejections": [],
        "type": "pattern_collect_information",
    }


def test_run_command_on_tracker_correcting_previous_flow():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
            steps:
            - id: collect_foo
              collect: foo
              next: collect_bar
            - id: collect_bar
              collect: bar
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    update_tracker_with_path_through_flow(
        tracker, "my_flow", ["collect_foo", "collect_bar"]
    )
    command = CorrectSlotsCommand(
        corrected_slots=[
            CorrectedSlot(
                name="foo", value="not-foofoo", filled_by=SetSlotExtractor.LLM.value
            )
        ]
    )

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    # flow should still be on the stack and a correction pattern should have been added
    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 2

    assert dialogue_stack_dump[1]["type"] == "pattern_correction"
    assert dialogue_stack_dump[1]["flow_id"] == "pattern_correction"
    assert dialogue_stack_dump[1]["step_id"] == "START"
    assert dialogue_stack_dump[1]["corrected_slots"] == {
        "foo": {"value": "not-foofoo", "filled_by": SetSlotExtractor.LLM.value}
    }
    assert dialogue_stack_dump[1]["reset_flow_id"] == "my_flow"
    assert dialogue_stack_dump[1]["reset_step_id"] == "collect_foo"
    assert dialogue_stack_dump[1]["is_reset_only"] is False
    assert dialogue_stack_dump[1]["new_slot_values"] == ["not-foofoo"]


def test_run_command_on_tracker_correcting_current_flow(
    collect_foo_tracker_update: Dict[str, Any],
    collect_bar_tracker_update: Dict[str, Any],
):
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
            steps:
            - id: collect_foo
              collect: foo
              next: collect_bar
            - id: collect_bar
              collect: bar
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                collect_foo_tracker_update,
                collect_bar_tracker_update,
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "my_flow",
                    "step_id": "collect_bar",
                    "frame_id": "some-frame-id",
                },
            ]
        )
    )
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="bar", value="barbar")]
    )

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    # flow should still be on the stack and a correction flow should have been added
    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 4

    assert dialogue_stack_dump[3]["type"] == "pattern_correction"
    assert dialogue_stack_dump[3]["flow_id"] == "pattern_correction"
    assert dialogue_stack_dump[3]["step_id"] == "START"
    assert dialogue_stack_dump[3]["corrected_slots"] == {
        "bar": {"value": "barbar", "filled_by": None}
    }
    assert dialogue_stack_dump[3]["reset_flow_id"] == "my_flow"
    assert dialogue_stack_dump[3]["reset_step_id"] == "collect_bar"
    assert dialogue_stack_dump[3]["is_reset_only"] is False
    assert dialogue_stack_dump[3]["new_slot_values"] == ["barbar"]


# Skipped in https://rasahq.atlassian.net/browse/ENG-687
# To be fixed in https://rasahq.atlassian.net/browse/ENG-690
@pytest.mark.skip(reason="ENG-687")
@pytest.mark.parametrize(
    "corrected_slots, events, dialogue_stack",
    [
        (
            [CorrectedSlot(name="foobar", value="foobarfoobar")],
            [
                SlotSet("foo", "foo"),
            ],
            {
                "type": "flow",
                "frame_type": "regular",
                "flow_id": "my_flow",
                "step_id": "collect_bar",
                "frame_id": "some-frame-id",
            },
        ),
        (
            [CorrectedSlot(name="bar", value="barbar")],
            [
                SlotSet("foo", "foo"),
                SlotSet("foobar", "foobar"),
            ],
            {
                "type": "flow",
                "frame_type": "regular",
                "flow_id": "my_flow",
                "step_id": "collect_foobar",
                "frame_id": "some-frame-id",
            },
        ),
        (
            [CorrectedSlot(name="foo", value="foofoo")],
            [],
            {
                "type": "flow",
                "frame_type": "regular",
                "flow_id": "my_flow",
                "step_id": "collect_foo",
                "frame_id": "some-frame-id",
            },
        ),
        (
            [CorrectedSlot(name="bar", value="barbar")],
            [
                SlotSet("foo", "foofoo"),
                SlotSet("bar", "bar"),
                SlotSet("foobar", "foobar"),
            ],
            {
                "type": "flow",
                "frame_type": "regular",
                "flow_id": "my_flow",
                "step_id": "collect_foobar",
                "frame_id": "some-frame-id",
            },
        ),
    ],
)
def test_run_command_on_tracker_correcting_invalid_slot(
    corrected_slots: List[CorrectedSlot],
    events: List[Event],
    dialogue_stack: Dict[str, str],
):
    all_flows = flows_from_str_with_defaults(
        """
        flows:
          my_flow:
            name: foo flow
            steps:
            - id: collect_foo
              collect: foo
              next:
              - if: foo == "foo"
                then: collect_bar
              - else: collect_foobar
            - id: collect_bar
              collect: bar
              next: END
            - id: collect_foobar
              collect: foobar
              next: END
        """
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=events,
    )
    tracker.update_stack(DialogueStack.from_dict([dialogue_stack]))
    command = CorrectSlotsCommand(corrected_slots=corrected_slots)

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 0


def test_run_command_on_tracker_correcting_slot_with_asked_before_filling():
    all_flows = flows_from_str_with_defaults(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
            steps:
            - id: collect_foo
              collect: foo
              ask_before_filling: true
              next: collect_bar
            - id: collect_bar
              collect: bar
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    update_tracker_with_path_through_flow(
        tracker, "my_flow", ["collect_foo", "collect_bar"]
    )
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="foo", value="foofoo")]
    )

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 2

    frame = updated_stack.frames[1]
    assert isinstance(frame, CorrectionPatternFlowStackFrame)
    assert frame.flow_id == "pattern_correction"
    assert frame.is_reset_only
    assert frame.corrected_slots == {"foo": {"value": "foofoo", "filled_by": None}}
    assert frame.step_id == "START"
    assert frame.reset_step_id == "collect_foo"
    assert frame.new_slot_values == ["foofoo"]


def test_run_command_on_tracker_correcting_during_a_correction():
    all_flows = flows_from_str_including_defaults(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
            steps:
            - id: collect_foo
              collect: foo
              next: collect_bar
            - id: collect_bar
              collect: bar
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    update_tracker_with_path_through_flow(
        tracker, "my_flow", ["collect_foo", "collect_bar"]
    )
    stack = tracker.stack
    stack.push(
        CorrectionPatternFlowStackFrame(
            corrected_slots={
                "foo": {"value": "not-foofoo", "filled_by": SetSlotExtractor.LLM.value}
            },
            step_id="pattern_correction_0_action_correct_flow_slot",
            reset_flow_id="my_flow",
            reset_step_id="collect_foo",
            is_reset_only=False,
            frame_id="some-other-id",
            new_slot_values=["not-foofoo"],
        )
    )
    tracker.update_stack(stack)

    command = CorrectSlotsCommand(
        corrected_slots=[
            CorrectedSlot(
                name="bar", value="barbar", filled_by=SetSlotExtractor.LLM.value
            )
        ]
    )

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 3

    assert dialogue_stack_dump[1]["type"] == "pattern_correction"
    assert dialogue_stack_dump[1]["flow_id"] == "pattern_correction"
    assert dialogue_stack_dump[1]["step_id"] == "START"
    assert dialogue_stack_dump[1]["corrected_slots"] == {
        "bar": {"value": "barbar", "filled_by": SetSlotExtractor.LLM.value}
    }
    assert dialogue_stack_dump[1]["reset_flow_id"] == "my_flow"
    assert dialogue_stack_dump[1]["reset_step_id"] == "collect_bar"
    assert dialogue_stack_dump[1]["new_slot_values"] == ["barbar"]

    assert dialogue_stack_dump[2]["type"] == "pattern_correction"
    assert dialogue_stack_dump[2]["corrected_slots"] == {
        "foo": {"value": "not-foofoo", "filled_by": SetSlotExtractor.LLM.value}
    }
    assert dialogue_stack_dump[2]["new_slot_values"] == ["not-foofoo"]


def test_determine_index_for_new_correction_frame_handles_empty_stack():
    stack = DialogueStack.empty()
    top_flow_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    assert (
        CorrectSlotsCommand.determine_index_for_new_correction_frame(
            top_flow_frame, stack
        )
        == 0
    )


def test_index_for_correction_handles_non_correction_pattern_at_the_top_of_stack():
    top_flow_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            ),
            top_flow_frame,
        ]
    )
    assert (
        CorrectSlotsCommand.determine_index_for_new_correction_frame(
            top_flow_frame, stack
        )
        == 2
    )


def test_index_for_correction_handles_correction_pattern_at_the_top_of_stack():
    top_flow_frame = CorrectionPatternFlowStackFrame(
        corrected_slots={"foo": "not-foofoo"},
    )
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            ),
            top_flow_frame,
        ]
    )
    # new correction pattern should be inserted "under" the existing correction pattern
    assert (
        CorrectSlotsCommand.determine_index_for_new_correction_frame(
            top_flow_frame, stack
        )
        == 1
    )


def test_index_for_correction_handles_call_and_existing_correction_pattern():
    top_flow_frame = CorrectionPatternFlowStackFrame(
        corrected_slots={"foo": "not-foofoo"},
    )
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            ),
            UserFlowStackFrame(
                flow_id="bar",
                step_id="first_step",
                frame_id="some-call-id",
                frame_type=FlowStackFrameType.CALL,
            ),
            top_flow_frame,
        ]
    )
    # new correction pattern should be inserted "under" the existing correction pattern
    assert (
        CorrectSlotsCommand.determine_index_for_new_correction_frame(
            top_flow_frame, stack
        )
        == 2
    )


def test_end_previous_correction():
    top_flow_frame = CorrectionPatternFlowStackFrame(
        corrected_slots={"foo": "not-foofoo"},
    )
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            ),
            top_flow_frame,
        ]
    )
    CorrectSlotsCommand.end_previous_correction(top_flow_frame, stack)
    # the previous pattern should be about to end
    assert stack.frames[1].step_id == "NEXT:END"
    # make sure the user flow has not been modified
    assert stack.frames[0].step_id == "first_step"


def test_end_previous_correction_no_correction_present():
    top_flow_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[top_flow_frame])
    CorrectSlotsCommand.end_previous_correction(top_flow_frame, stack)
    # make sure the user flow has not been modified
    assert stack.frames[0].step_id == "first_step"


@pytest.mark.parametrize(
    "updated_slots, expected_step_id",
    [
        (
            ["foo", "bar"],
            "collect_foo",
        ),
        (
            ["bar", "foo"],
            "collect_foo",
        ),
        (
            ["bar"],
            "collect_bar",
        ),
        (
            ["foo"],
            "collect_foo",
        ),
        (
            [],
            None,
        ),
    ],
)
def test_find_earliest_updated_collect_info(
    updated_slots: List[str],
    expected_step_id: str,
):
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            name: foo flow
            description: foo flow
            steps:
            - id: collect_foo
              collect: foo
              next: collect_bar
            - id: collect_bar
              collect: bar
              next: collect_baz
            - id: collect_baz
              collect: baz
        """
    )

    tracker = DialogueStateTracker.from_events("bot", evts=[])
    update_tracker_with_path_through_flow(
        tracker, "my_flow", ["collect_foo", "collect_bar"]
    )

    step = CorrectSlotsCommand.find_earliest_updated_collect_info(
        updated_slots, all_flows, tracker
    )
    if expected_step_id is not None:
        assert step is not None
        assert step.step.id == expected_step_id
    else:
        assert step is None


@pytest.mark.parametrize(
    "proposed_slots, expected",
    [
        ({}, True),
        ({"foo": "foofoo"}, True),
        ({"bar": "barbar"}, False),
        ({"foo": "foofoo", "bar": "barbar"}, False),
    ],
)
def test_are_all_slots_reset_only(proposed_slots: Dict[str, Any], expected: bool):
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
            steps:
            - id: collect_foo
              collect: foo
              ask_before_filling: true
              next: collect_bar
            - id: collect_bar
              collect: bar
        """
    )
    assert (
        CorrectSlotsCommand.are_all_slots_reset_only(proposed_slots, all_flows)
        == expected
    )


@pytest.mark.parametrize("step", ["call"])
def test_run_command_on_tracker_with_prefilled_slots_of_child_flows(
    step: str,
):
    all_flows = flows_from_str(
        f"""
        flows:
          flow1:
            description: flow one
            name: first flow
            steps:
            - id: collect_bar
              collect: bar
            - id: collect_test
              collect: test
            - {step}: flow2
          flow2:
            description: flow 2
            name: second flow
            steps:
            - id: collect_foo
              collect: foo
            - id: collect_name
              collect: name
        """
    )
    slot_set_events = [SlotSet("bar", "bar"), SlotSet("foo", "foo")]

    tracker = DialogueStateTracker.from_events("test", evts=slot_set_events)
    update_tracker_with_path_through_flow(
        tracker, "flow1", ["collect_bar", "collect_test"]
    )
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="foo", value="foofoo")]
    )

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 2

    assert dialogue_stack_dump[0]["type"] == "flow"
    assert dialogue_stack_dump[0]["flow_id"] == "flow1"
    assert dialogue_stack_dump[0]["step_id"] == "collect_test"
    assert dialogue_stack_dump[0]["frame_type"] == FlowStackFrameType.REGULAR
    assert dialogue_stack_dump[0]["frame_id"] == "some-frame-id"

    assert dialogue_stack_dump[1]["type"] == "pattern_correction"
    assert dialogue_stack_dump[1]["flow_id"] == "pattern_correction"
    assert dialogue_stack_dump[1]["step_id"] == "START"
    assert dialogue_stack_dump[1]["corrected_slots"] == {
        "foo": {"value": "foofoo", "filled_by": None}
    }
    assert dialogue_stack_dump[1]["reset_flow_id"] is None
    assert dialogue_stack_dump[1]["reset_step_id"] is None
    assert dialogue_stack_dump[1]["is_reset_only"] is False
    assert dialogue_stack_dump[1]["new_slot_values"] == ["foofoo"]


@pytest.mark.parametrize("step", ["call"])
def test_create_correction_frame_with_prefilled_slots_of_child_flows(
    step: str,
):
    all_flows = flows_from_str(
        f"""
        flows:
          flow1:
            description: flow one
            name: first flow
            steps:
            - id: collect_bar
              collect: bar
            - id: collect_test
              collect: test
            - {step}: flow2
          flow2:
            description: flow 2
            name: second flow
            steps:
            - id: collect_foo
              collect: foo
            - id: collect_name
              collect: name
        """
    )
    proposed_slots = {"foo": {"value": "foofoo", "filled_by": None}}
    slot_set_events = [SlotSet("bar", "bar"), SlotSet("foo", "foo")]
    tracker = DialogueStateTracker.from_events("test", evts=slot_set_events)
    update_tracker_with_path_through_flow(
        tracker, "flow1", ["collect_bar", "collect_test"]
    )
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="foo", value="foofoo")]
    )
    correction_frame = command.create_correction_frame(
        proposed_slots, all_flows, tracker
    )
    assert isinstance(correction_frame, CorrectionPatternFlowStackFrame)
    assert correction_frame.flow_id == "pattern_correction"
    assert correction_frame.step_id == "START"
    assert correction_frame.reset_flow_id is None
    assert correction_frame.reset_step_id is None
    assert correction_frame.is_reset_only is False
    assert correction_frame.corrected_slots == {
        "foo": {"value": "foofoo", "filled_by": None}
    }
    assert correction_frame.new_slot_values == ["foofoo"]
