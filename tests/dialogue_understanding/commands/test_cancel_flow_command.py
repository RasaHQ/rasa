from typing import Iterator
from unittest.mock import MagicMock, Mock, patch

import jsonpatch
import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.available_agents import AvailableAgents
from rasa.dialogue_understanding.commands.cancel_flow_command import CancelFlowCommand
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.engine.language import Language
from rasa.shared.core.events import AgentCancelled, DialogueStackUpdated, FlowCancelled
from rasa.shared.core.slots import StrictCategoricalSlot
from rasa.shared.core.trackers import DialogueStateTracker
from tests.utilities import flows_from_str


@pytest.fixture
def mock_available_agents(monkeypatch: MonkeyPatch) -> Iterator[MagicMock]:
    mock_instance = MagicMock()
    mock_instance.agents = {
        "car-research": {},
    }

    with patch.object(
        AvailableAgents, "get_instance", return_value=mock_instance
    ) as mock_method:
        yield mock_method


def test_command_name():
    # names of commands should not change as they are part of persisted
    # trackers
    assert CancelFlowCommand.command() == "cancel flow"


def test_from_dict():
    assert CancelFlowCommand.from_dict({}) == CancelFlowCommand()


def test_run_command_on_tracker_without_flows():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = CancelFlowCommand()

    assert command.run_command_on_tracker(tracker, [], tracker) == []


def test_run_command_on_tracker():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            name: foo flow
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[],
    )
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "first_step",
                    "frame_id": "some-frame-id",
                }
            ]
        )
    )
    command = CancelFlowCommand()

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 2

    # the first event should be a flow canceled event
    flow_cancelled_event = events[0]
    assert flow_cancelled_event == FlowCancelled("foo", "first_step")

    dialogue_stack_event = events[1]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    # flow should still be on the stack and a cancel flow should have been added
    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 2

    assert dialogue_stack_dump[1]["type"] == "pattern_cancel_flow"
    assert dialogue_stack_dump[1]["flow_id"] == "pattern_cancel_flow"
    assert dialogue_stack_dump[1]["step_id"] == "START"
    assert dialogue_stack_dump[1]["canceled_name"] == "foo flow"
    assert dialogue_stack_dump[1]["canceled_frames"] == ["some-frame-id"]


@pytest.mark.parametrize("step", ["link", "call"])
def test_run_command_on_tracker_with_linked_and_called_flows(step: str):
    all_flows = flows_from_str(
        f"""
        flows:
          foo:
            description: flow foo
            name: foo flow
            steps:
            - id: first_step
              action: action_listen
            - {step}: bar
          bar:
            description: flow bar
            name: bar flow
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[],
    )
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "first_step",
                    "frame_id": "some-frame-id",
                },
                {
                    "type": "flow",
                    "frame_type": step,
                    "flow_id": "bar",
                    "step_id": "second_step",
                    "frame_id": "some-other-frame-id",
                },
            ]
        )
    )
    command = CancelFlowCommand()

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 2

    # the first event should be a flow canceled event
    flow_cancelled_event = events[0]
    assert flow_cancelled_event == FlowCancelled("bar", "second_step")

    dialogue_stack_event = events[1]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    # flow should still be on the stack and a cancel flow should have been added
    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 3

    assert dialogue_stack_dump[2]["type"] == "pattern_cancel_flow"
    assert dialogue_stack_dump[2]["flow_id"] == "pattern_cancel_flow"
    assert dialogue_stack_dump[2]["step_id"] == "START"
    assert dialogue_stack_dump[2]["canceled_name"] == "bar flow"
    assert dialogue_stack_dump[2]["canceled_frames"][0] == "some-other-frame-id"


def test_select_canceled_frames_cancels_patterns():
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            ),
            CollectInformationPatternFlowStackFrame(
                collect="bar", frame_id="some-other-id"
            ),
        ]
    )

    canceled_frames = CancelFlowCommand.select_canceled_frames(stack)
    assert len(canceled_frames) == 2
    assert canceled_frames[0] == "some-other-id"
    assert canceled_frames[1] == "some-frame-id"


def test_select_canceled_frames_cancels_calls():
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            ),
            UserFlowStackFrame(
                flow_id="bar",
                step_id="collect_bar",
                frame_id="some-call-id",
                frame_type=FlowStackFrameType.CALL,
            ),
            CollectInformationPatternFlowStackFrame(
                collect="bar", frame_id="some-other-id"
            ),
        ]
    )

    canceled_frames = CancelFlowCommand.select_canceled_frames(stack)
    assert len(canceled_frames) == 3
    assert canceled_frames[0] == "some-other-id"
    assert canceled_frames[1] == "some-call-id"
    assert canceled_frames[2] == "some-frame-id"


def test_select_canceled_frames_cancels_only_top_user_flow():
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="bar", step_id="first_step", frame_id="some-bar-id"
            ),
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-foo-id"
            ),
        ]
    )

    canceled_frames = CancelFlowCommand.select_canceled_frames(stack)
    assert len(canceled_frames) == 1
    assert canceled_frames[0] == "some-foo-id"


@pytest.mark.parametrize(
    "agent_stack_frame, should_cancel_agent",
    [
        (
            AgentStackFrame(
                frame_id="agent-frame-id",
                state=AgentState.WAITING_FOR_INPUT,
                agent_id="car-research",
                flow_id="bar",
            ),
            True,
        ),
        (
            AgentStackFrame(
                frame_id="agent-frame-id",
                state=AgentState.INTERRUPTED,
                agent_id="car-research",
                flow_id="bar",
            ),
            False,
        ),
    ],
)
def test_cancel_agent_when_cancelling_flow(
    agent_stack_frame: AgentStackFrame,
    should_cancel_agent: bool,
    mock_available_agents: Mock,
) -> None:
    all_flows = flows_from_str(
        """
            flows:
              foo:
                description: flow foo
                name: foo flow
                steps:
                - collect: foo_slot
              bar:
                description: flow bar
                name: bar flow
                steps:
                - id: first_step
                  call: car-research
            """
    )

    interrupted_flow_stack_frame = UserFlowStackFrame(
        flow_id="foo", step_id="START", frame_id="interrupted-frame-id"
    )
    stack = DialogueStack(
        frames=[
            interrupted_flow_stack_frame,
            UserFlowStackFrame(
                flow_id="bar",
                step_id="START",
                frame_id="some-other-frame-id",
                frame_type=FlowStackFrameType.INTERRUPT,
            ),
            agent_stack_frame,
        ]
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[],
    )
    tracker.update_stack(stack)
    tracker_before_command_execution = tracker.copy()

    cancel_command = CancelFlowCommand()

    events = cancel_command.run_command_on_tracker(tracker, all_flows, tracker)

    if should_cancel_agent:
        # The actual tracker stack should not be changed, we only expect an
        # DialogueStackUpdated event that removes the agent frame to be emitted
        assert tracker.stack == tracker_before_command_execution.stack
        assert agent_stack_frame in tracker.stack.frames

        # Check that DialogueStackUpdated event is present and correct
        dialogue_stack_updated_events = [
            e for e in events if isinstance(e, DialogueStackUpdated)
        ]
        assert len(dialogue_stack_updated_events) == 1
        tracker_before_command_execution.update(dialogue_stack_updated_events[0])
        # After we applied the produced event, the agent frame should be removed
        # from the stack
        assert agent_stack_frame not in tracker_before_command_execution.stack.frames

        # Check that AgentCancelled event is present and correct
        agent_cancelled_events = [e for e in events if isinstance(e, AgentCancelled)]
        assert len(agent_cancelled_events) == 1
        assert agent_cancelled_events[0].agent_id == agent_stack_frame.agent_id
        assert agent_cancelled_events[0].flow_id == agent_stack_frame.flow_id

    else:
        assert agent_stack_frame in tracker.stack.frames
        assert tracker.stack == tracker_before_command_execution.stack


def test_interrupted_agent_of_other_flow_is_not_removed(
    mock_available_agents: Mock,
) -> None:
    all_flows = flows_from_str(
        """
            flows:
              foo:
                description: flow foo
                name: foo flow
                steps:
                - call: car-research
              bar:
                description: flow bar
                name: bar flow
                steps:
                - call: car-research
            """
    )

    foo_agent_stack_frame = AgentStackFrame(
        frame_id="agent-frame-id-1",
        state=AgentState.INTERRUPTED,
        agent_id="car-research",
        flow_id="foo",
    )
    bar_agent_stack_frame = AgentStackFrame(
        frame_id="agent-frame-id-2",
        state=AgentState.WAITING_FOR_INPUT,
        agent_id="car-research",
        flow_id="bar",
    )

    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="START", frame_id="interrupted-frame-id"
            ),
            foo_agent_stack_frame,
            UserFlowStackFrame(
                flow_id="bar",
                step_id="START",
                frame_id="some-other-frame-id",
                frame_type=FlowStackFrameType.INTERRUPT,
            ),
            bar_agent_stack_frame,
        ]
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[],
    )
    tracker.update_stack(stack)
    tracker_before_command_execution = tracker.copy()

    cancel_command = CancelFlowCommand()

    events = cancel_command.run_command_on_tracker(tracker, all_flows, tracker)

    # The actual tracker stack should not be changed, we only expect an
    # DialogueStackUpdated event that removes the agent frame to be emitted
    assert tracker.stack == tracker_before_command_execution.stack

    # Check that DialogueStackUpdated event is present and correct
    dialogue_stack_updated_events = [
        e for e in events if isinstance(e, DialogueStackUpdated)
    ]
    assert len(dialogue_stack_updated_events) == 1
    tracker_before_command_execution.update(dialogue_stack_updated_events[0])
    # After we applied the produced event, the agent frame should be removed
    # from the stack
    assert foo_agent_stack_frame in tracker_before_command_execution.stack.frames
    assert bar_agent_stack_frame not in tracker_before_command_execution.stack.frames

    # Check that AgentCancelled event is present and correct
    agent_cancelled_events = [e for e in events if isinstance(e, AgentCancelled)]
    assert len(agent_cancelled_events) == 1
    assert agent_cancelled_events[0].agent_id == bar_agent_stack_frame.agent_id
    assert agent_cancelled_events[0].flow_id == bar_agent_stack_frame.flow_id


def test_select_canceled_frames_empty_stack():
    stack = DialogueStack.empty()

    with pytest.raises(ValueError):
        # this shouldn't actually, happen. if the stack is empty we shouldn't
        # try to cancel anything.
        CancelFlowCommand.select_canceled_frames(stack)


def test_select_canceled_frames_raises_if_frame_not_found():
    stack = DialogueStack.empty()

    with pytest.raises(ValueError):
        # can't cancel if there is no user flow on the stack. in reality
        # this should never happen as the flow should always be on the stack
        # when this command is executed.
        CancelFlowCommand.select_canceled_frames(stack)


def test_to_dsl_default():
    command = CancelFlowCommand()
    assert command.to_dsl() == "CancelFlow()"


def test_from_dsl():
    assert CancelFlowCommand.from_dsl(None) == CancelFlowCommand()


def test_regex_pattern_default():
    assert CancelFlowCommand.regex_pattern() == r"CancelFlow\(\)"


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = CancelFlowCommand()
    assert command.to_dsl() == "cancel flow"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert CancelFlowCommand.regex_pattern() == r"""^[\s\W\d]*cancel flow['"`]*$"""

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_is_instance_of_prompt_command():
    # Check if the command adheres to the PromptCommand protocol.
    assert isinstance(CancelFlowCommand(), PromptCommand) is True


def test_cancel_flow_command_uses_localized_flow_name(monkeypatch: pytest.MonkeyPatch):
    # Load a flow with translations.
    german_flow_name = "German foo"
    all_flows = flows_from_str(
        f"""
        flows:
          foo:
            description: flow foo
            name: foo flow
            translation:
                de:
                  name: {german_flow_name}
            steps:
            - id: first_step
              action: action_listen
        """
    )

    # Create a tracker with a language slot set to German language.
    language = Language.from_language_code("de", is_default=True)
    slots = [
        StrictCategoricalSlot(
            "language", [], initial_value=language.code, values=[language.code]
        )
    ]
    tracker = DialogueStateTracker.from_events("test", evts=[], slots=slots)

    # Add a flow to the tracker.
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "first_step",
                    "frame_id": "some-frame-id",
                }
            ]
        )
    )

    # Run the cancel flow command.
    command = CancelFlowCommand()
    events = command.run_command_on_tracker(tracker, all_flows, tracker)

    # Check that the canceled name is the German translation.
    dialogue_stack_event = events[-1]
    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())
    assert dialogue_stack_dump[-1]["canceled_name"] == german_flow_name


def test_to_dsl_v3_command_syntax():
    # Set the syntax version to v3 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    command = CancelFlowCommand()
    assert command.to_dsl() == "cancel flow"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v3_command_syntax():
    # Set the syntax version to v3 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    assert CancelFlowCommand.regex_pattern() == r"""^[\s\W\d]*cancel flow['"`]*$"""

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()
