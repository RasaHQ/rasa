from typing import List
import pytest

from rasa.dialogue_understanding.commands import StartFlowCommand
from rasa.dialogue_understanding.processor.command_processor import execute_commands
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import UserFlowStackFrame
from rasa.shared.core.events import UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import COMMANDS
from rasa.shared.core.flows.yaml_flows_io import flows_from_str


@pytest.fixture
def all_flows() -> FlowsList:
    return flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
          bar:
            description: flow bar
            steps:
            - id: also_first_step
              action: action_listen
        """
    )


start_foo_user_uttered = UserUttered(
    "start foo", None, None, {COMMANDS: [StartFlowCommand("foo").as_dict()]}
)

start_bar_user_uttered = UserUttered(
    "start bar", None, None, {COMMANDS: [StartFlowCommand("bar").as_dict()]}
)


@pytest.fixture
def tracker(all_flows: FlowsList) -> DialogueStateTracker:
    # Creates a useful tracker that has a started flow and the current flows hashed
    tracker = DialogueStateTracker.from_events("test", evts=[start_foo_user_uttered])
    execute_commands(tracker, all_flows)
    return tracker


def update_tracker_with_path_through_flow(
    tracker: DialogueStateTracker,
    flow_id: str,
    step_ids: List[str],
    frame_id: str = "some-frame-id",
) -> None:
    """Update the tracker with a path through a flow.

    CAUTION: this is not a full simulation of the flow and no side effects,
    such as starting patterns, calls or links are realised.

    CAUTION: this does not check whether the flow or step ids are valid.

    The goal is to create a tracker with stack update events that reflect
    a path through a flow. This is useful for testing the stack update
    logic.

    Args:
        tracker: The tracker to update.
        flow_id: The id of the flow to update the tracker with.
        step_ids: The step ids to update the tracker with.
        frame_id: The frame id to update the tracker with.
    """
    stack = DialogueStack.from_dict(
        [
            {
                "type": "flow",
                "flow_id": flow_id,
                "step_id": "START",
                "frame_id": frame_id,
            }
        ]
    )
    tracker.update_stack(stack)
    for step_id in step_ids:
        advance_top_tracker_flow(tracker, step_id)


def advance_top_tracker_flow(tracker: DialogueStateTracker, step_id: str) -> None:
    """Advance the top tracker flow.

    Args:
        tracker: The tracker to advance the top flow of.
        step_id: The step id to advance the top flow to.
    """
    stack = tracker.stack
    top_frame = stack.top()
    if isinstance(top_frame, UserFlowStackFrame):
        top_frame.step_id = step_id
    else:
        raise ValueError(f"Top frame is not a user flow frame: {top_frame}")
    tracker.update_stack(stack)
