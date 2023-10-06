import pytest

from rasa.dialogue_understanding.patterns.code_change import FLOW_PATTERN_CODE_CHANGE_ID
from rasa.dialogue_understanding.processor.command_processor import (
    execute_commands,
    find_updated_flows,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    UserFlowStackFrame,
    PatternFlowStackFrame,
)
from rasa.shared.core.constants import FLOW_HASHES_SLOT
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from tests.dialogue_understanding.commands.conftest import start_bar_user_uttered
from tests.utilities import flows_from_str


def test_properly_prepared_tracker(tracker: DialogueStateTracker):
    # flow hashes have been initialized
    assert "foo" in tracker.get_slot(FLOW_HASHES_SLOT)

    # foo flow is on the stack
    dialogue_stack = DialogueStack.from_tracker(tracker)
    assert (top_frame := dialogue_stack.top())
    assert isinstance(top_frame, UserFlowStackFrame)
    assert top_frame.flow_id == "foo"


def test_detects_no_changes_when_nothing_changed(
    tracker: DialogueStateTracker, all_flows: FlowsList
):
    assert find_updated_flows(tracker, all_flows) == set()


def test_detects_no_changes_for_not_started_flows(
    tracker: DialogueStateTracker,
):
    bar_changed_flows = flows_from_str(
        """
        flows:
          foo:
            steps:
            - id: first_step
              action: action_listen
          bar:
            steps:
            - id: also_first_step_BUT_CHANGED
              action: action_listen
    """
    )
    assert find_updated_flows(tracker, bar_changed_flows) == set()


change_cases = {
    "step_id_changed": """
        flows:
          foo:
            steps:
            - id: first_step_id_BUT_CHANGED
              action: action_listen
          bar:
            steps:
            - id: also_first_step
              action: action_listen
        """,
    "action_changed": """
        flows:
          foo:
            steps:
            - id: first_step_id
              action: action_CHANGED
          bar:
            steps:
            - id: also_first_step
              action: action_listen
        """,
    "new_step": """
        flows:
          foo:
            steps:
            - id: first_step_id
              action: action_listen
              next: second_step_id
            - id: second_step_id
              action: action_cool_stuff
          bar:
            steps:
            - id: also_first_step
              action: action_listen
        """,
    "flow_removed": """
        flows:
          bar:
            steps:
            - id: also_first_step
              action: action_listen
        """,
}


@pytest.mark.parametrize("case, flow_yaml", list(change_cases.items()))
def test_detects_changes(case: str, flow_yaml: str, tracker: DialogueStateTracker):
    all_flows = flows_from_str(flow_yaml)
    assert find_updated_flows(tracker, all_flows) == {"foo"}


def test_starting_of_another_flow(tracker: DialogueStateTracker, all_flows: FlowsList):
    """Tests that commands are not discarded when there is no change."""
    tracker.update_with_events([start_bar_user_uttered], None)
    execute_commands(tracker, all_flows)
    dialogue_stack = DialogueStack.from_tracker(tracker)
    assert len(dialogue_stack.frames) == 2
    assert (top_frame := dialogue_stack.top())
    assert isinstance(top_frame, UserFlowStackFrame)
    assert top_frame.flow_id == "bar"


def test_stack_cleaning_command_is_applied_on_changes(tracker: DialogueStateTracker):
    all_flows = flows_from_str(change_cases["step_id_changed"])
    tracker.update_with_events([start_bar_user_uttered], None)
    execute_commands(tracker, all_flows)
    dialogue_stack = DialogueStack.from_tracker(tracker)
    assert len(dialogue_stack.frames) == 2
    assert (top_frame := dialogue_stack.top())
    assert isinstance(top_frame, PatternFlowStackFrame)
    assert top_frame.flow_id == FLOW_PATTERN_CODE_CHANGE_ID
