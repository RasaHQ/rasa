from typing import List, Type
from unittest.mock import Mock

import pytest

from rasa.dialogue_understanding.commands import (
    Command,
    CorrectSlotsCommand,
    RepeatBotMessagesCommand,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.patterns.code_change import FLOW_PATTERN_CODE_CHANGE_ID
from rasa.dialogue_understanding.processor.command_processor import (
    ensure_max_number_of_command_type,
    execute_commands,
    filter_start_flow_commands,
    find_updated_flows,
)
from rasa.dialogue_understanding.stack.frames import (
    PatternFlowStackFrame,
    UserFlowStackFrame,
)
from rasa.shared.core.constants import FLOW_HASHES_SLOT
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from tests.dialogue_understanding.conftest import start_bar_user_uttered
from tests.utilities import flows_from_str


def test_properly_prepared_tracker(tracker: DialogueStateTracker):
    # flow hashes have been initialized
    assert "foo" in tracker.get_slot(FLOW_HASHES_SLOT)

    # foo flow is on the stack
    stack = tracker.stack
    top_frame = stack.top()
    assert top_frame
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
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
          bar:
            description: flow bar
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
            description: flow foo
            steps:
            - id: first_step_id_BUT_CHANGED
              action: action_listen
          bar:
            description: flow bar
            steps:
            - id: also_first_step
              action: action_listen
        """,
    "action_changed": """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step_id
              action: action_CHANGED
          bar:
            description: flow bar
            steps:
            - id: also_first_step
              action: action_listen
        """,
    "new_step": """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step_id
              action: action_listen
              next: second_step_id
            - id: second_step_id
              action: action_cool_stuff
          bar:
            description: flow bar
            steps:
            - id: also_first_step
              action: action_listen
        """,
    "flow_removed": """
        flows:
          bar:
            description: flow bar
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
    tracker.update_with_events([start_bar_user_uttered])
    execute_commands(tracker, all_flows, Mock())
    stack = tracker.stack
    assert len(stack.frames) == 2
    top_frame = stack.top()
    assert top_frame
    assert isinstance(top_frame, UserFlowStackFrame)
    assert top_frame.flow_id == "bar"


def test_stack_cleaning_command_is_applied_on_changes(tracker: DialogueStateTracker):
    all_flows = flows_from_str(change_cases["step_id_changed"])
    tracker.update_with_events([start_bar_user_uttered])
    execute_commands(tracker, all_flows, Mock())
    stack = tracker.stack
    assert len(stack.frames) == 2
    top_frame = stack.top()
    assert top_frame
    assert isinstance(top_frame, PatternFlowStackFrame)
    assert top_frame.flow_id == FLOW_PATTERN_CODE_CHANGE_ID


@pytest.mark.parametrize(
    "commands, expected_output",
    [
        ([], []),
        ([StartFlowCommand("foo"), SetSlotCommand("bar", "temp")], ["foo"]),
        (
            [
                StartFlowCommand("foo"),
                StartFlowCommand("bar"),
                CorrectSlotsCommand(["test"]),
            ],
            ["foo", "bar"],
        ),
    ],
)
def test_filter_start_flow_commands(
    commands: List[Command], expected_output: List[str]
) -> None:
    assert filter_start_flow_commands(commands) == expected_output


@pytest.mark.parametrize(
    "commands,t,n,expected_commands",
    [
        ([], StartFlowCommand, 1, []),
        ([StartFlowCommand("abc")], StartFlowCommand, 10, [StartFlowCommand("abc")]),
        (
            [RepeatBotMessagesCommand(), RepeatBotMessagesCommand()],
            RepeatBotMessagesCommand,
            1,
            [RepeatBotMessagesCommand()],
        ),
        (
            [RepeatBotMessagesCommand(), RepeatBotMessagesCommand()],
            RepeatBotMessagesCommand,
            0,
            [],
        ),
        (
            [RepeatBotMessagesCommand(), RepeatBotMessagesCommand()],
            RepeatBotMessagesCommand,
            -1,
            [],
        ),
        (
            [
                RepeatBotMessagesCommand(),
                RepeatBotMessagesCommand(),
                StartFlowCommand("abc"),
            ],
            RepeatBotMessagesCommand,
            0,
            [StartFlowCommand("abc")],
        ),
        (
            [
                RepeatBotMessagesCommand(),
                RepeatBotMessagesCommand(),
                StartFlowCommand("abc"),
            ],
            RepeatBotMessagesCommand,
            -1,
            [StartFlowCommand("abc")],
        ),
        (
            [
                RepeatBotMessagesCommand(),
                RepeatBotMessagesCommand(),
                StartFlowCommand("abc"),
            ],
            RepeatBotMessagesCommand,
            1,
            [RepeatBotMessagesCommand(), StartFlowCommand("abc")],
        ),
        (
            [
                RepeatBotMessagesCommand(),
                RepeatBotMessagesCommand(),
                StartFlowCommand("abc"),
            ],
            RepeatBotMessagesCommand,
            2,
            [
                RepeatBotMessagesCommand(),
                RepeatBotMessagesCommand(),
                StartFlowCommand("abc"),
            ],
        ),
    ],
)
def test_ensure_max_number_of_command_type(
    commands: List[Command], t: Type[Command], n: int, expected_commands: List[Command]
) -> None:
    assert ensure_max_number_of_command_type(commands, t, n) == expected_commands
