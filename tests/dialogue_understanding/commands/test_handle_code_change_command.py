from unittest.mock import Mock

import pytest

from rasa.core.actions.action_clean_stack import ActionCleanStack
from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.commands.handle_code_change_command import (
    HandleCodeChangeCommand,
)
from rasa.dialogue_understanding.patterns.code_change import FLOW_PATTERN_CODE_CHANGE_ID
from rasa.dialogue_understanding.processor.command_processor import execute_commands
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    PatternFlowStackFrame,
    UserFlowStackFrame,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import DialogueStackUpdated
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.flow import (
    END_STEP,
    START_STEP,
    ContinueFlowStep,
)
from rasa.shared.core.trackers import DialogueStateTracker
from tests.dialogue_understanding.commands.test_command_processor import (
    change_cases,
    start_bar_user_uttered,
)
from tests.utilities import flows_from_str


def test_name_of_command():
    # names of commands should not change as they are part of persisted
    # trackers
    assert HandleCodeChangeCommand.command() == "handle code change"


def test_from_dict():
    assert HandleCodeChangeCommand.from_dict({}) == HandleCodeChangeCommand()


def test_run_command_on_tracker(tracker: DialogueStateTracker, all_flows: FlowsList):
    command = HandleCodeChangeCommand()
    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1
    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 2

    frame = updated_stack.frames[1]
    assert isinstance(frame, PatternFlowStackFrame)
    assert frame.type() == FLOW_PATTERN_CODE_CHANGE_ID


@pytest.fixture
def about_to_be_cleaned_tracker(tracker: DialogueStateTracker, all_flows: FlowsList):
    tracker.update_with_events([start_bar_user_uttered])
    execute_commands(tracker, all_flows, Mock())
    changed_flows = flows_from_str(change_cases["step_id_changed"])
    execute_commands(tracker, changed_flows, Mock())
    stack = tracker.stack
    assert len(stack.frames) == 3

    foo_frame = stack.frames[0]
    assert isinstance(foo_frame, UserFlowStackFrame)
    assert foo_frame.flow_id == "foo"
    assert foo_frame.step_id == START_STEP

    bar_frame = stack.frames[1]
    assert isinstance(bar_frame, UserFlowStackFrame)
    assert bar_frame.flow_id == "bar"
    assert bar_frame.step_id == START_STEP

    stack_clean_frame = stack.frames[2]
    assert isinstance(stack_clean_frame, PatternFlowStackFrame)
    assert stack_clean_frame.flow_id == FLOW_PATTERN_CODE_CHANGE_ID
    assert stack_clean_frame.step_id == START_STEP

    return tracker


async def test_stack_cleaning_action(about_to_be_cleaned_tracker: DialogueStateTracker):
    events = await ActionCleanStack().run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        about_to_be_cleaned_tracker,
        Domain.empty(),
    )
    about_to_be_cleaned_tracker.update_with_events(events)

    stack = about_to_be_cleaned_tracker.stack
    assert len(stack.frames) == 3

    foo_frame = stack.frames[0]
    assert isinstance(foo_frame, UserFlowStackFrame)
    assert foo_frame.flow_id == "foo"
    assert foo_frame.step_id == ContinueFlowStep.continue_step_for_id(END_STEP)

    bar_frame = stack.frames[1]
    assert isinstance(bar_frame, UserFlowStackFrame)
    assert bar_frame.flow_id == "bar"
    assert bar_frame.step_id == ContinueFlowStep.continue_step_for_id(END_STEP)


def test_run_command_returns_empty_when_active_frame_is_stale():
    """run_command_on_tracker must return [] — not raise InvalidFlowIdException —
    when the top UserFlowStackFrame references a flow removed from the model.

    Regression guard for the frame.flow() → flow_by_id() hardening in
    HandleCodeChangeCommand.run_command_on_tracker.
    """
    flows = flows_from_str(
        """
        flows:
          foo:
            name: foo
            description: Flow with no translation.
            steps:
              - id: noop
                noop: true
                next: END
        """
    )
    stale_frame = UserFlowStackFrame(
        flow_id="removed_flow", step_id="1", frame_id="stale-1"
    )
    tracker = DialogueStateTracker.from_events("t", evts=[])
    tracker.update_stack(DialogueStack(frames=[stale_frame]))

    events = HandleCodeChangeCommand().run_command_on_tracker(
        tracker=tracker,
        all_flows=flows,
        original_tracker=tracker,
    )

    assert events == []
