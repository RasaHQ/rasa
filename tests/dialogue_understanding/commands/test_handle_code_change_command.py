import pytest

from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.commands.handle_code_change_command import (
    HandleCodeChangeCommand,
)
from rasa.core.actions.action_clean_stack import ActionCleanStack

from rasa.dialogue_understanding.patterns.code_change import FLOW_PATTERN_CODE_CHANGE_ID
from rasa.dialogue_understanding.processor.command_processor import execute_commands
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    UserFlowStackFrame,
    PatternFlowStackFrame,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet
from rasa.shared.core.flows.flow import (
    FlowsList,
    START_STEP,
    ContinueFlowStep,
    END_STEP,
)
from rasa.shared.core.trackers import DialogueStateTracker
from tests.dialogue_understanding.commands.test_command_processor import (
    start_bar_user_uttered,
    change_cases,
)
from rasa.shared.core.flows.yaml_flows_io import flows_from_str


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
    assert isinstance(dialogue_stack_event, SlotSet)
    assert dialogue_stack_event.key == "dialogue_stack"
    assert len(dialogue_stack_event.value) == 2

    frame = dialogue_stack_event.value[1]
    assert frame["type"] == FLOW_PATTERN_CODE_CHANGE_ID


@pytest.fixture
def about_to_be_cleaned_tracker(tracker: DialogueStateTracker, all_flows: FlowsList):
    tracker.update_with_events([start_bar_user_uttered], None)
    execute_commands(tracker, all_flows)
    changed_flows = flows_from_str(change_cases["step_id_changed"])
    execute_commands(tracker, changed_flows)
    dialogue_stack = DialogueStack.from_tracker(tracker)
    assert len(dialogue_stack.frames) == 3

    foo_frame = dialogue_stack.frames[0]
    assert isinstance(foo_frame, UserFlowStackFrame)
    assert foo_frame.flow_id == "foo"
    assert foo_frame.step_id == START_STEP

    bar_frame = dialogue_stack.frames[1]
    assert isinstance(bar_frame, UserFlowStackFrame)
    assert bar_frame.flow_id == "bar"
    assert bar_frame.step_id == START_STEP

    stack_clean_frame = dialogue_stack.frames[2]
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
    about_to_be_cleaned_tracker.update_with_events(events, None)

    dialogue_stack = DialogueStack.from_tracker(about_to_be_cleaned_tracker)
    assert len(dialogue_stack.frames) == 3

    foo_frame = dialogue_stack.frames[0]
    assert isinstance(foo_frame, UserFlowStackFrame)
    assert foo_frame.flow_id == "foo"
    assert foo_frame.step_id == ContinueFlowStep.continue_step_for_id(END_STEP)

    bar_frame = dialogue_stack.frames[1]
    assert isinstance(bar_frame, UserFlowStackFrame)
    assert bar_frame.flow_id == "bar"
    assert bar_frame.step_id == ContinueFlowStep.continue_step_for_id(END_STEP)
