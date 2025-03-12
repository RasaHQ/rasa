import uuid

import jsonpatch
import pytest

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
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.engine.language import Language
from rasa.shared.core.events import DialogueStackUpdated, FlowCancelled
from rasa.shared.core.slots import StrictCategoricalSlot
from rasa.shared.core.trackers import DialogueStateTracker
from tests.utilities import flows_from_str


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


def test_select_canceled_frames_empty_stack():
    stack = DialogueStack.empty()

    with pytest.raises(ValueError):
        # this shouldn't actually, happen. if the stack is empty we shouldn't
        # try to cancel anything.
        CancelFlowCommand.select_canceled_frames(stack)


def test_select_canceled_frames_raises_if_frame_not_found():
    stack = DialogueStack.empty()

    with pytest.raises(ValueError):
        # can't cacenl if there is no user flow on the stack. in reality
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


def test_run_command_on_tracker_during_clarify():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            name: foo flow
            steps:
            - id: collect_test_slot
              collect: test_slot
              block_digressions: true

          baz:
            description: flow baz
            name: baz flow
            steps:
            - id: action_hello
              action: action_hello

          bar:
            description: flow bar
            name: bar flow
            steps:
            - id: utter_list_functions
              action: utter_list_functions
        """
    )

    tracker = DialogueStateTracker.from_events(
        uuid.uuid4().hex,
        evts=[],
    )
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "bar",
                    "step_id": "utter_list_functions",
                    "frame_id": "some-frame-id",
                },
                {
                    "type": "pattern_continue_interrupted",
                    "frame_type": "regular",
                    "flow_id": "pattern_continue_interrupted",
                    "step_id": "START",
                    "frame_id": "continue-interrupted-frame-id-1",
                    "previous_flow_name": "bar flow",
                },
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "baz",
                    "step_id": "action_hello",
                    "frame_id": "some-other-frame-id",
                },
                {
                    "type": "pattern_continue_interrupted",
                    "frame_type": "regular",
                    "flow_id": "pattern_continue_interrupted",
                    "step_id": "START",
                    "frame_id": "continue-interrupted-frame-id-2",
                    "previous_flow_name": "baz flow",
                },
                {
                    "type": "pattern_clarification",
                    "frame_type": "patter_clarification",
                    "clarification_options": "baz flow, bar flow",
                    "frame_id": "clarify-frame-id",
                    "step_id": "utter_clarify",
                    "names": ["baz flow", "bar flow"],
                },
            ]
        )
    )
    command = CancelFlowCommand()

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 3

    # the first events should be flow canceled events
    assert events[0] == FlowCancelled("baz", "action_hello")
    assert events[1] == FlowCancelled("bar", "utter_list_functions")

    dialogue_stack_event = events[2]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    # a cancel flow should have been added for both clarification options
    # and the pre-existing 5 frames should still be on the stack
    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 7

    assert dialogue_stack_dump[5]["type"] == "pattern_cancel_flow"
    assert dialogue_stack_dump[5]["flow_id"] == "pattern_cancel_flow"
    assert dialogue_stack_dump[5]["step_id"] == "START"
    assert dialogue_stack_dump[5]["canceled_name"] == "baz flow"
    assert dialogue_stack_dump[5]["canceled_frames"] == [
        "clarify-frame-id",
        "continue-interrupted-frame-id-2",
        "some-other-frame-id",
    ]

    assert dialogue_stack_dump[6]["type"] == "pattern_cancel_flow"
    assert dialogue_stack_dump[6]["flow_id"] == "pattern_cancel_flow"
    assert dialogue_stack_dump[6]["step_id"] == "START"
    assert dialogue_stack_dump[6]["canceled_name"] == "bar flow"
    assert dialogue_stack_dump[6]["canceled_frames"] == [
        "continue-interrupted-frame-id-1",
        "some-frame-id",
    ]


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
