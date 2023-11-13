import pytest
from rasa.dialogue_understanding.commands.clarify_command import ClarifyCommand
from rasa.shared.core.events import DialogueStackUpdated
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.flows.yaml_flows_io import flows_from_str
import jsonpatch


def test_command_name():
    # names of commands should not change as they are part of persisted
    # trackers
    assert ClarifyCommand.command() == "clarify"


def test_from_dict():
    assert ClarifyCommand.from_dict({"options": ["foo", "bar"]}) == ClarifyCommand(
        options=["foo", "bar"]
    )


def test_from_dict_fails_if_options_is_missing():
    with pytest.raises(ValueError):
        ClarifyCommand.from_dict({})


def test_run_command_skips_if_no_options():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = ClarifyCommand(options=[])

    assert command.run_command_on_tracker(tracker, [], tracker) == []


def test_run_command_skips_if_only_non_existant_flows():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = ClarifyCommand(options=["does-not-exist"])

    assert command.run_command_on_tracker(tracker, all_flows, tracker) == []


def test_run_command_ignores_non_existant_flows():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = ClarifyCommand(options=["does-not-exist", "foo"])

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1
    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())
    assert len(dialogue_stack_dump) == 1

    frame = dialogue_stack_dump[0]
    assert frame["type"] == "pattern_clarification"
    assert frame["flow_id"] == "pattern_clarification"
    assert frame["step_id"] == "START"
    assert frame["names"] == ["foo"]
    assert frame["clarification_options"] == ""


def test_run_command_uses_name_of_flow():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            name: some foo
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = ClarifyCommand(options=["foo"])

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1
    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())
    assert len(dialogue_stack_dump) == 1

    frame = dialogue_stack_dump[0]
    assert frame["type"] == "pattern_clarification"
    assert frame["names"] == ["some foo"]
