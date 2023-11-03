from rasa.dialogue_understanding.commands import SkipQuestionCommand
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.events import SlotSet
from rasa.shared.core.flows.yaml_flows_io import flows_from_str
from rasa.shared.core.trackers import DialogueStateTracker


def test_command_name():
    # names of commands should not change as they are part of persisted
    # trackers
    assert SkipQuestionCommand.command() == "skip question"


def test_from_dict():
    assert SkipQuestionCommand.from_dict({}) == SkipQuestionCommand()


def test_run_command_on_tracker_without_flows():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = SkipQuestionCommand()

    assert command.run_command_on_tracker(tracker, [], tracker) == []


def test_run_command_on_tracker():
    all_flows = flows_from_str(
        """
        flows:
            my_flow:
                steps:
                - id: collect_foo
                  collect: foo
                  next: collect_bar
                - id: collect_bar
                  collect: bar
        """
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            SlotSet(
                DIALOGUE_STACK_SLOT,
                [
                    {
                        "type": "flow",
                        "flow_id": "my_flow",
                        "step_id": "collect_bar",
                        "frame_id": "some-frame-id",
                    },
                ],
            ),
        ],
    )
    command = SkipQuestionCommand()

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, SlotSet)
    assert dialogue_stack_event.key == DIALOGUE_STACK_SLOT

    dialogue_stack_dump = dialogue_stack_event.value
    # flow should still be on the stack and a skip question pattern flow should have
    # been added
    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 2

    assert dialogue_stack_dump[1]["type"] == "pattern_skip_question"
    assert dialogue_stack_dump[1]["flow_id"] == "pattern_skip_question"
    assert dialogue_stack_dump[1]["step_id"] == "START"
