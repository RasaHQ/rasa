import jsonpatch

from rasa.dialogue_understanding.commands.chit_chat_answer_command import (
    ChitChatAnswerCommand,
)
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
)
from rasa.shared.core.events import AgentInterrupted, DialogueStackUpdated, UserUttered
from rasa.shared.core.flows.flows_list import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


def test_name_of_command():
    # names of commands should not change as they are part of persisted
    # trackers
    assert ChitChatAnswerCommand.command() == "chitchat"


def test_from_dict():
    assert ChitChatAnswerCommand.from_dict({}) == ChitChatAnswerCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered("hi", {"name": "greet"}),
        ],
    )
    command = ChitChatAnswerCommand()

    events = command.run_command_on_tracker(tracker, [], tracker)
    assert len(events) == 1
    dialogue_stack_event = events[0]

    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    assert len(dialogue_stack_dump) == 1

    frame = dialogue_stack_dump[0]
    assert frame["type"] == "pattern_chitchat"


def test_to_dsl_default():
    command = ChitChatAnswerCommand()
    assert command.to_dsl() == "ChitChat()"


def test_from_dsl():
    assert ChitChatAnswerCommand.from_dsl(None) == ChitChatAnswerCommand()


def test_regex_pattern_default():
    assert ChitChatAnswerCommand.regex_pattern() == r"ChitChat\(\)"


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = ChitChatAnswerCommand()
    assert command.to_dsl() == "offtopic reply"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert (
        ChitChatAnswerCommand.regex_pattern() == r"""^[\s\W\d]*offtopic reply['"`]*$"""
    )

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_is_instance_of_prompt_command():
    # Check if the command adheres to the PromptCommand protocol.
    assert isinstance(ChitChatAnswerCommand(), PromptCommand) is True


def test_to_dsl_v3_command_syntax():
    # Set the syntax version to v3 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    command = ChitChatAnswerCommand()
    assert command.to_dsl() == "offtopic reply"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v3_command_syntax():
    # Set the syntax version to v3 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    assert (
        ChitChatAnswerCommand.regex_pattern() == r"""^[\s\W\d]*offtopic reply['"`]*$"""
    )

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_run_command_on_tracker_interrupts_agent_and_adds_event():
    tracker = DialogueStateTracker.from_events("test", evts=[])

    agent_frame = AgentStackFrame(
        frame_id="agent-frame",
        state=AgentState.WAITING_FOR_INPUT,
        agent_id="car-research",
        flow_id="car_research",
    )
    tracker.update_stack(DialogueStack(frames=[agent_frame]))

    all_flows = FlowsList([])
    original_tracker = tracker

    command = ChitChatAnswerCommand()
    events = command.run_command_on_tracker(tracker, all_flows, original_tracker)

    # Check that an AgentInterrupted event is created
    assert any(
        isinstance(e, AgentInterrupted)
        and e.agent_id == "car-research"
        and e.flow_id == "car_research"
        for e in events
        if isinstance(e, AgentInterrupted)
    )
