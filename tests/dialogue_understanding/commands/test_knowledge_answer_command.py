from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.knowledge_answer_command import (
    KnowledgeAnswerCommand,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
)
from rasa.shared.core.events import AgentInterrupted, DialogueStackUpdated, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


def test_name_of_command():
    # names of commands should not change as they are part of persisted
    # trackers
    assert KnowledgeAnswerCommand.command() == "knowledge"


def test_from_dict():
    assert KnowledgeAnswerCommand.from_dict({}) == KnowledgeAnswerCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered("hi", {"name": "greet"}),
        ],
    )
    command = KnowledgeAnswerCommand()

    events = command.run_command_on_tracker(tracker, [], tracker)
    assert len(events) == 1
    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 1

    frame = updated_stack.frames[0]
    assert frame.type() == "pattern_search"


def test_to_dsl_default():
    command = KnowledgeAnswerCommand()
    assert command.to_dsl() == "SearchAndReply()"


def test_from_dsl():
    assert KnowledgeAnswerCommand.from_dsl(None) == KnowledgeAnswerCommand()


def test_regex_pattern_default():
    assert KnowledgeAnswerCommand.regex_pattern() == r"SearchAndReply\(\)"


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = KnowledgeAnswerCommand()
    assert command.to_dsl() == "provide info"

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert (
        KnowledgeAnswerCommand.regex_pattern() == r"""^[\s\W\d]*provide info['"`]*$"""
    )

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_is_instance_of_prompt_command():
    # Check if the command adheres to the PromptCommand protocol.
    assert isinstance(KnowledgeAnswerCommand(), PromptCommand) is True


def test_to_dsl_v3_command_syntax():
    # Set the syntax version to v3 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    command = KnowledgeAnswerCommand()
    assert command.to_dsl() == "search and reply"

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v3_command_syntax():
    # Set the syntax version to v3 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    assert (
        KnowledgeAnswerCommand.regex_pattern()
        == r"""^[\s\W\d]*search and reply['"`]*$"""
    )

    # Reset the syntax version to the default, otherwise it will affect other tests.
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

    command = KnowledgeAnswerCommand()
    events = command.run_command_on_tracker(tracker, all_flows, original_tracker)

    # Check that an AgentInterrupted event is created
    assert any(
        isinstance(e, AgentInterrupted)
        and e.agent_id == "car-research"
        and e.flow_id == "car_research"
        for e in events
        if isinstance(e, AgentInterrupted)
    )
