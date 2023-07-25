from typing import List, Set, Type

import structlog
from rasa.cdu.commands import (
    CancelFlowCommand,
    Command,
    CorrectSlotsCommand,
    CorrectedSlot,
    ErrorCommand,
    HandleInterruptionCommand,
    ListenCommand,
    SetSlotCommand,
    StartFlowCommand,
    command_from_json,
)
from rasa.cdu.flow_stack import FlowStack, FlowStackFrame, StackFrameType
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.constants import (
    CANCELLED_FLOW_SLOT,
    CORRECTED_SLOTS_SLOT,
    FLOW_STACK_SLOT,
)
from rasa.shared.core.events import Event, SlotSet, UserUttered
from rasa.shared.core.flows.flow import FlowsList, QuestionFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import COMMANDS


structlogger = structlog.get_logger()

FLOW_PATTERN_CORRECTION_ID = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "correction"

FLOW_PATTERN_CANCEl_ID = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "cancel_flow"

FLOW_PATTERN_LISTEN_ID = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "listen"

FLOW_PATTERN_INTERNAL_ERROR_ID = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "internal_error"


def contains_command(commands: List[Command], typ: Type[Command]) -> bool:
    """Check if a list of commands contains a command of a given type.

    Example:
        >>> contains_command([ListenCommand()], ListenCommand)
        True

    Args:
        commands: The commands to check.
        typ: The type of command to check for.

    Returns:
    `True` if the list of commands contains a command of the given type.
    """
    return any(isinstance(command, typ) for command in commands)


def _slot_sets_after_latest_message(tracker: DialogueStateTracker) -> List[SlotSet]:
    """Get all slot sets after the latest message.

    Args:
        tracker: The tracker to get the slot sets from.

    Returns:
    All slot sets after the latest user message.
    """
    if not tracker.latest_message:
        return []

    slot_sets = []

    for event in reversed(tracker.applied_events()):
        if isinstance(event, UserUttered):
            break
        elif isinstance(event, SlotSet):
            slot_sets.append(event)
    return slot_sets


def _get_commands_from_tracker(tracker: DialogueStateTracker) -> List[Command]:
    """Extracts the commands from the tracker.

    Args:
        tracker: The tracker containing the conversation history up to now.

    Returns:
    The commands.
    """
    if tracker.latest_message:
        dumped_commands = tracker.latest_message.parse_data.get(COMMANDS) or []
        assert isinstance(dumped_commands, list)
        return [command_from_json(command) for command in dumped_commands]
    else:
        return []


def validate_state_of_commands(commands: List[Command]) -> None:
    """Validates the state of the commands."""
    # assert that that at max there is only one cancel flow command at
    # the beginning of the list of commands
    assert len([c for c in commands if isinstance(c, CancelFlowCommand)]) <= 1

    # assert that interrupt commands are only at the beginning of the list
    interrupt_commands = [
        c for c in commands if isinstance(c, HandleInterruptionCommand)
    ]
    assert interrupt_commands == commands[: len(interrupt_commands)]

    # assert that there is at max only one correctslots command
    assert len([c for c in commands if isinstance(c, CorrectSlotsCommand)]) <= 1


def execute_commands(
    tracker: DialogueStateTracker, all_flows: FlowsList
) -> List[Event]:
    """Executes a list of commands.

    Args:
        commands: The commands to execute.
        tracker: The tracker to execute the commands on.
        all_flows: All flows.

    Returns:
    A tuple of the action to execute and the events that were created.
    """
    commands: List[Command] = _get_commands_from_tracker(tracker)
    flow_stack = FlowStack.from_tracker(tracker)
    original_stack_dump = flow_stack.as_dict()

    current_top_flow = flow_stack.top_flow(all_flows)

    commands = clean_up_commands(commands, tracker, all_flows)

    events: List[Event] = []

    # commands need to be reversed to make sure they end up in the right order
    # on the stack. e.g. if there multiple start flow commands, the first one
    # should be on top of the stack. this is achieved by reversing the list
    # and then pushing the commands onto the stack in the reversed order.
    reversed_commands = list(reversed(commands))

    validate_state_of_commands(commands)

    for i, command in enumerate(reversed_commands):
        if isinstance(command, CorrectSlotsCommand):
            structlogger.debug("command_executor.correct_slots", command=command)
            for correction in command.corrected_slots:
                events.append(SlotSet(correction.name, correction.value))
            events.append(
                SlotSet(CORRECTED_SLOTS_SLOT, [s.name for s in command.corrected_slots])
            )
            flow_stack.push(
                FlowStackFrame(
                    flow_id=FLOW_PATTERN_CORRECTION_ID,
                    frame_type=StackFrameType.CORRECTION,
                )
            )
        elif isinstance(command, SetSlotCommand):
            structlogger.debug("command_executor.set_slot", command=command)
            events.append(SlotSet(command.name, command.value))
        elif isinstance(command, StartFlowCommand):
            if current_top_flow:
                frame_type = StackFrameType.INTERRUPT
            else:
                frame_type = StackFrameType.REGULAR
            structlogger.debug("command_executor.start_flow", command=command)
            flow_stack.push(FlowStackFrame(flow_id=command.flow, frame_type=frame_type))
        elif isinstance(command, CancelFlowCommand):
            if not current_top_flow:
                structlogger.debug(
                    "command_executor.skip_cancel_flow.no_active_flow", command=command
                )
                continue
            # in between the prediction and this canceling command, we might have
            # added some stack frames. hence, we can't just cancle the current top frame
            # but need to find the frame that was at the top before we started
            # processing the commands.
            for idx, frame in enumerate(flow_stack.frames):
                if frame.flow_id == current_top_flow.id:
                    structlogger.debug("command_executor.cancel_flow", command=command)
                    del flow_stack.frames[idx]
            events.append(SlotSet(CANCELLED_FLOW_SLOT, current_top_flow.id))
            flow_stack.push(
                FlowStackFrame(
                    flow_id=FLOW_PATTERN_CANCEl_ID,
                    # TODO: the stack frame type should be renamed
                    frame_type=StackFrameType.CORRECTION,
                )
            )
        elif isinstance(command, ListenCommand):
            structlogger.debug("command_executor.listen", command=command)
            flow_stack.push(
                FlowStackFrame(
                    flow_id=FLOW_PATTERN_LISTEN_ID,
                    # TODO: the stack frame type should be renamed
                    frame_type=StackFrameType.CORRECTION,
                )
            )
        elif isinstance(command, HandleInterruptionCommand):
            flow_stack.push(
                FlowStackFrame(
                    # TODO: not quite sure if we need an id here
                    flow_id="NO_FLOW",
                    frame_type=StackFrameType.DOCSEARCH,
                )
            )
        elif isinstance(command, ErrorCommand):
            structlogger.debug("command_executor.error", command=command)
            flow_stack.push(
                FlowStackFrame(
                    flow_id=FLOW_PATTERN_INTERNAL_ERROR_ID,
                    frame_type=StackFrameType.CORRECTION,
                )
            )

    # if the flow stack has changed, persist it in a set slot event
    if original_stack_dump != flow_stack.as_dict():
        events.append(SlotSet(FLOW_STACK_SLOT, flow_stack.as_dict()))
    return events


def filled_slots_for_active_flow(
    tracker: DialogueStateTracker, all_flows: FlowsList
) -> Set[str]:
    """Get all slots that have been filled for the current flow.

    Args:
        tracker: The tracker to get the filled slots from.
        all_flows: All flows.

    Returns:
    All slots that have been filled for the current flow.
    """
    flow_stack = FlowStack.from_tracker(tracker)

    top_flow = flow_stack.top_flow(all_flows)
    top_flow_step = flow_stack.top_flow_step(all_flows)

    current_question = (
        top_flow_step.question if isinstance(top_flow_step, QuestionFlowStep) else None
    )

    if top_flow_step is not None and top_flow is not None:
        return {
            q.question
            for q in top_flow.previously_asked_questions(top_flow_step.id)
            if q.question != current_question
        }
    else:
        return set()


def clean_up_commands(
    commands: List[Command], tracker: DialogueStateTracker, all_flows: FlowsList
) -> List[Command]:
    """Clean up a list of commands.

    This will remove commands that are not necessary anymore, e.g. because the slot
    they set is already set to the same value. It will also remove commands that
    start a flow that is already on the stack.

    Args:
        commands: The commands to clean up.
        tracker: The tracker to clean up the commands for.
        all_flows: All flows.

    Returns:
    The cleaned up commands.
    """
    flow_stack = FlowStack.from_tracker(tracker)

    flows_on_the_stack = {f.flow_id for f in flow_stack.frames}

    slots_so_far = filled_slots_for_active_flow(tracker, all_flows)

    clean_commands: List[Command] = []

    for command in commands:
        if isinstance(command, StartFlowCommand) and command.flow in flows_on_the_stack:
            structlogger.debug(
                "command_executor.skip_command.already_started_flow", command=command
            )
            continue

        if (
            isinstance(command, SetSlotCommand)
            and tracker.get_slot(command.name) == command.value
        ):
            # value hasn't changed, skip this one
            structlogger.debug(
                "command_executor.skip_command.slot_already_set", command=command
            )
            continue

        if isinstance(command, SetSlotCommand) and command.name in slots_so_far:
            structlogger.debug(
                "command_executor.convert_command.correction", command=command
            )

            corrected_slot = CorrectedSlot(command.name, command.value)
            for c in clean_commands:
                if isinstance(c, CorrectSlotsCommand):
                    c.corrected_slots.append(corrected_slot)
                    break
            else:
                clean_commands.append(
                    CorrectSlotsCommand(corrected_slots=[corrected_slot])
                )

        elif isinstance(command, CancelFlowCommand) and contains_command(
            clean_commands, CancelFlowCommand
        ):
            structlogger.debug(
                "command_executor.skip_command.already_cancelled_flow", command=command
            )
            continue
        elif isinstance(command, HandleInterruptionCommand):
            structlogger.debug(
                "command_executor.prepend_command.handle_interruption", command=command
            )
            clean_commands.insert(0, command)
            continue
        else:
            clean_commands.append(command)

    return clean_commands
