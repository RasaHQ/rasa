from typing import List, Optional, Set, Type

import structlog
from rasa.cdu.commands import (
    CancelFlowCommand,
    Command,
    CorrectSlotsCommand,
    CorrectedSlot,
    ErrorCommand,
    ListenCommand,
    SetSlotCommand,
    StartFlowCommand,
    command_from_json,
    FreeFormAnswerCommand,
    KnowledgeAnswerCommand,
    ChitChatAnswerCommand,
)
from rasa.cdu.conversation_patterns import (
    FLOW_PATTERN_CORRECTION_ID,
    FLOW_PATTERN_INTERNAL_ERROR_ID,
    FLOW_PATTERN_LISTEN_ID,
    FLOW_PATTERN_CANCEl_ID,
)
from rasa.cdu.flow_stack import (
    STACK_FRAME_TYPES_WITH_USER_FLOWS,
    FlowStack,
    FlowStackFrame,
    StackFrameType,
)
from rasa.shared.core.constants import (
    FLOW_STACK_SLOT,
)
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import (
    END_STEP,
    Flow,
    FlowStep,
    FlowsList,
    QuestionFlowStep,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import COMMANDS


structlogger = structlog.get_logger()


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
    # assert that there is only at max one cancel flow command
    assert sum(isinstance(c, CancelFlowCommand) for c in commands) <= 1

    # assert that free form answer commands are only at the beginning of the list
    free_form_answer_commands = [
        c for c in commands if isinstance(c, FreeFormAnswerCommand)
    ]
    assert free_form_answer_commands == commands[: len(free_form_answer_commands)]

    # assert that there is at max only one correctslots command
    assert sum(isinstance(c, CorrectSlotsCommand) for c in commands) <= 1


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

    user_step, user_flow = flow_stack.topmost_user_frame(all_flows)

    current_top_flow = flow_stack.top_flow(all_flows)

    commands = clean_up_commands(commands, tracker, all_flows)

    events: List[Event] = []

    # commands need to be reversed to make sure they end up in the right order
    # on the stack. e.g. if there multiple start flow commands, the first one
    # should be on top of the stack. this is achieved by reversing the list
    # and then pushing the commands onto the stack in the reversed order.
    reversed_commands = list(reversed(commands))

    validate_state_of_commands(commands)

    for command in reversed_commands:
        if isinstance(command, CorrectSlotsCommand):
            structlogger.debug("command_executor.correct_slots", command=command)
            proposed_slots = {c.name: c.value for c in command.corrected_slots}

            reset_step = _find_earliest_updated_question(
                user_step, user_flow, proposed_slots
            )
            context = {
                "corrected_slots": proposed_slots,
                "corrected_reset_point": {
                    "id": user_flow.id if user_flow else None,
                    "step_id": reset_step.id if reset_step else None,
                },
            }
            correction_frame = FlowStackFrame(
                flow_id=FLOW_PATTERN_CORRECTION_ID,
                frame_type=StackFrameType.CORRECTION,
                context=context,
            )
            if (
                not current_top_flow
                or current_top_flow.id != FLOW_PATTERN_CORRECTION_ID
            ):
                flow_stack.push(correction_frame)
            else:
                # wrap up the previous correction flow
                flow_stack.frames[-1].step_id = END_STEP
                # push a new correction flow
                flow_stack.push(
                    correction_frame,
                    # we allow the previous correction to finish first before
                    # starting the new one
                    index=-1,
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

            canceled_frames = []
            original_frames = FlowStack.from_dict(original_stack_dump).frames
            for i, frame in enumerate(reversed(original_frames)):
                # Setting the stack frame to the end step so it is properly
                # wrapped up by the flow policy
                canceled_frames.append(len(original_frames) - i - 1)
                if user_flow and frame.flow_id == user_flow.id:
                    break

            flow_stack.push(
                FlowStackFrame(
                    flow_id=FLOW_PATTERN_CANCEl_ID,
                    frame_type=StackFrameType.REMARK,
                    context={
                        "canceled_name": user_flow.readable_name()
                        if user_flow
                        else None,
                        "canceled_frames": canceled_frames,
                    },
                )
            )
        elif isinstance(command, ListenCommand):
            structlogger.debug("command_executor.listen", command=command)
            flow_stack.push(
                FlowStackFrame(
                    flow_id=FLOW_PATTERN_LISTEN_ID,
                    frame_type=StackFrameType.REMARK,
                )
            )
        elif isinstance(command, KnowledgeAnswerCommand):
            flow_stack.push(
                FlowStackFrame(
                    # TODO: not quite sure if we need an id here
                    flow_id="NO_FLOW",
                    frame_type=StackFrameType.DOCSEARCH,
                )
            )
        elif isinstance(command, ChitChatAnswerCommand):
            flow_stack.push(
                FlowStackFrame(
                    flow_id="NO_FLOW",
                    frame_type=StackFrameType.INTENTLESS,
                )
            )
        elif isinstance(command, ErrorCommand):
            structlogger.debug("command_executor.error", command=command)
            flow_stack.push(
                FlowStackFrame(
                    flow_id=FLOW_PATTERN_INTERNAL_ERROR_ID,
                    frame_type=StackFrameType.REMARK,
                )
            )

    # if the flow stack has changed, persist it in a set slot event
    if original_stack_dump != flow_stack.as_dict():
        events.append(SlotSet(FLOW_STACK_SLOT, flow_stack.as_dict()))
    return events


def _find_earliest_updated_question(
    current_step: Optional[FlowStep], flow: Optional[Flow], updated_slots: List[str]
) -> Optional[FlowStep]:
    """Find the question that was updated."""
    if not flow or not current_step:
        return None
    asked_question_steps = flow.previously_asked_questions(current_step.id)

    for question_step in reversed(asked_question_steps):
        if question_step.question in updated_slots:
            return question_step
    return None


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
    top_flow_step = flow_stack.top_flow_step(all_flows)

    current_question = (
        top_flow_step.question if isinstance(top_flow_step, QuestionFlowStep) else None
    )

    asked_questions = set()

    for frame in reversed(flow_stack.frames):
        if not (flow := all_flows.flow_by_id(frame.flow_id)):
            break

        for q in flow.previously_asked_questions(frame.step_id):
            if q.question != current_question:
                asked_questions.add(q.question)

        if frame.frame_type in STACK_FRAME_TYPES_WITH_USER_FLOWS:
            # as soon as we hit the first stack frame that is a "normal"
            # user defined flow we stop looking for previously asked questions
            # because we only want to ask questions that are part of the
            # current flow.
            continue

    return asked_questions


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

    startable_flow_ids = [
        f.id for f in all_flows.underlying_flows if not f.is_handling_pattern()
    ]

    for command in commands:
        if isinstance(command, StartFlowCommand) and command.flow in flows_on_the_stack:
            structlogger.debug(
                "command_executor.skip_command.already_started_flow", command=command
            )
        elif (
            isinstance(command, StartFlowCommand)
            and command.flow not in startable_flow_ids
        ):
            structlogger.debug(
                "command_executor.skip_command.start_invalid_flow_id", command=command
            )
        elif isinstance(command, StartFlowCommand):
            flows_on_the_stack.add(command.flow)
            clean_commands.append(command)
        elif (
            isinstance(command, SetSlotCommand)
            and tracker.get_slot(command.name) == command.value
        ):
            # value hasn't changed, skip this one
            structlogger.debug(
                "command_executor.skip_command.slot_already_set", command=command
            )

        elif isinstance(command, SetSlotCommand) and command.name in slots_so_far:
            structlogger.debug(
                "command_executor.convert_command.correction", command=command
            )
            if (top := flow_stack.top()) and top.context:
                already_corrected_slots = top.context.get("corrected_slots", {})
            else:
                already_corrected_slots = {}

            if (
                command.name in already_corrected_slots
                and already_corrected_slots[command.name] == command.value
            ):
                structlogger.debug(
                    "command_executor.skip_command.slot_already_corrected",
                    command=command,
                )
                continue

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
        elif isinstance(command, FreeFormAnswerCommand):
            structlogger.debug(
                "command_executor.prepend_command.free_form_answer", command=command
            )
            clean_commands.insert(0, command)
        else:
            clean_commands.append(command)
    return clean_commands
