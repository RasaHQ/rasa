from typing import List, Optional, Set, Type, Dict, Any

import structlog
from rasa.cdu.commands import (
    CancelFlowCommand,
    Command,
    CorrectSlotsCommand,
    CorrectedSlot,
    ErrorCommand,
    SetSlotCommand,
    StartFlowCommand,
    command_from_json,
    FreeFormAnswerCommand,
    KnowledgeAnswerCommand,
    ChitChatAnswerCommand,
    ClarifyCommand,
)
from rasa.cdu.conversation_patterns import (
    FLOW_PATTERN_COLLECT_INFORMATION,
    FLOW_PATTERN_CORRECTION_ID,
    FLOW_PATTERN_INTERNAL_ERROR_ID,
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
    ContinueFlowStep,
    Flow,
    FlowStep,
    FlowsList,
    CollectInformationFlowStep,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import COMMANDS


structlogger = structlog.get_logger()


def contains_command(commands: List[Command], typ: Type[Command]) -> bool:
    """Check if a list of commands contains a command of a given type.

    Example:
        >>> contains_command([SetSlotCommand("foo", "bar")], SetSlotCommand)
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
            top_non_collect_info_frame = flow_stack.top(
                ignore_frame=FLOW_PATTERN_COLLECT_INFORMATION
            )
            if not top_non_collect_info_frame:
                # we shouldn't end up here as a correction shouldn't be triggered
                # if there is nothing to correct. but just in case we do, we
                # just skip the command.
                structlogger.warning(
                    "command_executor.correct_slots.no_active_flow", command=command
                )
                continue
            structlogger.debug("command_executor.correct_slots", command=command)
            proposed_slots = {c.name: c.value for c in command.corrected_slots}

            # check if all corrected slots have ask_before_filling=True
            # if this is a case, we are not correcting a value but we
            # are resetting the slots and jumping back to the first question
            is_reset_only = all(
                collect_information_step.collect_information not in proposed_slots
                or collect_information_step.ask_before_filling
                for flow in all_flows.underlying_flows
                for collect_information_step in flow.get_collect_information_steps()
            )

            reset_step = _find_earliest_updated_collect_info(
                user_step, user_flow, proposed_slots
            )
            context: Dict[str, Any] = {
                "is_reset_only": is_reset_only,
                "corrected_slots": proposed_slots,
                "corrected_reset_point": {
                    "id": user_flow.id if user_flow else None,
                    "step_id": reset_step.id if reset_step else None,
                },
            }
            correction_frame = FlowStackFrame(
                flow_id=FLOW_PATTERN_CORRECTION_ID,
                frame_type=StackFrameType.REMARK,
                context=context,
            )

            if top_non_collect_info_frame.flow_id != FLOW_PATTERN_CORRECTION_ID:
                flow_stack.push(correction_frame)
            else:
                # wrap up the previous correction flow
                for i, frame in enumerate(reversed(flow_stack.frames)):
                    frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)
                    if frame.frame_id == top_non_collect_info_frame.frame_id:
                        break

                # push a new correction flow
                flow_stack.push(
                    correction_frame,
                    # we allow the previous correction to finish first before
                    # starting the new one
                    index=len(flow_stack.frames) - i - 1,
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
            # we need to go through the original stack dump in reverse order
            # to find the frames that were canceled. we cancel everthing from
            # the top of the stack until we hit the user flow that was canceled.
            # this will also cancel any patterns put ontop of that user flow,
            # e.g. corrections.
            for frame in reversed(original_frames):
                canceled_frames.append(frame.frame_id)
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
        elif isinstance(command, ClarifyCommand):
            relevant_flows = [all_flows.flow_by_id(opt) for opt in command.options]
            names = [
                flow.readable_name() for flow in relevant_flows if flow is not None
            ]
            context = {
                "names": names,
            }
            flow_stack.push(
                FlowStackFrame(
                    flow_id="pattern_clarification",
                    frame_type=StackFrameType.REGULAR,
                    context=context,
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


def _find_earliest_updated_collect_info(
    current_step: Optional[FlowStep], flow: Optional[Flow], updated_slots: List[str]
) -> Optional[FlowStep]:
    """Find the collect infos that was updated."""
    if not flow or not current_step:
        return None
    asked_collect_info_steps = flow.previously_asked_collect_information(
        current_step.id
    )

    for collect_info_step in reversed(asked_collect_info_steps):
        if collect_info_step.collect_information in updated_slots:
            return collect_info_step
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

    asked_collect_information = set()

    for frame in reversed(flow_stack.frames):
        if not (flow := all_flows.flow_by_id(frame.flow_id)):
            break

        for q in flow.previously_asked_collect_information(frame.step_id):
            asked_collect_information.add(q.collect_information)

        if frame.frame_type in STACK_FRAME_TYPES_WITH_USER_FLOWS:
            # as soon as we hit the first stack frame that is a "normal"
            # user defined flow we stop looking for previously asked collect infos
            # because we only want to ask collect infos that are part of the
            # current flow.
            break

    return asked_collect_information


def get_current_collect_information(
    flow_stack: FlowStack, all_flows: FlowsList
) -> Optional[CollectInformationFlowStep]:
    """Get the current collect information if the conversation is currently in one.

    If we are currently in a collect information step, the stack should have at least
    two frames. The top frame is the collect information pattern and the frame below
    is the flow that triggered the collect information pattern. We can use the flow
    id to get the collect information step from the flow.

    Args:
        flow_stack: The flow stack.
        all_flows: All flows.

    Returns:
    The current collect information if the conversation is currently in one,
    `None` otherwise.
    """
    if not (top_frame := flow_stack.top()):
        # we are currently not in a flow
        return None

    if top_frame.flow_id != FLOW_PATTERN_COLLECT_INFORMATION:
        # we are currently not in a collect information
        return None

    if len(flow_stack.frames) <= 1:
        # for some reason only the collect information pattern step is on the stack
        # but no flow that triggered it. this should never happen.
        structlogger.warning(
            "command_executor.get_current_collect information.no_flow_on_stack",
            stack=flow_stack,
        )
        return None

    frame_that_triggered_collect_infos = flow_stack.frames[-2]
    if not (step := frame_that_triggered_collect_infos.step(all_flows)):
        # this is a failure, if there is a frame, we should be able to get the
        # step from it
        structlogger.warning(
            "command_executor.get_current_collect_information.no_step_for_frame",
            frame=frame_that_triggered_collect_infos,
        )
        return None

    if isinstance(step, CollectInformationFlowStep):
        # we found it!
        return step
    else:
        # this should never happen as we only push collect information patterns
        # onto the stack if there is a collect information step
        structlogger.warning(
            "command_executor.get_current_collect_information.step_not_collect_information",
            step=step,
        )
        return None


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
        elif isinstance(command, SetSlotCommand) and command.name not in slots_so_far:
            # only fill slots that belong to a collect infos that can be asked
            use_slot_fill = any(
                step.collect_information == command.name and not step.ask_before_filling
                for flow in all_flows.underlying_flows
                for step in flow.get_collect_information_steps()
            )

            if use_slot_fill:
                clean_commands.append(command)
            else:
                structlogger.debug(
                    "command_executor.skip_command.slot_not_asked_for", command=command
                )
                continue

        elif isinstance(command, SetSlotCommand) and command.name in slots_so_far:
            current_collect_info = get_current_collect_information(
                flow_stack, all_flows
            )

            if (
                current_collect_info
                and current_collect_info.collect_information == command.name
            ):
                # not a correction but rather an answer to the current collect info
                clean_commands.append(command)
                continue

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
        elif isinstance(command, ClarifyCommand):
            flows = [all_flows.flow_by_id(opt) for opt in command.options]
            clean_options = [flow.id for flow in flows if flow is not None]
            if len(clean_options) != len(command.options):
                structlogger.debug(
                    "command_executor.altered_command.dropped_clarification_options",
                    command=command,
                    original_options=command.options,
                    cleaned_options=clean_options,
                )
            if len(clean_options) == 0:
                structlogger.debug(
                    "command_executor.skip_command.empty_clarification", command=command
                )
            else:
                clean_commands.append(ClarifyCommand(clean_options))
        else:
            clean_commands.append(command)
    return clean_commands
