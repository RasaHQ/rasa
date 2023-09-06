from typing import List, Optional, Set, Type

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
from rasa.cdu.patterns.cancel import CancelPatternFlowStackFrame
from rasa.cdu.patterns.clarify import ClarifyPatternFlowStackFrame
from rasa.cdu.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.cdu.patterns.correction import (
    FLOW_PATTERN_CORRECTION_ID,
    CorrectionPatternFlowStackFrame,
)
from rasa.cdu.patterns.internal_error import InternalErrorPatternFlowStackFrame
from rasa.cdu.stack.dialogue_stack import DialogueStack
from rasa.cdu.stack.frames import (
    ChitChatStackFrame,
    BaseFlowStackFrame,
    UserFlowStackFrame,
    SearchStackFrame,
)
from rasa.cdu.stack.frames.flow_frame import FlowStackFrameType
from rasa.cdu.stack.utils import top_flow_frame, top_user_flow_frame
from rasa.shared.core.constants import (
    DIALOGUE_STACK_SLOT,
)
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import (
    END_STEP,
    ContinueFlowStep,
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
    """Validates the state of the commands.

    We have some invariants that should always hold true. This function
    checks if they do. Executing the commands relies on these invariants.

    We cleanup the commands before executing them, so the cleanup should
    always make sure that these invariants hold true - no matter the commands
    that are provided.

    Args:
        commands: The commands to validate.
    """
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
    dialogue_stack = DialogueStack.from_tracker(tracker)
    original_stack_dump = dialogue_stack.as_dict()

    current_user_frame = top_user_flow_frame(dialogue_stack)
    current_top_flow = (
        current_user_frame.flow(all_flows) if current_user_frame else None
    )

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
            top_non_collect_info_frame = top_flow_frame(dialogue_stack)
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
                current_user_frame, proposed_slots, all_flows
            )
            correction_frame = CorrectionPatternFlowStackFrame(
                is_reset_only=is_reset_only,
                corrected_slots=proposed_slots,
                reset_flow_id=current_user_frame.flow_id
                if current_user_frame
                else None,
                reset_step_id=reset_step.id if reset_step else None,
            )

            if top_non_collect_info_frame.flow_id != FLOW_PATTERN_CORRECTION_ID:
                dialogue_stack.push(correction_frame)
            else:
                # wrap up the previous correction flow
                for i, frame in enumerate(reversed(dialogue_stack.frames)):
                    if isinstance(frame, BaseFlowStackFrame):
                        frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)
                        if frame.frame_id == top_non_collect_info_frame.frame_id:
                            break

                # push a new correction flow
                dialogue_stack.push(
                    correction_frame,
                    # we allow the previous correction to finish first before
                    # starting the new one
                    index=len(dialogue_stack.frames) - i - 1,
                )
        elif isinstance(command, SetSlotCommand):
            structlogger.debug("command_executor.set_slot", command=command)
            events.append(SlotSet(command.name, command.value))
        elif isinstance(command, StartFlowCommand):
            if current_top_flow:
                frame_type = FlowStackFrameType.INTERRUPT
            else:
                frame_type = FlowStackFrameType.REGULAR
            structlogger.debug("command_executor.start_flow", command=command)
            dialogue_stack.push(
                UserFlowStackFrame(flow_id=command.flow, frame_type=frame_type)
            )
        elif isinstance(command, CancelFlowCommand):
            if not current_top_flow:
                structlogger.debug(
                    "command_executor.skip_cancel_flow.no_active_flow", command=command
                )
                continue

            canceled_frames = []
            original_frames = DialogueStack.from_dict(original_stack_dump).frames
            # we need to go through the original stack dump in reverse order
            # to find the frames that were canceled. we cancel everthing from
            # the top of the stack until we hit the user flow that was canceled.
            # this will also cancel any patterns put ontop of that user flow,
            # e.g. corrections.
            for frame in reversed(original_frames):
                canceled_frames.append(frame.frame_id)
                if (
                    current_user_frame
                    and isinstance(frame, BaseFlowStackFrame)
                    and frame.flow_id == current_user_frame.flow_id
                ):
                    break

            dialogue_stack.push(
                CancelPatternFlowStackFrame(
                    canceled_name=current_user_frame.flow(all_flows).readable_name()
                    if current_user_frame
                    else None,
                    canceled_frames=canceled_frames,
                )
            )
        elif isinstance(command, KnowledgeAnswerCommand):
            dialogue_stack.push(SearchStackFrame())
        elif isinstance(command, ChitChatAnswerCommand):
            dialogue_stack.push(ChitChatStackFrame())
        elif isinstance(command, ClarifyCommand):
            relevant_flows = [all_flows.flow_by_id(opt) for opt in command.options]
            names = [
                flow.readable_name() for flow in relevant_flows if flow is not None
            ]
            dialogue_stack.push(ClarifyPatternFlowStackFrame(names=names))
        elif isinstance(command, ErrorCommand):
            structlogger.debug("command_executor.error", command=command)
            dialogue_stack.push(InternalErrorPatternFlowStackFrame())

    # if the dialogue stack has changed, persist it in a set slot event
    if original_stack_dump != dialogue_stack.as_dict():
        events.append(SlotSet(DIALOGUE_STACK_SLOT, dialogue_stack.as_dict()))
    return events


def _find_earliest_updated_collect_info(
    current_user_flow_frame: Optional[UserFlowStackFrame],
    updated_slots: List[str],
    all_flows: FlowsList,
) -> Optional[FlowStep]:
    """Find the collect infos that was updated."""
    if not current_user_flow_frame:
        return None
    flow = current_user_flow_frame.flow(all_flows)
    step = current_user_flow_frame.step(all_flows)
    asked_collect_info_steps = flow.previously_asked_collect_information(step.id)

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
    dialogue_stack = DialogueStack.from_tracker(tracker)

    asked_collect_information = set()

    for frame in reversed(dialogue_stack.frames):
        if not isinstance(frame, BaseFlowStackFrame):
            break
        flow = frame.flow(all_flows)
        for q in flow.previously_asked_collect_information(frame.step_id):
            asked_collect_information.add(q.collect_information)

        if isinstance(frame, UserFlowStackFrame):
            # as soon as we hit the first stack frame that is a "normal"
            # user defined flow we stop looking for previously asked collect infos
            # because we only want to ask collect infos that are part of the
            # current flow.
            break

    return asked_collect_information


def get_current_collect_information(
    dialogue_stack: DialogueStack, all_flows: FlowsList
) -> Optional[CollectInformationFlowStep]:
    """Get the current collect information if the conversation is currently in one.

    If we are currently in a collect information step, the stack should have at least
    two frames. The top frame is the collect information pattern and the frame below
    is the flow that triggered the collect information pattern. We can use the flow
    id to get the collect information step from the flow.

    Args:
        dialogue_stack: The dialogue stack.
        all_flows: All flows.

    Returns:
    The current collect information if the conversation is currently in one,
    `None` otherwise.
    """
    if not (top_frame := dialogue_stack.top()):
        # we are currently not in a flow
        return None

    if not isinstance(top_frame, CollectInformationPatternFlowStackFrame):
        # we are currently not in a collect information
        return None

    if len(dialogue_stack.frames) <= 1:
        # for some reason only the collect information pattern step is on the stack
        # but no flow that triggered it. this should never happen.
        structlogger.warning(
            "command_executor.get_current_collect information.no_flow_on_stack",
            stack=dialogue_stack,
        )
        return None

    frame_that_triggered_collect_infos = dialogue_stack.frames[-2]
    if not isinstance(frame_that_triggered_collect_infos, BaseFlowStackFrame):
        # this is a failure, if there is a frame, we should be able to get the
        # step from it
        structlogger.warning(
            "command_executor.get_current_collect_information.no_step_for_frame",
            frame=frame_that_triggered_collect_infos,
        )
        return None

    step = frame_that_triggered_collect_infos.step(all_flows)
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
    dialogue_stack = DialogueStack.from_tracker(tracker)

    flows_on_the_stack = {
        f.flow_id for f in dialogue_stack.frames if isinstance(f, UserFlowStackFrame)
    }

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
                dialogue_stack, all_flows
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
            if (top := top_flow_frame(dialogue_stack)) and isinstance(
                top, CorrectionPatternFlowStackFrame
            ):
                already_corrected_slots = top.corrected_slots
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
