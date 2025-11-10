import os
from typing import Dict, List, Optional, Set, Type

import structlog

from rasa.agents.utils import (
    is_agent_completed,
    is_agent_valid,
)
from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    CannotHandleCommand,
    ChitChatAnswerCommand,
    ClarifyCommand,
    Command,
    ContinueAgentCommand,
    CorrectedSlot,
    CorrectSlotsCommand,
    FreeFormAnswerCommand,
    RepeatBotMessagesCommand,
    RestartAgentCommand,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.handle_code_change_command import (
    HandleCodeChangeCommand,
)
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.dialogue_understanding.commands.utils import (
    create_validate_frames_from_slot_set_events,
)
from rasa.dialogue_understanding.patterns.chitchat import FLOW_PATTERN_CHITCHAT
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.correction import (
    CorrectionPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.validate_slot import (
    ValidateSlotPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    BaseFlowStackFrame,
)
from rasa.dialogue_understanding.stack.utils import (
    is_pattern_active,
    top_flow_frame,
    top_user_flow_frame,
)
from rasa.engine.graph import ExecutionContext
from rasa.shared.constants import (
    RASA_PATTERN_CANNOT_HANDLE_CHITCHAT,
    ROUTE_TO_CALM_SLOT,
)
from rasa.shared.core.constants import (
    ACTION_TRIGGER_CHITCHAT,
    FLOW_HASHES_SLOT,
    SlotMappingType,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.steps.collect import CollectInformationFlowStep
from rasa.shared.core.policies.utils import contains_intentless_policy_responses
from rasa.shared.core.slot_mappings import SlotMapping
from rasa.shared.core.slots import Slot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.nlu.constants import COMMANDS

structlogger = structlog.get_logger()

CLARIFY_ON_MULTIPLE_START_FLOWS_ENV_VAR_NAME = "CLARIFY_ON_MULTIPLE_START_FLOWS"


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


def get_commands_from_tracker(tracker: DialogueStateTracker) -> List[Command]:
    """Extracts the commands from the tracker.

    Args:
        tracker: The tracker containing the conversation history up to now.

    Returns:
    The commands.
    """
    if tracker.latest_message:
        dumped_commands = tracker.latest_message.parse_data.get(COMMANDS) or []
        assert isinstance(dumped_commands, list)
        return [Command.command_from_json(command) for command in dumped_commands]
    else:
        return []


def filter_start_flow_commands(commands: List[Command]) -> List[str]:
    """Filters the start flow commands from a list of commands."""
    return [
        command.flow for command in commands if isinstance(command, StartFlowCommand)
    ]


def validate_state_of_commands(commands: List[Command]) -> None:
    """Validates the state of the commands.

    We have some invariants that should always hold true. This function
    checks if they do. Executing the commands relies on these invariants.

    We cleanup the commands before executing them, so the cleanup should
    always make sure that these invariants hold true - no matter the commands
    that are provided.

    Args:
        commands: The commands to validate.

    Raises:
        ValueError: If the state of the commands is invalid.
    """
    # check that there is only at max one cancel flow command
    if sum(isinstance(c, CancelFlowCommand) for c in commands) > 1:
        structlogger.error(
            "command_processor.validate_state_of_commands."
            "multiple_cancel_flow_commands",
            commands=[command.__class__.__name__ for command in commands],
        )
        raise ValueError("There can only be one cancel flow command.")

    # check that free form answer commands are only at the beginning of the list
    free_form_answer_commands = [
        c for c in commands if isinstance(c, FreeFormAnswerCommand)
    ]
    if free_form_answer_commands != commands[: len(free_form_answer_commands)]:
        structlogger.error(
            "command_processor.validate_state_of_commands."
            "free_form_answer_commands_not_at_beginning",
            commands=[command.__class__.__name__ for command in commands],
        )
        raise ValueError(
            "Free form answer commands must be at start of the predicted command list."
        )

    # check that there is at max only one correctslots command
    if sum(isinstance(c, CorrectSlotsCommand) for c in commands) > 1:
        structlogger.error(
            "command_processor.validate_state_of_commands."
            "multiple_correct_slots_commands",
            commands=[command.__class__.__name__ for command in commands],
        )
        raise ValueError("There can only be one correct slots command.")


def find_updated_flows(tracker: DialogueStateTracker, all_flows: FlowsList) -> Set[str]:
    """Find the set of updated flows.

    Run through the current dialogue stack and compare the flow hashes of the
    flows on the stack with those stored in the tracker.

    Args:
        tracker: The tracker.
        all_flows: All flows.

    Returns:
    A set of flow ids of those flows that have changed
    """
    stored_fingerprints: Dict[str, str] = tracker.get_slot(FLOW_HASHES_SLOT) or {}
    stack = tracker.stack

    changed_flows = set()
    for frame in stack.frames:
        if isinstance(frame, BaseFlowStackFrame):
            flow = all_flows.flow_by_id(frame.flow_id)
            if flow is None or (
                flow.id in stored_fingerprints
                and flow.fingerprint != stored_fingerprints[flow.id]
            ):
                changed_flows.add(frame.flow_id)
    return changed_flows


def calculate_flow_fingerprints(all_flows: FlowsList) -> Dict[str, str]:
    """Calculate fingerprints for all flows."""
    return {flow.id: flow.fingerprint for flow in all_flows.underlying_flows}


def execute_commands(
    tracker: DialogueStateTracker,
    all_flows: FlowsList,
    execution_context: ExecutionContext,
    story_graph: Optional[StoryGraph] = None,
    domain: Optional[Domain] = None,
) -> List[Event]:
    """Executes a list of commands.

    Args:
        commands: The commands to execute.
        tracker: The tracker to execute the commands on.
        all_flows: All flows.
        execution_context: Information about the single graph run.
        story_graph: StoryGraph object with stories available for training.
        domain: The domain of the bot.

    Returns:
        A list of the events that were created.
    """
    commands: List[Command] = get_commands_from_tracker(tracker)
    original_tracker = tracker.copy()

    updated_flows = find_updated_flows(tracker, all_flows)
    if updated_flows:
        # if there are updated flows, we need to handle the code change
        structlogger.debug(
            "command_processor.execute_commands.running_flows_were_updated",
            updated_flow_ids=updated_flows,
        )
        commands = [HandleCodeChangeCommand()]
    else:
        commands = clean_up_commands(
            commands, tracker, all_flows, execution_context, story_graph, domain
        )

    # store current flow hashes if they changed
    new_hashes = calculate_flow_fingerprints(all_flows)
    flow_hash_events: List[Event] = []
    if new_hashes != (tracker.get_slot(FLOW_HASHES_SLOT) or {}):
        flow_hash_events.append(SlotSet(FLOW_HASHES_SLOT, new_hashes))
        tracker.update_with_events(flow_hash_events)

    events: List[Event] = flow_hash_events

    # reorder commands: in case there is no active flow, we want to make sure to
    # run the start flow commands first.
    final_commands = reorder_commands(commands, tracker)

    # we need to keep track of the ValidateSlotPatternFlowStackFrame that
    # should be pushed onto the stack before executing the StartFlowCommands.
    # This is necessary to make sure that slots filled before the start of a
    # flow can be immediately validated without waiting till the flow is started
    # and completed.
    stack_frames_to_follow_commands: List[ValidateSlotPatternFlowStackFrame] = []

    validate_state_of_commands(commands)

    for command in final_commands:
        new_events = command.run_command_on_tracker(
            tracker, all_flows, original_tracker
        )

        _, stack_frames_to_follow_commands = (
            create_validate_frames_from_slot_set_events(
                tracker, new_events, stack_frames_to_follow_commands
            )
        )

        events.extend(new_events)
        tracker.update_with_events(new_events)

    new_events = push_stack_frames_to_follow_commands(
        tracker, stack_frames_to_follow_commands
    )
    events.extend(new_events)

    return remove_duplicated_set_slots(events)


def push_stack_frames_to_follow_commands(
    tracker: DialogueStateTracker, stack_frames: List
) -> List[Event]:
    """Push stack frames to follow commands."""
    new_events = []

    for frame in stack_frames:
        stack = tracker.stack
        stack.push(frame)
        new_events.extend(tracker.create_stack_updated_events(stack))
    tracker.update_with_events(new_events)
    return new_events


def remove_duplicated_set_slots(events: List[Event]) -> List[Event]:
    """Removes duplicated set slot events.

    This can happen if a slot is set multiple times in a row. We only want to
    keep the last one.

    Args:
        events: The events to optimize.

    Returns:
    The optimized events.
    """
    slots_so_far = set()

    optimized_events: List[Event] = []

    for event in reversed(events):
        if isinstance(event, SlotSet) and event.key in slots_so_far:
            # slot will be overwritten, no need to set it
            continue
        elif isinstance(event, SlotSet):
            slots_so_far.add(event.key)

        optimized_events.append(event)

    # since we reversed the original events, we need to reverse the optimized
    # events again to get them in the right order
    return list(reversed(optimized_events))


def get_current_collect_step(
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
            "command_processor.get_current_collect_step.no_flow_on_stack",
        )
        return None

    frame_that_triggered_collect_infos = dialogue_stack.frames[-2]
    if not isinstance(frame_that_triggered_collect_infos, BaseFlowStackFrame):
        # this is a failure, if there is a frame, we should be able to get the
        # step from it
        structlogger.warning(
            "command_processor.get_current_collect_step.no_step_for_frame",
            frame=frame_that_triggered_collect_infos.frame_id,
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
            "command_processor.get_current_collect_step.step_not_collect",
            step=step,
        )
        return None


def clean_up_commands(
    commands: List[Command],
    tracker: DialogueStateTracker,
    all_flows: FlowsList,
    execution_context: ExecutionContext,
    story_graph: Optional[StoryGraph] = None,
    domain: Optional[Domain] = None,
) -> List[Command]:
    """Clean up a list of commands.

    This will clean commands that are not necessary anymore. e.g. removing commands
    where the slot they correct was previously corrected to the same value, grouping
    all slot corrections into one command, removing duplicate cancel flow commands
    and moving free form answer commands to the beginning of the list (to be last when
    reversed.)

    Args:
        commands: The commands to clean up.
        tracker: The tracker to clean up the commands for.
        all_flows: All flows.
        execution_context: Information about a single graph run.
        story_graph: StoryGraph object with stories available for training.
        domain: The domain of the bot.

    Returns:
    The cleaned up commands.
    """
    domain = domain if domain else Domain.empty()

    clean_commands: List[Command] = []

    for command in commands:
        if isinstance(command, SetSlotCommand):
            clean_commands = clean_up_slot_command(
                clean_commands, command, tracker, all_flows
            )

        elif isinstance(command, CancelFlowCommand):
            clean_commands = clean_up_cancel_flow_command(
                clean_commands, tracker, command
            )

        # if there is a cannot handle command after the previous step,
        # we don't want to add another one
        elif isinstance(command, CannotHandleCommand) and contains_command(
            clean_commands, CannotHandleCommand
        ):
            structlogger.debug(
                "command_processor"
                ".clean_up_commands"
                ".skip_command_already_has_cannot_handle",
                command=command,
            )

        elif isinstance(command, StartFlowCommand):
            clean_commands = clean_up_start_flow_command(
                clean_commands, tracker, command
            )

        # handle chitchat command differently from other free-form answer commands
        elif isinstance(command, ChitChatAnswerCommand):
            clean_commands = clean_up_chitchat_command(
                clean_commands,
                command,
                all_flows,
                execution_context,
                domain,
                story_graph,
            )

        elif isinstance(command, FreeFormAnswerCommand):
            structlogger.debug(
                "command_processor.clean_up_commands.prepend_command_freeform_answer",
                command=command,
            )
            clean_commands.insert(0, command)

        # drop all clarify commands if there are more commands. Otherwise, we might
        # get a situation where two questions are asked at once.
        elif isinstance(command, ClarifyCommand) and len(commands) > 1:
            clean_commands = clean_up_clarify_command(clean_commands, commands, command)
            if command not in clean_commands:
                structlogger.debug(
                    "command_processor.clean_up_commands."
                    "drop_clarify_given_other_commands",
                    command=command,
                )

        # Keep the Restart agent commands only if the command is referencing
        # a valid agent that was already completed
        elif isinstance(command, RestartAgentCommand):
            if not is_agent_valid(command.agent_id):
                structlogger.debug(
                    "command_processor.clean_up_commands.skip_restart_agent_invalid_agent",
                    agent_id=command.agent_id,
                    command=command,
                )
            elif not is_agent_completed(tracker, command.agent_id):
                structlogger.debug(
                    "command_processor.clean_up_commands.skip_restart_agent_not_completed",
                    agent_id=command.agent_id,
                    command=command,
                )
            else:
                clean_commands.append(command)

        # Clean up Continue agent commands if there is currently no active agent
        elif isinstance(command, ContinueAgentCommand):
            if not tracker.stack.agent_is_active():
                structlogger.debug(
                    "command_processor.clean_up_commands.skip_continue_agent_no_active_agent",
                    command=command,
                )
            else:
                clean_commands.append(command)

        else:
            clean_commands.append(command)

    clean_commands = _process_multiple_start_flow_commands(clean_commands, tracker)

    # ensure that there is only one command of a certain command type
    clean_commands = ensure_max_number_of_command_type(
        clean_commands, CannotHandleCommand, 1
    )
    clean_commands = ensure_max_number_of_command_type(
        clean_commands, RepeatBotMessagesCommand, 1
    )
    clean_commands = ensure_max_number_of_command_type(
        clean_commands, ChitChatAnswerCommand, 1
    )

    # Replace CannotHandleCommands with ContinueAgentCommand when an agent is active
    # to keep the agent running, but preserve chitchat
    clean_commands = _replace_cannot_handle_with_continue_agent(clean_commands, tracker)

    # filter out cannot handle commands if there are other commands present
    # when coexistence is enabled, by default there will be a SetSlotCommand
    # for the ROUTE_TO_CALM_SLOT slot.
    if tracker.has_coexistence_routing_slot and len(clean_commands) > 2:
        clean_commands = filter_cannot_handle_command(clean_commands)
    elif not tracker.has_coexistence_routing_slot and len(clean_commands) > 1:
        clean_commands = filter_cannot_handle_command(clean_commands)

    structlogger.debug(
        "command_processor.clean_up_commands.final_commands",
        command=clean_commands,
        event_info="Final commands",
        highlight=True,
    )

    return clean_commands


def _process_multiple_start_flow_commands(
    commands: List[Command],
    tracker: DialogueStateTracker,
) -> List[Command]:
    """Process multiple start flow commands.

    If there are multiple start flow commands, no active flows and the
    CLARIFY_ON_MULTIPLE_START_FLOWS env var is enabled, we replace the
    start flow commands with a clarify command.
    """
    start_flow_candidates = filter_start_flow_commands(commands)
    clarify_enabled = (
        os.getenv("CLARIFY_ON_MULTIPLE_START_FLOWS", "false").lower() == "true"
    )

    if clarify_enabled and len(start_flow_candidates) > 1 and tracker.stack.is_empty():
        # replace the start flow commands with a clarify command
        commands = [
            command for command in commands if not isinstance(command, StartFlowCommand)
        ]
        # avoid adding duplicate clarify commands
        if not any(isinstance(c, ClarifyCommand) for c in commands):
            structlogger.debug(
                "command_processor.clean_up_commands.trigger_clarify_for_multiple_start_flows",
                candidate_flows=start_flow_candidates,
            )
            commands.append(ClarifyCommand(options=start_flow_candidates))

    return commands


def _get_slots_eligible_for_correction(tracker: DialogueStateTracker) -> Set[str]:
    """Get all slots that are eligible for correction.

    # We consider all slots, which are not None, that were set in the tracker
    # eligible for correction.
    # In the correct_slot_command we will check if a slot should actually be
    # corrected.
    """
    # get all slots that were set in the tracker
    slots_so_far = set(
        [event.key for event in tracker.events if isinstance(event, SlotSet)]
    )

    # filter out slots that are set to None (None = empty value)
    slots_so_far = {slot for slot in slots_so_far if tracker.get_slot(slot) is not None}

    return slots_so_far


def ensure_max_number_of_command_type(
    commands: List[Command], command_type: Type[Command], n: int
) -> List[Command]:
    """Ensures that for a given command type only the first n stay in the list."""
    filtered: List[Command] = []
    count = 0
    for c in commands:
        if isinstance(c, command_type):
            if count >= n:
                continue
            else:
                count += 1
        filtered.append(c)
    return filtered


def clean_up_start_flow_command(
    clean_commands: List[Command],
    tracker: DialogueStateTracker,
    command: StartFlowCommand,
) -> List[Command]:
    """Clean up a start flow command."""
    from rasa.dialogue_understanding.patterns.continue_interrupted import (
        ContinueInterruptedPatternFlowStackFrame,
    )

    continue_interrupted_flow_active = is_pattern_active(
        tracker.stack, ContinueInterruptedPatternFlowStackFrame
    )

    top_user_frame = top_user_flow_frame(
        tracker.stack, ignore_call_and_link_frames=False
    )
    top_flow_id = top_user_frame.flow_id if top_user_frame else ""

    if top_flow_id == command.flow and not continue_interrupted_flow_active:
        # drop a start flow command if the starting flow is equal
        # to the currently active flow
        structlogger.debug(
            "command_processor.clean_up_commands.skip_command_flow_already_active",
            command=command,
        )
        return clean_commands

    clean_commands.append(command)
    return clean_commands


def clean_up_cancel_flow_command(
    clean_commands: List[Command],
    tracker: DialogueStateTracker,
    command: CancelFlowCommand,
) -> List[Command]:
    """Clean up a cancel flow command."""
    # If there's no active flow, replace CancelFlowCommand with CannotHandleCommand
    if tracker.active_flow is None:
        structlogger.debug(
            "command_processor.clean_up_commands"
            ".replace_cancel_flow_with_cannot_handle_no_active_flow",
            command=command,
        )
        if not contains_command(clean_commands, CannotHandleCommand):
            clean_commands.append(
                CannotHandleCommand(
                    reason=("CancelFlowCommand was predicted but no flows are active.")
                )
            )
    elif contains_command(clean_commands, CancelFlowCommand):
        # Skip duplicate CancelFlowCommand
        structlogger.debug(
            "command_processor.clean_up_commands"
            ".skip_command_flow_already_cancelled",
            command=command,
        )
    else:
        # Otherwise add the CancelFlowCommand
        clean_commands.append(command)

    return clean_commands


def _replace_cannot_handle_with_continue_agent(
    clean_commands: List[Command],
    tracker: DialogueStateTracker,
) -> List[Command]:
    """Replace CannotHandleCommands with ContinueAgentCommand when agent is active.

    ContinueAgentCommand is added in the following cases:

    1. LLM Command Generation Failures:
       - LLM parsing failures (default reason)
       - Force slot filling failures (default reason)

    2. Invalid Commands During Cleanup:
       - Invalid SetSlot commands:
         - Slot not in domain
         - Incompatible extractor
       (Note: ChitChatAnswer command failures are preserved as CannotHandleCommand)

    3. Empty Commands List:
       - When all commands are filtered out during cleanup

    Preserved as CannotHandleCommand (not replaced):
    - Chitchat: CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_CHITCHAT)
    """
    if not tracker.stack.agent_is_active():
        return clean_commands

    # If no commands at all and agent is active, add ContinueAgentCommand
    if not clean_commands:
        clean_commands.append(ContinueAgentCommand())
        return clean_commands

    has_continue_agent = any(
        isinstance(cmd, ContinueAgentCommand) for cmd in clean_commands
    )

    # Collect CannotHandleCommands that should be replaced with ContinueAgentCommand
    cannot_handle_commands = [
        cmd
        for cmd in clean_commands
        if isinstance(cmd, CannotHandleCommand)
        and cmd.reason != RASA_PATTERN_CANNOT_HANDLE_CHITCHAT
    ]

    if cannot_handle_commands:
        structlogger.debug(
            "command_processor.clean_up_commands"
            ".replace_cannot_handle_with_continue_agent",
            original_commands=clean_commands,
        )
        # Remove the CannotHandleCommands we collected
        for cmd in cannot_handle_commands:
            clean_commands.remove(cmd)

        # Add ContinueAgentCommand if not already present
        if not has_continue_agent:
            clean_commands.append(ContinueAgentCommand())

    return clean_commands


def clean_up_clarify_command(
    commands_so_far: List[Command],
    all_commands: List[Command],
    current_command: ClarifyCommand,
) -> List[Command]:
    """Clean up a clarify command.

    Args:
        commands_so_far: The commands cleaned up so far.
        all_commands: All the predicted commands.
        current_command: The current clarify command.

    Returns:
        The cleaned up commands.
    """
    # Get the commands after removing the ROUTE_TO_CALM_SLOT set slot command.
    commands_without_route_to_calm_set_slot = [
        c
        for c in all_commands
        if not (isinstance(c, SetSlotCommand) and c.name == ROUTE_TO_CALM_SLOT)
    ]

    # if all commands are clarify commands, add the first one only, otherwise add none
    if all(
        isinstance(c, ClarifyCommand) for c in commands_without_route_to_calm_set_slot
    ):
        # Check if clean_commands is empty or contains only ROUTE_TO_CALM_SLOT
        # set slot command.
        if not commands_so_far or (
            len(commands_so_far) == 1
            and isinstance(commands_so_far[0], SetSlotCommand)
            and commands_so_far[0].name == ROUTE_TO_CALM_SLOT
        ):
            commands_so_far.append(current_command)

    return commands_so_far


def clean_up_slot_command(
    commands_so_far: List[Command],
    command: SetSlotCommand,
    tracker: DialogueStateTracker,
    all_flows: FlowsList,
) -> List[Command]:
    """Clean up a slot command.

    This will remove commands that are not necessary anymore, e.g. because the slot
    they correct was previously corrected to the same value. It will group all slot
    corrections into one command.

    Args:
        commands_so_far: The commands cleaned up so far.
        command: The command to clean up.
        tracker: The dialogue state tracker.
        all_flows: All flows.

    Returns:
        The cleaned up commands.
    """
    stack = tracker.stack
    resulting_commands = commands_so_far[:]
    slot = tracker.slots.get(command.name)

    # if the slot is not in the domain, we cannot set it
    if slot is None:
        structlogger.debug(
            "command_processor.clean_up_slot_command.skip_command_slot_not_in_domain",
            command=command,
        )
        resulting_commands.append(
            CannotHandleCommand(
                reason="The slot predicted by the LLM is not defined in the domain."
            )
        )
        return resulting_commands

    # check if the slot should be set by the command
    if not should_slot_be_set(slot, command, resulting_commands):
        structlogger.debug(
            "command_processor.clean_up_slot_command.skip_command.extractor_"
            "does_not_match_slot_mapping",
            extractor=command.extractor,
            slot_name=slot.name,
        )

        # prevent adding a cannot handle command in case commands_so_far already
        # contains a valid prior set slot command for the same slot whose current
        # slot command was rejected by should_slot_be_set
        slot_command_exists_already = any(
            isinstance(command, SetSlotCommand) and command.name == slot.name
            for command in resulting_commands
        )

        cannot_handle = CannotHandleCommand(
            reason="A command generator attempted to set a slot with a value extracted "
            "by an extractor that is incompatible with the slot mapping type."
        )
        if not slot_command_exists_already and cannot_handle not in resulting_commands:
            resulting_commands.append(cannot_handle)

        return resulting_commands

    # check if the slot can be corrected by the LLM
    if (
        slot.filled_by == SetSlotExtractor.NLU.value
        and command.extractor == SetSlotExtractor.LLM.value
    ):
        allow_nlu_correction = any(
            [
                mapping.allow_nlu_correction is True
                for mapping in slot.mappings
                if mapping.type == SlotMappingType.FROM_LLM
            ]
        )

        if not allow_nlu_correction:
            structlogger.debug(
                "command_processor.clean_up_slot_command"
                ".skip_command.disallow_llm_correction_of_nlu_set_value",
                command=command,
            )
            return resulting_commands

    # get all slots that were set in the tracker and are eligible for correction
    slots_eligible_for_correction = _get_slots_eligible_for_correction(tracker)

    if (
        command.name in slots_eligible_for_correction
        and command.name != ROUTE_TO_CALM_SLOT
    ):
        current_collect_info = get_current_collect_step(stack, all_flows)

        if current_collect_info and current_collect_info.collect == command.name:
            # not a correction but rather an answer to the current collect info
            resulting_commands.append(command)
            return resulting_commands

        if should_slot_be_corrected(command, tracker, stack, all_flows):
            # if the slot was already set before, we need to convert it into
            # a correction
            return convert_set_slot_to_correction(command, resulting_commands)
        else:
            return resulting_commands

    resulting_commands.append(command)
    return resulting_commands


def should_slot_be_corrected(
    command: SetSlotCommand,
    tracker: DialogueStateTracker,
    stack: DialogueStack,
    all_flows: FlowsList,
) -> bool:
    """Check if a slot should be corrected."""
    if (slot := tracker.slots.get(command.name)) is not None and str(slot.value) == str(
        command.value
    ):
        # the slot is already set to the same value, we don't need to set it again
        structlogger.debug(
            "command_processor.clean_up_slot_command.skip_command_slot_already_set",
            command=command,
        )
        return False

    top = top_flow_frame(stack)
    if isinstance(top, CorrectionPatternFlowStackFrame):
        already_corrected_slots = top.corrected_slots
    else:
        already_corrected_slots = {}

    if command.name in already_corrected_slots and str(
        already_corrected_slots[command.name]
    ) == str(command.value):
        structlogger.debug(
            "command_processor.clean_up_slot_command"
            ".skip_command_slot_already_corrected",
            command=command,
        )
        return False

    return True


def convert_set_slot_to_correction(
    command: SetSlotCommand,
    resulting_commands: List[Command],
) -> List[Command]:
    """Convert a set slot command to a correction command."""
    structlogger.debug(
        "command_processor.convert_set_slot_to_correction",
        command=command,
    )

    # Group all corrections into one command
    corrected_slot = CorrectedSlot(command.name, command.value, command.extractor)
    for c in resulting_commands:
        if isinstance(c, CorrectSlotsCommand):
            c.corrected_slots.append(corrected_slot)
            break
    else:
        resulting_commands.append(CorrectSlotsCommand(corrected_slots=[corrected_slot]))

    return resulting_commands


def clean_up_chitchat_command(
    commands_so_far: List[Command],
    command: ChitChatAnswerCommand,
    flows: FlowsList,
    execution_context: ExecutionContext,
    domain: Domain,
    story_graph: Optional[StoryGraph] = None,
) -> List[Command]:
    """Clean up a chitchat answer command.

    Respond with 'cannot handle' if 'IntentlessPolicy' is unset in
    model config but 'action_trigger_chitchat' is used within the pattern_chitchat

    Args:
        commands_so_far: The commands cleaned up so far.
        command: The command to clean up.
        flows: All flows.
        execution_context: Information about a single graph run.
        story_graph: StoryGraph object with stories available for training.
        domain: The domain of the bot.

    Returns:
        The cleaned up commands.
    """
    from rasa.core.policies.intentless_policy import IntentlessPolicy

    resulting_commands = commands_so_far[:]

    pattern_chitchat = flows.flow_by_id(FLOW_PATTERN_CHITCHAT)

    # very unlikely to happen, placed here due to mypy checks
    if pattern_chitchat is None:
        resulting_commands.insert(
            0, CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_CHITCHAT)
        )
        structlogger.warn(
            "command_processor.clean_up_chitchat_command.pattern_chitchat_not_found",
            command=resulting_commands[0],  # no PII
        )
        return resulting_commands

    has_action_trigger_chitchat = pattern_chitchat.has_action_step(
        ACTION_TRIGGER_CHITCHAT
    )
    defines_intentless_policy = execution_context.has_node(IntentlessPolicy)

    if (has_action_trigger_chitchat and not defines_intentless_policy) or (
        defines_intentless_policy
        and not contains_intentless_policy_responses(flows, domain, story_graph)
    ):
        resulting_commands.insert(
            0, CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_CHITCHAT)
        )
        structlogger.warn(
            "command_processor.clean_up_chitchat_command."
            "replace_chitchat_answer_with_cannot_handle",
            command=resulting_commands[0],  # no PII
            pattern_chitchat_uses_action_trigger_chitchat=has_action_trigger_chitchat,
            defined_intentless_policy_in_config=defines_intentless_policy,
        )
    else:
        resulting_commands.insert(0, command)
        structlogger.debug(
            "command_processor.clean_up_commands.prepend_command_chitchat_answer",
            command=command,
            pattern_chitchat_uses_action_trigger_chitchat=has_action_trigger_chitchat,
            defined_intentless_policy_in_config=defines_intentless_policy,
        )

    return resulting_commands


def should_slot_be_set(
    slot: Slot, command: SetSlotCommand, commands_so_far: Optional[List[Command]] = None
) -> bool:
    """Check if a slot should be set by a command."""
    if command.extractor == SetSlotExtractor.COMMAND_PAYLOAD_READER.value:
        # if the command is issued by the command payload reader, it means the slot
        # was set deterministically via a response button. In this case,
        # we can always set it
        return True

    if commands_so_far is None:
        commands_so_far = []

    set_slot_commands_so_far = [
        command
        for command in commands_so_far
        if isinstance(command, SetSlotCommand) and command.name == slot.name
    ]

    slot_mappings = slot.mappings

    if not slot.mappings:
        slot_mappings = [SlotMapping(type=SlotMappingType.FROM_LLM)]

    mapping_types = [mapping.type for mapping in slot_mappings]

    slot_has_nlu_mapping = any(
        [mapping_type.is_predefined_type() for mapping_type in mapping_types]
    )
    slot_has_llm_mapping = any(
        [mapping_type == SlotMappingType.FROM_LLM for mapping_type in mapping_types]
    )
    slot_has_controlled_mapping = any(
        [mapping_type == SlotMappingType.CONTROLLED for mapping_type in mapping_types]
    )

    if set_slot_commands_so_far and command.extractor == SetSlotExtractor.LLM.value:
        # covers the following scenarios:
        # scenario 1: NLU mapping extracts a value for slot_a → If LLM extracts a value for slot_a, it is discarded.  # noqa: E501
        # scenario 2: NLU mapping is unable to extract a value for slot_a → If LLM extracts a value for slot_a, it is accepted.  # noqa: E501
        command_has_nlu_extractor = any(
            [
                command.extractor == SetSlotExtractor.NLU.value
                for command in set_slot_commands_so_far
            ]
        )
        return not command_has_nlu_extractor and slot_has_llm_mapping

    if (
        slot_has_nlu_mapping
        and command.extractor == SetSlotExtractor.LLM.value
        and not slot_has_llm_mapping
    ):
        return False

    if (
        slot_has_llm_mapping
        and command.extractor == SetSlotExtractor.NLU.value
        and not slot_has_nlu_mapping
    ):
        return False

    if slot_has_controlled_mapping and not (
        slot_has_nlu_mapping or slot_has_llm_mapping
    ):
        return False

    return True


def filter_cannot_handle_command(
    clean_commands: List[Command],
) -> List[Command]:
    """Filter out a 'cannot handle' command.

    This is used to filter out a 'cannot handle' command
    in case other commands are present.

    Returns:
        The filtered commands.
    """
    return [
        command
        for command in clean_commands
        if not isinstance(command, CannotHandleCommand)
    ]


def reorder_commands(
    commands: List[Command], tracker: DialogueStateTracker
) -> List[Command]:
    """Reorder commands.

    In case there is no active flow, we want to make sure to run the start flow
    commands first.
    """
    reordered_commands = commands

    top_flow_frame = top_user_flow_frame(tracker.stack)

    if top_flow_frame is None:
        # no active flow, we want to make sure to run the start flow commands first
        start_flow_commands: List[Command] = [
            command for command in commands if isinstance(command, StartFlowCommand)
        ]

        # if there are no start flow commands, we can return the commands as they are
        if not start_flow_commands:
            reordered_commands = commands

        # if there is just one start flow command, we want to run it first
        # as the order of commands is reserved later,
        # we need to add it to the end of the list
        elif len(start_flow_commands) == 1:
            reordered_commands = [
                command for command in commands if command not in start_flow_commands
            ] + start_flow_commands

        # if there are multiple start flow commands,
        # we just make sure to move the first start flow command to the end of the list
        # (due to the reverse execution order of commands) and keep the other commands
        # as they are.
        else:
            reordered_commands = [
                command for command in commands if command != start_flow_commands[-1]
            ] + [start_flow_commands[-1]]

    # commands need to be reversed to make sure they end up in the right order
    # on the stack. e.g. if there multiple start flow commands, the first one
    # should be on top of the stack. this is achieved by reversing the list
    # and then pushing the commands onto the stack in the reversed order.
    return list(reversed(reordered_commands))
