from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple, cast

import structlog

from rasa.agents.agent_manager import AgentManager
from rasa.agents.constants import (
    A2A_AGENT_CONTEXT_ID_KEY,
    AGENT_METADATA_AGENT_ID_KEY,
    AGENT_METADATA_AGENT_RESPONSE_KEY,
    AGENT_METADATA_EXIT_IF_KEY,
    AGENT_METADATA_MODEL_ID_KEY,
    AGENT_METADATA_SENDER_ID_KEY,
    AGENT_METADATA_STRUCTURED_RESULTS_KEY,
    MAX_AGENT_RETRY_DELAY_SECONDS,
)
from rasa.agents.core.types import AgentStatus, ProtocolType
from rasa.agents.schemas import AgentInput, AgentOutput
from rasa.agents.schemas.agent_input import AgentInputSlot
from rasa.core.channels.channel import OutputChannel
from rasa.core.config.configuration import Configuration
from rasa.core.policies.flows.flow_step_result import (
    ContinueFlowWithNextStep,
    FlowActionPrediction,
    FlowStepResult,
    PauseFlowReturnPrediction,
)
from rasa.core.utils import get_slot_names_from_exit_conditions
from rasa.dialogue_understanding.patterns.cancel import CancelPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
    BaseFlowStackFrame,
)
from rasa.shared.agents.utils import get_protocol_type
from rasa.shared.core.constants import (
    ACTION_AGENT_REQUEST_USER_INPUT_NAME,
    ACTION_METADATA_MESSAGE_KEY,
    ACTION_METADATA_TEXT_KEY,
    ACTION_SEND_TEXT_NAME,
    SLOTS_EXCLUDED_FOR_AGENT,
)
from rasa.shared.core.events import (
    AgentCancelled,
    AgentCompleted,
    AgentResumed,
    AgentStarted,
    Event,
    FlowCancelled,
    SlotSet,
    deserialise_events,
)
from rasa.shared.core.flows.flows_list import FlowsList
from rasa.shared.core.flows.steps import (
    CallFlowStep,
)
from rasa.shared.core.slots import CategoricalSlot, Slot
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.utils.llm import tracker_as_readable_transcript

structlogger = structlog.get_logger()

MAX_AGENT_RETRIES = 3


def remove_agent_stack_frame(stack: DialogueStack, agent_id: str) -> None:
    """Finishes the agentic loop by popping the agent stack frame from provided `stack`.

    The `tracker.stack` is NOT modified.
    """
    agent_stack_frame = stack.find_agent_stack_frame_by_agent(agent_id)
    if not agent_stack_frame:
        return

    while removed_frame := stack.pop():
        structlogger.debug(
            "flow_executor.remove_agent_stack_frame",
            removed_frame=removed_frame,
        )
        if removed_frame == agent_stack_frame:
            break


async def run_agent(
    initial_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
    tracker: DialogueStateTracker,
    slots: List[Slot],
    flows: FlowsList,
    output_channel: Optional[OutputChannel] = None,
) -> FlowStepResult:
    """Run an agent call step."""
    structlogger.debug(
        "flow.step.run_agent",
        agent_id=step.call,
        step_id=step.id,
        flow_id=step.flow_id,
        event_info=f"Agent {step.call} started",
        highlight=True,
    )

    final_events = initial_events
    agent_stack_frame = tracker.stack.find_agent_stack_frame_by_agent(
        agent_id=step.call
    )

    if (
        agent_stack_frame
        and agent_stack_frame == stack.top()
        and agent_stack_frame.state == AgentState.INTERRUPTED
    ):
        # if an agent is interrupted, repeat the last message from the
        # agent and wait for user input
        return _handle_resume_interrupted_agent(
            agent_stack_frame, final_events, stack, step, tracker
        )

    # Reset the slots covered by the exit_if
    # Code smell: this is a temporary fix and will be addressed in ENG-2148
    if (
        step.exit_if
        and agent_stack_frame
        and agent_stack_frame.frame_id == f"restart_agent_{step.call}"
    ):
        # when restarting an agent, we need to reset the slots covered by the
        # exit_if condition so that the agent can run again.
        _reset_slots_covered_by_exit_if(step.exit_if, tracker)

    # generate the agent input
    agent_input = _prepare_agent_input(agent_stack_frame, step, tracker, slots)
    structlogger.debug(
        "flow.step.run_agent.agent_input",
        agent_name=step.call,
        step_id=step.id,
        flow_id=step.flow_id,
        agent_input=agent_input.model_dump(),
        json_formatting=["agent_input"],
    )

    # add the AgentStarted event to the list of final events
    final_events.append(AgentStarted(step.call, step.flow_id))

    # send the input to the agent and wait for a response
    protocol_type = get_protocol_type(
        step, Configuration.get_instance().available_agents.get_agent_config(step.call)
    )
    output: AgentOutput = await _call_agent_with_retry(
        agent_name=step.call,
        protocol_type=protocol_type,
        agent_input=agent_input,
        max_retries=MAX_AGENT_RETRIES,
        output_channel=output_channel,
    )

    structlogger.debug(
        "flow.step.run_agent.agent_response",
        agent_name=step.call,
        step_id=step.id,
        flow_id=step.flow_id,
        agent_response=output.model_dump(),
        json_formatting=["agent_response"],
        event_info="Agent Output",
    )
    structlogger.debug(
        "flow.step.run_agent.agent_finished",
        agent_name=step.call,
        flow_id=step.flow_id,
        agent_state=str(output.status),
        agent_response_message=output.response_message,
        highlight=True,
        event_info=f"Agent {step.call} finished",
    )

    # add the set slot events returned by the agent to the list of final events
    if output.events:
        final_events.extend(output.events)

    # handle the agent output based on the agent status
    if output.status == AgentStatus.INPUT_REQUIRED:
        return _handle_agent_input_required(output, final_events, stack, step)
    elif output.status == AgentStatus.COMPLETED:
        return _handle_agent_completed(output, final_events, stack, step)
    elif output.status == AgentStatus.FATAL_ERROR:
        return _handle_agent_fatal_error(
            output, final_events, stack, step, flows, tracker
        )
    else:
        return _handle_agent_unknown_status(
            output, final_events, stack, step, flows, tracker
        )


async def _call_agent_with_retry(
    agent_name: str,
    protocol_type: ProtocolType,
    agent_input: AgentInput,
    max_retries: int,
    output_channel: Optional[OutputChannel] = None,
) -> AgentOutput:
    """Call an agent with retries in case of recoverable errors."""
    for attempt in range(max_retries):
        if attempt > 0:
            structlogger.debug(
                "flow_executor.call_agent_with_retry.retrying",
                agent_name=agent_name,
                attempt=attempt + 1,
                num_retries=max_retries,
            )
        try:
            agent_response: AgentOutput = await AgentManager().run_agent(
                agent_name=agent_name,
                protocol_type=protocol_type,
                context=agent_input,
                output_channel=output_channel,
            )
        except Exception as e:
            # We don't have a vaild agent response at this time to act based
            # on the agent status, so we return a fatal error.
            structlogger.error(
                "flow_executor.call_agent_with_retry.exception",
                agent_name=agent_name,
                error_message=str(e),
            )
            return AgentOutput(
                id=agent_name,
                status=AgentStatus.FATAL_ERROR,
                error_message=str(e),
            )

        if agent_response.status != AgentStatus.RECOVERABLE_ERROR:
            return agent_response

        structlogger.warning(
            "flow_executor.call_agent_with_retry.recoverable_error",
            agent_name=agent_name,
            attempt=attempt + 1,
            num_retries=max_retries,
            error_message=agent_response.error_message,
        )
        if attempt < max_retries - 1:
            # exponential backoff - wait longer with each retry
            # 1 second, 2 seconds, 4 seconds, etc.
            await asyncio.sleep(min(2**attempt, MAX_AGENT_RETRY_DELAY_SECONDS))

    # we exhausted all retries, return fatal error
    structlogger.warning(
        "flow_executor.call_agent_with_retry.exhausted_retries",
        agent_name=agent_name,
        num_retries=max_retries,
    )
    return AgentOutput(
        id=agent_name,
        status=AgentStatus.FATAL_ERROR,
        error_message="Exhausted all retries for agent call.",
    )


################################################################################
# Handle resume interrupted agent
################################################################################


def _handle_resume_interrupted_agent(
    agent_stack_frame: AgentStackFrame,
    final_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
    tracker: DialogueStateTracker,
) -> FlowStepResult:
    """Handle resuming an interrupted agent.

    Args:
        agent_stack_frame: The interrupted agent stack frame
        final_events: List of events to be added to the final result
        stack: The dialogue stack
        step: The flow step that called the agent
        tracker: The dialogue state tracker

    Returns:
        FlowStepResult indicating to pause for user input
    """
    structlogger.debug(
        "flow.step.run_agent.resume_interrupted_agent",
        agent_id=step.call,
        step_id=step.id,
        flow_id=step.flow_id,
    )
    # The agent was previously interrupted when waiting for user input.
    # Now we're back to the agent execution step and need to output the last message
    # from the agent (user input request) again and wait for user input
    cast(AgentStackFrame, stack.top()).state = AgentState.WAITING_FOR_INPUT
    tracker.update_stack(stack)
    utterance = (
        agent_stack_frame.metadata.get(AGENT_METADATA_AGENT_RESPONSE_KEY, "")
        if agent_stack_frame.metadata
        else ""
    )
    final_events.append(AgentResumed(agent_id=step.call, flow_id=step.flow_id))
    return PauseFlowReturnPrediction(
        _create_agent_request_user_input_prediction(utterance, final_events)
    )


################################################################################
# Handle agent output
################################################################################


def _handle_agent_unknown_status(
    output: AgentOutput,
    final_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
    flows: FlowsList,
    tracker: DialogueStateTracker,
) -> FlowStepResult:
    """Handle unknown agent status.

    Args:
        output: The agent output with unknown status
        final_events: List of events to be added to the final result
        stack: The dialogue stack
        step: The flow step that called the agent
        flows: All flows
        tracker: The dialogue state tracker

    Returns:
        FlowStepResult indicating to continue with internal error pattern
    """
    output.metadata = output.metadata or {}
    _update_agent_events(final_events, output.metadata)
    structlogger.error(
        "flow.step.run_agent.unknown_status",
        agent_name=step.call,
        step_id=step.id,
        flow_id=step.flow_id,
        status=output.status,
    )
    # remove the agent stack frame
    remove_agent_stack_frame(stack, step.call)
    final_events.append(AgentCancelled(agent_id=step.call, flow_id=step.flow_id))

    # cancel the current active flow:
    # push the cancel pattern stack frame and add the flow cancelled event
    cancel_pattern_stack_frame, flow_cancelled_event = _cancel_flow(
        stack, flows, tracker, step
    )
    if cancel_pattern_stack_frame:
        stack.push(cancel_pattern_stack_frame)
    if flow_cancelled_event:
        final_events.append(flow_cancelled_event)

    # trigger the internal error pattern
    stack.push(InternalErrorPatternFlowStackFrame())
    return ContinueFlowWithNextStep(events=final_events)


def _handle_agent_input_required(
    output: AgentOutput,
    final_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
) -> FlowStepResult:
    """Handle agent that requires user input.

    Args:
        output: The agent output containing input request information
        final_events: List of events to be added to the final result
        stack: The dialogue stack
        step: The flow step that called the agent

    Returns:
        FlowStepResult indicating to pause for user input
    """
    output.metadata = output.metadata or {}
    output.metadata[AGENT_METADATA_AGENT_RESPONSE_KEY] = output.response_message or ""
    output.metadata[AGENT_METADATA_STRUCTURED_RESULTS_KEY] = (
        output.structured_results or []
    )
    _update_agent_events(final_events, output.metadata)

    top_stack_frame = stack.top()
    # update the agent stack frame if it is already on the stack
    # otherwise push a new one
    if isinstance(top_stack_frame, AgentStackFrame):
        top_stack_frame.state = AgentState.WAITING_FOR_INPUT
        top_stack_frame.metadata = output.metadata
        top_stack_frame.step_id = step.id
        top_stack_frame.agent_id = step.call
        top_stack_frame.flow_id = step.flow_id
    else:
        stack.push(
            AgentStackFrame(
                flow_id=step.flow_id,
                agent_id=step.call,
                state=AgentState.WAITING_FOR_INPUT,
                step_id=step.id,
                metadata=output.metadata,
            )
        )

    action_prediction = _create_agent_request_user_input_prediction(
        output.response_message, final_events
    )
    return PauseFlowReturnPrediction(action_prediction)


def _handle_agent_completed(
    output: AgentOutput,
    final_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
) -> FlowStepResult:
    """Handle completed agent execution.

    Args:
        output: The agent output containing completion information
        final_events: List of events to be added to the final result
        stack: The dialogue stack
        step: The flow step that called the agent

    Returns:
        FlowStepResult indicating to continue with next step or pause for response
    """
    output.metadata = output.metadata or {}
    _update_agent_events(final_events, output.metadata)
    structlogger.debug(
        "flow.step.run_agent.completed",
        agent_name=step.call,
        step_id=step.id,
        flow_id=step.flow_id,
    )
    remove_agent_stack_frame(stack, step.call)
    agent_completed_event = AgentCompleted(agent_id=step.call, flow_id=step.flow_id)
    final_events.append(agent_completed_event)
    if output.response_message:
        # for open-ended agents we want to utter the last agent message
        return PauseFlowReturnPrediction(
            _create_send_text_prediction(output.response_message, final_events)
        )
    else:
        return ContinueFlowWithNextStep(events=final_events)


def _handle_agent_fatal_error(
    output: AgentOutput,
    final_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
    flows: FlowsList,
    tracker: DialogueStateTracker,
) -> FlowStepResult:
    """Handle fatal error from agent execution.

    Args:
        output: The agent output containing error information
        final_events: List of events to be added to the final result
        stack: The dialogue stack
        step: The flow step that called the agent
        flows: All flows
        tracker: The dialogue state tracker

    Returns:
        FlowStepResult indicating to continue with internal error pattern
    """
    output.metadata = output.metadata or {}
    _update_agent_events(final_events, output.metadata)
    # the agent failed, cancel the current flow and trigger pattern_internal_error
    structlogger.error(
        "flow.step.run_agent.fatal_error",
        agent_name=step.call,
        step_id=step.id,
        flow_id=step.flow_id,
        error_message=output.error_message,
    )
    # remove the agent stack frame
    remove_agent_stack_frame(stack, step.call)
    final_events.append(
        AgentCancelled(
            agent_id=step.call, flow_id=step.flow_id, reason=output.error_message
        )
    )

    # cancel the current active flow:
    # push the cancel pattern stack frame and add the flow cancelled event
    cancel_pattern_stack_frame, flow_cancelled_event = _cancel_flow(
        stack, flows, tracker, step
    )
    if cancel_pattern_stack_frame:
        stack.push(cancel_pattern_stack_frame)
    if flow_cancelled_event:
        final_events.append(flow_cancelled_event)

    # push the internal error pattern stack frame
    stack.push(InternalErrorPatternFlowStackFrame())
    return ContinueFlowWithNextStep(events=final_events)


def _cancel_flow(
    stack: DialogueStack,
    flows: FlowsList,
    tracker: DialogueStateTracker,
    step: CallFlowStep,
) -> Tuple[Optional[CancelPatternFlowStackFrame], Optional[FlowCancelled]]:
    """Cancel the current active flow.

    Creates a cancel pattern stack frame and a flow cancelled event.
    """
    from rasa.dialogue_understanding.commands import CancelFlowCommand

    cancel_pattern_stack_frame = None
    flow_cancelled_event = None

    top_frame = stack.top()

    if isinstance(top_frame, BaseFlowStackFrame):
        flow = flows.flow_by_id(step.flow_id)
        flow_name = (
            flow.readable_name(language=tracker.current_language)
            if flow
            else step.flow_id
        )

        canceled_frames = CancelFlowCommand.select_canceled_frames(stack)

        cancel_pattern_stack_frame = CancelPatternFlowStackFrame(
            canceled_name=flow_name,
            canceled_frames=canceled_frames,
        )

        flow_cancelled_event = FlowCancelled(step.flow_id, step.id)

    return cancel_pattern_stack_frame, flow_cancelled_event


################################################################################
# Create predictions
################################################################################


def _create_action_prediction(
    action_name: str, message: Optional[str], events: Optional[List[Event]]
) -> FlowActionPrediction:
    """Create a prediction for an action with a text message."""
    action_metadata = {
        ACTION_METADATA_MESSAGE_KEY: {
            ACTION_METADATA_TEXT_KEY: message,
        }
    }
    return FlowActionPrediction(
        action_name,
        1.0,
        events=events if events else [],
        metadata=action_metadata,
    )


def _create_agent_request_user_input_prediction(
    message: Optional[str], events: Optional[List[Event]]
) -> FlowActionPrediction:
    """Create a prediction for requesting user input from the agent and waiting for it.

    This function creates a prediction that will pause the flow and wait for user input.
    """
    return _create_action_prediction(
        ACTION_AGENT_REQUEST_USER_INPUT_NAME, message, events
    )


def _create_send_text_prediction(
    message: Optional[str], events: Optional[List[Event]]
) -> FlowActionPrediction:
    """Create a prediction for sending a text message to the user."""
    return _create_action_prediction(ACTION_SEND_TEXT_NAME, message, events)


################################################################################
# Prepare agent input
################################################################################


def _prepare_agent_input(
    agent_stack_frame: Optional[AgentStackFrame],
    step: CallFlowStep,
    tracker: DialogueStateTracker,
    slots: List[Slot],
) -> AgentInput:
    """Prepare the agent input data.

    Args:
        agent_stack_frame: The agent stack frame if it exists
        step: The flow step that called the agent
        tracker: The dialogue state tracker
        slots: List of slot definitions

    Returns:
        AgentInput object ready for agent execution
    """
    agent_input_metadata = (
        agent_stack_frame.metadata
        if agent_stack_frame and agent_stack_frame.metadata
        else {}
    )
    _update_agent_input_metadata_with_events(
        agent_input_metadata, step.call, step.flow_id, tracker
    )

    if step.exit_if:
        agent_input_metadata[AGENT_METADATA_EXIT_IF_KEY] = step.exit_if

    agent_input_metadata[AGENT_METADATA_SENDER_ID_KEY] = tracker.sender_id
    agent_input_metadata[AGENT_METADATA_AGENT_ID_KEY] = tracker.assistant_id
    agent_input_metadata[AGENT_METADATA_MODEL_ID_KEY] = tracker.model_id

    return AgentInput(
        id=step.call,
        user_message=tracker.latest_message.text or ""
        if tracker.latest_message
        else "",
        slots=_prepare_slots_for_agent(
            tracker.current_slot_values(), slots, step.exit_if
        ),
        conversation_history=tracker_as_readable_transcript(tracker),
        events=deserialise_events(
            tracker.current_state(EventVerbosity.ALL).get("events") or []
        ),
        metadata=agent_input_metadata,
        recipient_id=tracker.sender_id,
    )


def _prepare_slots_for_agent(
    slot_values: Dict[str, Any],
    slot_definitions: List[Slot],
    exit_if: Optional[List[str]],
) -> List[AgentInputSlot]:
    """Prepare the slots for the agent.

    Filter out slots that should not be forwarded to agents.
    Add the slot type and allowed values to the slot dictionary.

    Filter out slots that are None.
    Keep slots that are part of the exit_if conditions.

    Args:
        slot_values: The full slot dictionary from the tracker.
        slot_definitions: The slot definitions from the domain.
        exit_if: Optional list of exit conditions that determine which slots to keep.

    Returns:
        A list of slots containing the name, current value, type, and allowed values.
    """

    def _get_slot_definition(slot_name: str) -> Optional[Slot]:
        for slot in slot_definitions:
            if slot.name == slot_name:
                return slot
        return None

    exit_if_slot_names = []
    if exit_if:
        exit_if_slot_names = get_slot_names_from_exit_conditions(exit_if)

    filtered_slots: List[AgentInputSlot] = []
    for key, value in slot_values.items():
        if key in SLOTS_EXCLUDED_FOR_AGENT:
            continue
        if value is None and key not in exit_if_slot_names:
            continue
        slot_definition = _get_slot_definition(key)
        if slot_definition:
            filtered_slots.append(
                AgentInputSlot(
                    name=key,
                    value=value,
                    type=slot_definition.type_name if slot_definition else "any",
                    allowed_values=slot_definition.values
                    if isinstance(slot_definition, CategoricalSlot)
                    else None,
                )
            )

    return filtered_slots


def _update_agent_input_metadata_with_events(
    metadata: Dict[str, Any], agent_id: str, flow_id: str, tracker: DialogueStateTracker
) -> None:
    """Update the agent input metadata with the events."""
    agent_started_events = [
        event
        for event in tracker.events
        if type(event) == AgentStarted
        and event.agent_id == agent_id
        and event.flow_id == flow_id
    ]
    if agent_started_events:
        # If we have context ID from the previous agent run, we want to
        # include it in the metadata so that the agent can continue the same
        # context.
        agent_started_event = agent_started_events[-1]
        if agent_started_event.context_id:
            metadata[A2A_AGENT_CONTEXT_ID_KEY] = agent_started_event.context_id


################################################################################
# Other helper methods
################################################################################


def _update_agent_events(events: List[Event], metadata: Dict[str, Any]) -> None:
    """Update the agent events based on the agent output metadata if needed."""
    if A2A_AGENT_CONTEXT_ID_KEY in metadata:
        # If the context ID is present, we need to store it in the AgentStarted
        # event, so that it can be re-used later in case the agent is restarted.
        for event in events:
            if isinstance(event, AgentStarted):
                event.context_id = metadata[A2A_AGENT_CONTEXT_ID_KEY]


def _reset_slots_covered_by_exit_if(
    exit_conditions: List[str], tracker: DialogueStateTracker
) -> None:
    """Reset the slots covered by the exit_if condition."""
    reset_slot_names = get_slot_names_from_exit_conditions(exit_conditions)
    for slot_name in reset_slot_names:
        if tracker.slots.get(slot_name) is not None:
            tracker.update(SlotSet(slot_name, None))
