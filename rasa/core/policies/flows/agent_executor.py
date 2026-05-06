from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import structlog

from rasa.agents.agent_manager import AgentManager
from rasa.agents.constants import (
    A2A_AGENT_CONTEXT_ID_KEY,
    AGENT_METADATA_AGENT_ID_KEY,
    AGENT_METADATA_AGENT_RESPONSE_KEY,
    AGENT_METADATA_CANCELLATION_REASON_KEY,
    AGENT_METADATA_EXIT_IF_KEY,
    AGENT_METADATA_MODEL_ID_KEY,
    AGENT_METADATA_RESTARTED_KEY,
    AGENT_METADATA_RESUMED_AFTER_INTERRUPTION,
    AGENT_METADATA_SENDER_ID_KEY,
    AGENT_METADATA_STRUCTURED_RESULTS_KEY,
    MAX_AGENT_RETRY_DELAY_SECONDS,
)
from rasa.agents.core.cancellation import CancellationToken
from rasa.agents.core.types import AgentStatus, ProtocolType
from rasa.agents.schemas import AgentInput, AgentOutput
from rasa.agents.schemas.agent_input import AgentInputSlot
from rasa.agents.utils import map_agent_metadata_to_bot_uttered
from rasa.core.channels.channel import OutputChannel
from rasa.core.config.configuration import Configuration
from rasa.core.constants import (
    ACTIVE_FLOW_METADATA_KEY,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FINAL_RESPONSE,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_INPUT_REQUIRED,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY,
    BOT_UTTERANCE_AGENT_NAME_KEY,
    STEP_ID_METADATA_KEY,
)
from rasa.core.policies.flows.flow_step_result import (
    ContinueFlowWithNextStep,
    FlowActionPrediction,
    FlowStepResult,
    PauseFlowReturnPrediction,
)
from rasa.core.utils import get_slot_names_from_exit_conditions
from rasa.dialogue_understanding.patterns.cancel import CancelPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.internal_error import (
    INTERNAL_ERROR_SOURCE_AGENT,
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
    BaseFlowStackFrame,
    UserFlowStackFrame,
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
from rasa.shared.core.flows.steps.constants import END_STEP
from rasa.shared.core.flows.steps.continuation import ContinueFlowStep
from rasa.shared.core.slots import CategoricalSlot, Slot
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.utils.llm import tracker_as_readable_transcript

structlogger = structlog.get_logger()

MAX_AGENT_RETRIES = 3


def normalize_agent_output_events(
    events: Optional[List[Any]],
    *,
    agent_name: Optional[str] = None,
) -> List[Event]:
    """Coerce agent ``output.events`` entries to Rasa :class:`Event` instances.

    Flow code (e.g. ``attach_stack_metadata_to_events``) assumes each item is an
    ``Event`` with a ``metadata`` dict. Agents may return serialized dicts; those
    are parsed via :meth:`Event.from_parameters`. Unsupported values are omitted
    and an error is logged.

    Args:
        events: Raw list from :class:`~rasa.agents.schemas.agent_output.AgentOutput`.
        agent_name: Optional agent id for structured logs.

    Returns:
        A new list containing only valid ``Event`` instances.
    """
    if not events:
        return []

    normalized: List[Event] = []
    for index, item in enumerate(events):
        if isinstance(item, Event):
            if not isinstance(item.metadata, dict):
                structlogger.error(
                    "flow_executor.normalize_agent_output_events.invalid_metadata_type",
                    event_info=(
                        "Agent output event has non-dict metadata, "
                        "replacing with empty dict."
                    ),
                    agent_name=agent_name,
                    implementation_class=item.__class__.__name__,
                    metadata_type=type(item.metadata).__name__,
                )
                item.metadata = {}
            normalized.append(item)
            continue

        if isinstance(item, dict):
            try:
                parsed = Event.from_parameters(item)
            except Exception as e:
                structlogger.error(
                    "flow_executor.normalize_agent_output_events.parse_failed",
                    event_info=(
                        "Failed to parse agent output event dict into a Rasa Event."
                    ),
                    agent_name=agent_name,
                    error=str(e),
                    declared_type=item.get("event"),
                    parameter_keys=sorted(item.keys()),
                )
                continue
            if parsed is None:
                structlogger.error(
                    "flow_executor.normalize_agent_output_events.unsupported_dict_event",
                    event_info=(
                        "Agent output event dict is not a supported Rasa event "
                        "(missing or unknown `event` type)."
                    ),
                    agent_name=agent_name,
                    declared_type=item.get("event"),
                    parameter_keys=sorted(item.keys()),
                )
                continue
            if not isinstance(parsed.metadata, dict):
                parsed.metadata = {}
            normalized.append(parsed)
            continue

        structlogger.error(
            "flow_executor.normalize_agent_output_events.unsupported_event_type",
            event_info=(
                "Dropping agent output event: not a Rasa Event instance. "
                "Send an event of `rasa.shared.core.events.Event` type instead."
            ),
            agent_name=agent_name,
            unsupported_event_type=type(item).__name__,
        )
    return normalized


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


def _tracker_has_prior_agent_completed(
    tracker: DialogueStateTracker, agent_id: str, flow_id: str
) -> bool:
    """Return True if a completed run for this agent/flow is already on the tracker."""
    return any(
        isinstance(e, AgentCompleted)
        and e.agent_id == agent_id
        and e.flow_id == flow_id
        for e in tracker.events
    )


def _effective_agent_restart(
    stack: DialogueStack,
    step: CallFlowStep,
    tracker: DialogueStateTracker,
    agent_stack_frame: Optional[AgentStackFrame],
) -> bool:
    """Whether this call should reset exit_if slots and mark the agent as restarted.

    Yes after a ``restart agent`` command or when the same agent/flow already finished
    once in this tracker. No while the user is still in the middle of the same agent
    turn (waiting for the next message).
    """
    # If the agent stack frame is a restart, we need to reset the exit_if slots
    if agent_stack_frame is not None and agent_stack_frame.is_restart:
        return True

    # If the user is still in the middle of the same agent turn,
    # we don't need to reset the exit_if slots
    active = stack.find_active_agent_frame()
    continuing_non_restart = (
        active is not None
        and active.agent_id == step.call
        and active.flow_id == step.flow_id
        and not active.is_restart
    )
    if continuing_non_restart:
        return False

    # If the agent/flow already finished once in this tracker,
    # we need to reset the exit_if slots
    return _tracker_has_prior_agent_completed(tracker, step.call, step.flow_id)


async def run_agent(
    initial_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
    tracker: DialogueStateTracker,
    slots: List[Slot],
    flows: FlowsList,
    output_channel: Optional[OutputChannel] = None,
    cancellation_token: Optional[CancellationToken] = None,
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
    agent_config = Configuration.get_instance().available_agents.get_agent_config(
        step.call
    )

    if (
        agent_stack_frame
        and agent_stack_frame == stack.top()
        and agent_stack_frame.state == AgentState.INTERRUPTED
    ):
        structlogger.debug(
            "flow.step.run_agent.resuming_interrupted_agent",
            agent_id=step.call,
            flow_id=step.flow_id,
        )

        # Reinvoke the agent with resume context; events are still submitted.
        final_events.append(AgentResumed(agent_id=step.call, flow_id=step.flow_id))
        agent_input = _prepare_agent_input(
            agent_stack_frame, step, tracker, slots, restarted=False
        )
        last_request = (agent_stack_frame.metadata or {}).get(
            AGENT_METADATA_AGENT_RESPONSE_KEY, ""
        ) or ""
        agent_input = agent_input.model_copy(
            update={
                "metadata": {
                    **agent_input.metadata,
                    AGENT_METADATA_RESUMED_AFTER_INTERRUPTION: True,
                    AGENT_METADATA_AGENT_RESPONSE_KEY: last_request,
                }
            }
        )
    else:
        # Reset exit_if slots when explicitly restarting or re-entering after a prior
        # completed run.
        effective_restart = _effective_agent_restart(
            stack, step, tracker, agent_stack_frame
        )
        if step.exit_if and effective_restart:
            _reset_slots_covered_by_exit_if(step.exit_if, tracker)

        # generate the agent input
        agent_input = _prepare_agent_input(
            agent_stack_frame, step, tracker, slots, restarted=effective_restart
        )

        # add the AgentStarted event to the list of final events
        agent_metadata: Dict[str, Any] = {}
        if agent_config:
            agent_metadata["description"] = agent_config.agent.description
            if agent_config.connections:
                servers = agent_config.connections.mcp_servers or []
                agent_metadata["mcp_tools"] = [
                    t for s in servers for t in (s.include_tools or [])
                ]
                agent_metadata["excluded_mcp_tools"] = [
                    t for s in servers for t in (s.exclude_tools or [])
                ]
        if step.exit_if:
            agent_metadata["exit_conditions"] = step.exit_if
        final_events.append(
            AgentStarted(step.call, step.flow_id, metadata=agent_metadata)
        )

    structlogger.debug(
        "flow.step.run_agent.agent_input",
        agent_name=step.call,
        step_id=step.id,
        flow_id=step.flow_id,
        agent_input=agent_input.model_dump(),
        json_formatting=["agent_input"],
    )

    # send the input to the agent and wait for a response
    protocol_type = get_protocol_type(step, agent_config)
    output: AgentOutput = await _call_agent_with_retry(
        agent_name=step.call,
        protocol_type=protocol_type,
        agent_input=agent_input,
        max_retries=MAX_AGENT_RETRIES,
        output_channel=output_channel,
        cancellation_token=cancellation_token,
    )

    # Ensure baseline metadata for agent name if the agent didn't provide it.
    # Prefer agent-provided metadata if present.
    output.metadata = output.metadata or {}
    output.metadata.setdefault(BOT_UTTERANCE_AGENT_NAME_KEY, step.call)

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
        normalized_events = normalize_agent_output_events(
            output.events, agent_name=step.call
        )
        output.events = normalized_events or None
        final_events.extend(normalized_events)

    # handle the agent output based on the agent status
    if output.status == AgentStatus.INPUT_REQUIRED:
        return _handle_agent_input_required(output, final_events, stack, step)
    elif output.status == AgentStatus.COMPLETED:
        return _handle_agent_completed(output, final_events, stack, step)
    elif output.status == AgentStatus.CANCELLED:
        return _handle_agent_cancelled(output, final_events, stack, step)
    elif output.status == AgentStatus.FATAL_ERROR:
        return _handle_agent_fatal_error(
            output, final_events, stack, step, flows, tracker, protocol_type
        )
    else:
        return _handle_agent_unknown_status(
            output, final_events, stack, step, flows, tracker, protocol_type
        )


async def _call_agent_with_retry(
    agent_name: str,
    protocol_type: ProtocolType,
    agent_input: AgentInput,
    max_retries: int,
    output_channel: Optional[OutputChannel] = None,
    cancellation_token: Optional[CancellationToken] = None,
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
                cancellation_token=cancellation_token,
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
# Handle agent output
################################################################################


def _internal_error_info_for_agent_failure(
    step: CallFlowStep,
    output: AgentOutput,
    protocol_type: Optional[ProtocolType] = None,
) -> Dict[str, Any]:
    """Populate ``info`` for internal-error stack frame (agent failure)."""
    info: Dict[str, Any] = {
        "error_source": INTERNAL_ERROR_SOURCE_AGENT,
        "agent_name": step.call,
        "flow_id": step.flow_id,
        "step_id": step.id,
    }
    if protocol_type is not None:
        info["agent_type"] = protocol_type.value
    if output.error_message:
        info["error_message"] = output.error_message
    return info


def _handle_agent_unknown_status(
    output: AgentOutput,
    final_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
    flows: FlowsList,
    tracker: DialogueStateTracker,
    protocol_type: Optional[ProtocolType] = None,
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
    stack.push(
        InternalErrorPatternFlowStackFrame(
            info=_internal_error_info_for_agent_failure(step, output, protocol_type),
        )
    )
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
    defaults = _build_default_agent_message_metadata(
        step, BOT_UTTERANCE_AGENT_MESSAGE_TYPE_INPUT_REQUIRED
    )
    output.metadata = {**defaults, **(output.metadata or {})}
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
        output.response_message, final_events, output.metadata
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
    defaults = _build_default_agent_message_metadata(
        step, BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FINAL_RESPONSE
    )
    output.metadata = {**defaults, **(output.metadata or {})}
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
            _create_send_text_prediction(
                output.response_message, final_events, output.metadata
            )
        )
    else:
        return ContinueFlowWithNextStep(events=final_events)


def _handle_agent_cancelled(
    output: AgentOutput,
    final_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
) -> FlowStepResult:
    """Handle cancellation of agent execution.

    Silently ends the owning flow: removes the agent and flow stack frames,
    appends ``AgentCancelled`` and ``FlowCancelled`` events, and returns
    ``ContinueFlowWithNextStep``.  The loop will see no active flow on the
    stack and fall through to ``action_listen``.

    Unlike fatal errors, ``pattern_internal_error`` is **not** triggered.
    Unlike user-initiated cancellation, ``CancelPatternFlowStackFrame`` is
    **not** pushed, so no bot message is sent and no flow-level side-effects
    occur.
    """
    reason = (output.metadata or {}).get(AGENT_METADATA_CANCELLATION_REASON_KEY)
    structlogger.info(
        "flow.step.run_agent.cancelled",
        agent_name=step.call,
        step_id=step.id,
        flow_id=step.flow_id,
        reason=reason,
    )
    remove_agent_stack_frame(stack, step.call)
    # Remove the owning flow frame so the flow doesn't advance to the next step
    stack.frames = [
        f
        for f in stack.frames
        if not (isinstance(f, UserFlowStackFrame) and f.flow_id == step.flow_id)
    ]
    final_events.append(
        AgentCancelled(agent_id=step.call, flow_id=step.flow_id, reason=reason)
    )
    final_events.append(FlowCancelled(step.flow_id, step.id))
    return ContinueFlowWithNextStep(events=final_events)


def _mark_canceled_frames_ended(stack: DialogueStack) -> None:
    """Mark flow frames that were canceled as ended (END_STEP). Skips agent frames."""
    from rasa.dialogue_understanding.commands import CancelFlowCommand

    canceled_frame_ids = CancelFlowCommand.select_canceled_frames(stack)
    for frame in stack.frames:
        if (
            frame.frame_id in canceled_frame_ids
            and isinstance(frame, BaseFlowStackFrame)
            and not isinstance(frame, AgentStackFrame)
        ):
            frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)


def _handle_agent_fatal_error(
    output: AgentOutput,
    final_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
    flows: FlowsList,
    tracker: DialogueStateTracker,
    protocol_type: Optional[ProtocolType] = None,
) -> FlowStepResult:
    """Handle fatal error from agent execution.

    Cancels all agents of the failing step's flow (same flow) and ends that flow
    for tracking. Agents belonging to other flows are left on the stack (e.g. when
    the user digressed to this flow). No cancel-pattern frame is pushed so the user
    only sees the internal error message, not a "flow cancelled" utterance.

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

    # Cancel all agents of this flow; leave agents from other flows on the stack.
    canceled_agent_ids = set()
    for frame in stack.find_agent_stack_frames_for_flow(step.flow_id):
        remove_agent_stack_frame(stack, frame.agent_id)
        canceled_agent_ids.add(frame.agent_id)
        final_events.append(
            AgentCancelled(
                agent_id=frame.agent_id,
                flow_id=step.flow_id,
                reason=output.error_message,
            )
        )
    # If the failing agent was never on the stack (e.g. started then failed immediately
    # before INPUT_REQUIRED), final_events already has AgentStarted; still emit
    # AgentCancelled so the event stream records the cancellation.
    if step.call not in canceled_agent_ids:
        final_events.append(
            AgentCancelled(
                agent_id=step.call,
                flow_id=step.flow_id,
                reason=output.error_message,
            )
        )

    # Mark the current flow as ended (no cancel pattern).
    _mark_canceled_frames_ended(stack)
    final_events.append(FlowCancelled(step.flow_id, step.id))

    stack.push(
        InternalErrorPatternFlowStackFrame(
            info=_internal_error_info_for_agent_failure(step, output, protocol_type),
        )
    )
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
    action_name: str,
    message: Optional[str],
    events: Optional[List[Event]],
    metadata: Optional[Dict[str, Any]] = None,
) -> FlowActionPrediction:
    """Create a prediction for an action with a text message."""
    # Build message payload, filtering out None values
    message_payload: dict[str, Any] = {}
    if message is not None:
        message_payload[ACTION_METADATA_TEXT_KEY] = message
    action_metadata: dict[str, Any] = {ACTION_METADATA_MESSAGE_KEY: message_payload}

    if metadata:
        mapped = map_agent_metadata_to_bot_uttered(metadata)
        action_metadata[ACTION_METADATA_MESSAGE_KEY].update(mapped)

    return FlowActionPrediction(
        action_name,
        1.0,
        events=events if events else [],
        metadata=action_metadata,
    )


def _create_agent_request_user_input_prediction(
    message: Optional[str],
    events: Optional[List[Event]],
    metadata: Optional[Dict[str, Any]] = None,
) -> FlowActionPrediction:
    """Create a prediction for requesting user input from the agent and waiting for it.

    This function creates a prediction that will pause the flow and wait for user input.
    """
    return _create_action_prediction(
        ACTION_AGENT_REQUEST_USER_INPUT_NAME, message, events, metadata
    )


def _create_send_text_prediction(
    message: Optional[str],
    events: Optional[List[Event]],
    metadata: Optional[Dict[str, Any]] = None,
) -> FlowActionPrediction:
    """Create a prediction for sending a text message to the user."""
    return _create_action_prediction(ACTION_SEND_TEXT_NAME, message, events, metadata)


################################################################################
# Prepare agent input
################################################################################


def _prepare_agent_input(
    agent_stack_frame: Optional[AgentStackFrame],
    step: CallFlowStep,
    tracker: DialogueStateTracker,
    slots: List[Slot],
    restarted: bool = False,
) -> AgentInput:
    """Prepare the agent input data.

    Args:
        agent_stack_frame: The agent stack frame if it exists
        step: The flow step that called the agent
        tracker: The dialogue state tracker
        slots: List of slot definitions
        restarted: When True, set restarted metadata for agents (e.g. MCP).

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

    if restarted:
        agent_input_metadata[AGENT_METADATA_RESTARTED_KEY] = True

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


def _build_default_agent_message_metadata(
    step: CallFlowStep, message_type: Optional[str] = None
) -> Dict[str, Any]:
    """Construct default metadata for agent-generated bot messages.

    Populates fields that can be derived from the flow step itself, so customers
    customizing agents still get baseline metadata even if they don't attach it
    to AgentOutput explicitly.
    """
    base: Dict[str, Any] = {
        BOT_UTTERANCE_AGENT_NAME_KEY: step.call,
        ACTIVE_FLOW_METADATA_KEY: step.flow_id,
        STEP_ID_METADATA_KEY: step.id,
    }
    if message_type:
        base[BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY] = message_type
    return base
