from __future__ import annotations

from typing import Any, Dict, List, Optional, Text

import structlog
from jinja2 import Template
from structlog.contextvars import bound_contextvars

from rasa.core.channels.channel import OutputChannel
from rasa.core.channels.voice_stream.call_state import (
    call_state,
)
from rasa.core.config.configuration import Configuration
from rasa.core.constants import ACTIVE_FLOW_METADATA_KEY, STEP_ID_METADATA_KEY
from rasa.core.policies.flows.agent_executor import run_agent
from rasa.core.policies.flows.flow_exceptions import (
    FlowCircuitBreakerTrippedException,
    FlowException,
    NoNextStepInFlowException,
)
from rasa.core.policies.flows.flow_step_result import (
    ContinueFlowWithNextStep,
    FlowActionPrediction,
    FlowStepResult,
    PauseFlowReturnPrediction,
)
from rasa.core.policies.flows.mcp_tool_executor import call_mcp_tool
from rasa.dialogue_understanding.patterns.cancel import CancelPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.collect_information import (
    FLOW_PATTERN_COLLECT_INFORMATION,
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.completed import (
    CompletedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.continue_interrupted import (
    ContinueInterruptedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.human_handoff import (
    HumanHandoffPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.search import SearchPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.user_silence import FLOW_PATTERN_USER_SILENCE
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    BaseFlowStackFrame,
    DialogueStackFrame,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    FlowStackFrameType,
)
from rasa.dialogue_understanding.stack.utils import (
    user_frames_on_the_stack,
)
from rasa.dialogue_understanding.utils import assemble_options_string
from rasa.shared.constants import RASA_PATTERN_HUMAN_HANDOFF
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE,
    SILENCE_TIMEOUT_CHANNEL_KEY,
    SILENCE_TIMEOUT_SLOT,
)
from rasa.shared.core.events import (
    Event,
    FlowCompleted,
    FlowResumed,
    FlowStarted,
    SlotSet,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.flow import END_STEP, Flow, FlowStep
from rasa.shared.core.flows.flow_step_links import (
    ElseFlowStepLink,
    IfFlowStepLink,
    StaticFlowStepLink,
)
from rasa.shared.core.flows.steps import (
    ActionFlowStep,
    CallFlowStep,
    CollectInformationFlowStep,
    ContinueFlowStep,
    EndFlowStep,
    LinkFlowStep,
    NoOperationFlowStep,
    SetSlotsFlowStep,
)
from rasa.shared.core.flows.steps.constants import START_STEP
from rasa.shared.core.slots import Slot, SlotRejection
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.pypred import Predicate

structlogger = structlog.get_logger()

MAX_NUMBER_OF_STEPS = 250


def render_template_variables(text: str, context: Dict[Text, Any]) -> str:
    """Replace context variables in a text."""
    return Template(text).render(context)


def is_condition_satisfied(
    predicate: Text, context: Dict[str, Any], tracker: DialogueStateTracker
) -> bool:
    """Evaluate a predicate condition."""
    # attach context to the predicate evaluation to allow conditions using it
    context = {"context": context}

    document: Dict[str, Any] = context.copy()
    # add slots namespace to the document
    document["slots"] = tracker.current_slot_values()

    rendered_condition = render_template_variables(predicate, context)
    p = Predicate(rendered_condition)
    structlogger.debug(
        "flow.predicate.evaluating",
        condition=predicate,
        rendered_condition=rendered_condition,
    )
    try:
        return p.evaluate(document)
    except (TypeError, Exception) as e:
        structlogger.error(
            "flow.predicate.error",
            predicate=predicate,
            error=str(e),
        )
        return False


def is_step_end_of_flow(step: FlowStep) -> bool:
    """Check if a step is the end of a flow."""
    return (
        step.id == END_STEP
        or
        # not quite at the end but almost, so we'll treat it as the end
        step.id == ContinueFlowStep.continue_step_for_id(END_STEP)
    )


def select_next_step_id(
    current: FlowStep,
    condition_evaluation_context: Dict[str, Any],
    tracker: DialogueStateTracker,
) -> Optional[Text]:
    """Selects the next step id based on the current step."""
    # if the current step is a call step to an agent, and we already have an
    # AgentStackFrame on top of the stack, we need to return the current
    # step id again in order to loop back to the agent.
    top_stack_frame = tracker.stack.top()
    if top_stack_frame and isinstance(top_stack_frame, AgentStackFrame):
        return current.id

    next_step = current.next
    if len(next_step.links) == 1 and isinstance(next_step.links[0], StaticFlowStepLink):
        return next_step.links[0].target

    # evaluate if conditions
    for link in next_step.links:
        if isinstance(link, IfFlowStepLink) and link.condition:
            if is_condition_satisfied(
                link.condition, condition_evaluation_context, tracker
            ):
                structlogger.debug(
                    "flow.link.if_condition_satisfied",
                    current_id=current.id,
                    target=link.target,
                )
                return link.target

    # evaluate else condition
    for link in next_step.links:
        if isinstance(link, ElseFlowStepLink):
            structlogger.debug(
                "flow.link.else_condition_satisfied",
                current_id=current.id,
                target=link.target,
            )
            return link.target

    if next_step.links:
        structlogger.error(
            "flow.link.failed_to_select_branch",
            current=current,
            links=next_step.links,
            sender_id=tracker.sender_id,
        )
        return None

    if current.id == END_STEP:
        # we are already at the very end of the flow. There is no next step.
        return None
    elif isinstance(current, LinkFlowStep):
        # link steps don't have a next step, so we'll return the end step
        return END_STEP
    else:
        structlogger.error(
            "flow.step.failed_to_select_next_step",
            step=current,
            sender_id=tracker.sender_id,
        )
        return None


def select_next_step(
    current_step: FlowStep,
    current_flow: Flow,
    stack: DialogueStack,
    tracker: DialogueStateTracker,
) -> Optional[FlowStep]:
    """Get the next step to execute."""
    next_id = select_next_step_id(current_step, stack.current_context(), tracker)
    step = current_flow.step_by_id(next_id)
    structlogger.debug(
        "flow.step.next",
        next_id=step.id if step else None,
        current_id=current_step.id,
        flow_id=current_flow.id,
    )
    return step


def update_top_flow_step_id(updated_id: str, stack: DialogueStack) -> DialogueStack:
    """Update the top flow on the stack."""
    if (top := stack.top()) and isinstance(top, BaseFlowStackFrame):
        top.step_id = updated_id
    return stack


def events_from_set_slots_step(step: SetSlotsFlowStep) -> List[Event]:
    """Create events from a set slots step."""
    return [SlotSet(slot["key"], slot["value"]) for slot in step.slots]


def trigger_pattern_continue_interrupted(
    current_frame: DialogueStackFrame,
    stack: DialogueStack,
    flows: FlowsList,
    tracker: DialogueStateTracker,
) -> None:
    """Trigger the pattern to continue an interrupted flow if needed."""
    # only trigger the pattern if the current frame is a user flow frame
    # with a frame type of interrupt
    if (
        not isinstance(current_frame, UserFlowStackFrame)
        or current_frame.frame_type != FlowStackFrameType.INTERRUPT
    ):
        return None

    # get all previously interrupted user flows
    interrupted_user_flow_stack_frames = user_frames_on_the_stack(stack)

    interrupted_user_flows_to_continue: List[UserFlowStackFrame] = []
    # check if interrupted user flows can be continued
    # i.e. the flow is not at the end of the flow
    for frame in interrupted_user_flow_stack_frames:
        interrupted_user_flow_step = frame.step(flows)
        interrupted_user_flow = frame.flow(flows)
        if (
            interrupted_user_flow_step is not None
            and interrupted_user_flow is not None
            and not is_step_end_of_flow(interrupted_user_flow_step)
        ):
            interrupted_user_flows_to_continue.append(frame)

    # if there are no interrupted user flows to continue,
    # we don't need to trigger the pattern
    if len(interrupted_user_flows_to_continue) == 0:
        return None

    # get the flow names and ids of the interrupted flows
    # and assemble the options string
    flow_names: List[str] = []
    flow_ids: List[str] = []
    for frame in interrupted_user_flows_to_continue:
        flow_names.append(
            frame.flow(flows).readable_name(language=tracker.current_language)
        )
        flow_ids.append(frame.flow_id)
    options_string = assemble_options_string(flow_names)

    # trigger the pattern to continue the interrupted flows
    stack.push(
        ContinueInterruptedPatternFlowStackFrame(
            interrupted_flow_names=flow_names,
            interrupted_flow_ids=flow_ids,
            interrupted_flow_options=options_string,
        )
    )

    return None


def trigger_pattern_completed(
    current_frame: DialogueStackFrame, stack: DialogueStack, flows: FlowsList
) -> None:
    """Trigger the pattern indicating that the stack is empty, if needed."""
    # trigger pattern if the stack is empty and the last frame was either a user flow
    # frame or a search frame
    if stack.is_empty() and (
        isinstance(current_frame, UserFlowStackFrame)
        or isinstance(current_frame, SearchPatternFlowStackFrame)
    ):
        completed_flow = current_frame.flow(flows)
        if not completed_flow.run_pattern_completed:
            return

        completed_flow_name = completed_flow.readable_name() if completed_flow else None
        stack.push(
            CompletedPatternFlowStackFrame(
                previous_flow_name=completed_flow_name,
            )
        )


def trigger_pattern_ask_collect_information(
    collect: str,
    stack: DialogueStack,
    rejections: List[SlotRejection],
    utter: str,
    collect_action: str,
) -> None:
    """Trigger the pattern to ask for a slot value."""
    stack.push(
        CollectInformationPatternFlowStackFrame(
            collect=collect,
            utter=utter,
            collect_action=collect_action,
            rejections=rejections,
        )
    )


def reset_scoped_slots(
    current_frame: DialogueStackFrame, current_flow: Flow, tracker: DialogueStateTracker
) -> List[Event]:
    """Reset all scoped slots."""

    def _reset_slot(slot_name: Text, dialogue_tracker: DialogueStateTracker) -> None:
        slot = dialogue_tracker.slots.get(slot_name, None)
        initial_value = slot.initial_value if slot else None
        events.append(SlotSet(slot_name, initial_value, metadata={"reset": True}))

    if (
        isinstance(current_frame, UserFlowStackFrame)
        and current_frame.frame_type == FlowStackFrameType.CALL
    ):
        # if a called frame is completed, we don't reset the slots
        # as they are scoped to the called flow. resetting will happen as part
        # of the flow that contained the call step triggering this called flow
        return []

    events: List[Event] = []

    not_resettable_slot_names = set()
    flow_persistable_slots = current_flow.persisted_slots

    for step in current_flow.steps_with_calls_resolved:
        # take persisted slots from called flows into consideration
        # before resetting slots
        if isinstance(step, CallFlowStep) and step.called_flow_reference:
            flow_persistable_slots.extend(step.called_flow_reference.persisted_slots)
        if isinstance(step, CollectInformationFlowStep):
            # reset all slots scoped to the flow
            slot_name = step.collect
            if step.reset_after_flow_ends and slot_name not in flow_persistable_slots:
                _reset_slot(slot_name, tracker)
            else:
                not_resettable_slot_names.add(slot_name)

    # slots set by the set slots step should be reset after the flow ends
    # unless they are also used in a collect step where `reset_after_flow_ends`
    # is set to `False` or set in the `persisted_slots` list.
    resettable_set_slots = _get_resettable_set_slots(
        current_flow, not_resettable_slot_names, flow_persistable_slots
    )
    for name in resettable_set_slots:
        _reset_slot(name, tracker)

    return events


def _get_resettable_set_slots(
    current_flow: Flow,
    not_resettable_slot_names: set[Text],
    flow_persistable_slots: List[Text],
) -> List[Text]:
    """Get list of slot names from SetSlotsFlowStep that should be reset."""
    return [
        slot["key"]
        for step in current_flow.steps_with_calls_resolved
        if isinstance(step, SetSlotsFlowStep)
        for slot in step.slots
        if slot["key"] not in not_resettable_slot_names
        and slot["key"] not in flow_persistable_slots
    ]


async def advance_flows(
    tracker: DialogueStateTracker,
    available_actions: List[str],
    flows: FlowsList,
    slots: List[Slot],
    output_channel: Optional[OutputChannel] = None,
) -> FlowActionPrediction:
    """Advance the current flows until the next action.

    Args:
        tracker: The tracker to get the next action for.
        available_actions: The actions that are available in the domain.
        flows: All flows.
        slots: The slots that are available in the domain.
        output_channel: The output channel to use.

    Returns:
    The predicted action and the events to run.
    """
    stack = tracker.stack
    if stack.is_empty():
        # if there are no flows, there is nothing to do
        return FlowActionPrediction(None, 0.0)

    return await advance_flows_until_next_action(
        tracker, available_actions, flows, slots, output_channel=output_channel
    )


async def advance_flows_until_next_action(
    tracker: DialogueStateTracker,
    available_actions: List[str],
    flows: FlowsList,
    slots: List[Slot],
    output_channel: Optional[OutputChannel] = None,
) -> FlowActionPrediction:
    """Advance the flow and select the next action to execute.

    Advances the current flow and returns the next action to execute. A flow
    is advanced until it is completed or until it predicts an action. If
    the flow is completed, the next flow is popped from the stack and
    advanced. If there are no more flows, the action listen is predicted.

    Args:
        tracker: The tracker to get the next action for.
        available_actions: The actions that are available in the domain.
        flows: All flows.
        slots: The slots that are available in the domain.
        output_channel: The output channel to use.

    Returns:
        The next action to execute, the events that should be applied to the
    tracker and the confidence of the prediction.
    """
    step_result: FlowStepResult = ContinueFlowWithNextStep()

    tracker = tracker.copy()

    number_of_initial_events = len(tracker.events)

    number_of_steps_taken = 0

    while isinstance(step_result, ContinueFlowWithNextStep):
        number_of_steps_taken += 1
        if number_of_steps_taken > MAX_NUMBER_OF_STEPS:
            raise FlowCircuitBreakerTrippedException(
                tracker.stack, number_of_steps_taken
            )

        active_frame = tracker.stack.top()
        if not isinstance(active_frame, BaseFlowStackFrame):
            # If there is no current flow, we assume that all flows are done
            # and there is nothing to do. The assumption here is that every
            # flow ends with an action listen.
            step_result = PauseFlowReturnPrediction(
                FlowActionPrediction(ACTION_LISTEN_NAME, 1.0)
            )
            break

        with bound_contextvars(flow_id=active_frame.flow_id):
            previous_step_id = active_frame.step_id
            structlogger.debug("flow.execution.loop", previous_step_id=previous_step_id)
            current_flow = active_frame.flow(flows)
            next_step = select_next_step(
                active_frame.step(flows), current_flow, tracker.stack, tracker
            )

            if not next_step:
                raise NoNextStepInFlowException(tracker.stack)

            tracker.update_stack(update_top_flow_step_id(next_step.id, tracker.stack))

            with bound_contextvars(step_id=next_step.id):
                step_stack = tracker.stack
                step_result = await run_step(
                    next_step,
                    current_flow,
                    step_stack,
                    tracker,
                    available_actions,
                    flows,
                    previous_step_id,
                    slots,
                    output_channel=output_channel,
                )
                new_events = step_result.events
                if (
                    isinstance(step_result, ContinueFlowWithNextStep)
                    and step_result.has_flow_ended
                ):
                    # insert flow completed before flow resumed event
                    offset = (
                        -1
                        if new_events and isinstance(new_events[-1], FlowResumed)
                        else 0
                    )
                    idx = len(new_events) + offset
                    new_events.insert(
                        idx, FlowCompleted(active_frame.flow_id, previous_step_id)
                    )
                attach_stack_metadata_to_events(
                    next_step.id, current_flow.id, new_events
                )
                tracker.update_stack(step_stack)
                tracker.update_with_events(new_events)

    gathered_events = list(tracker.events)[number_of_initial_events:]
    if isinstance(step_result, PauseFlowReturnPrediction):
        prediction = step_result.action_prediction
        # make sure we really return all events that got created during the
        # step execution of all steps (not only the last one)
        prediction.events = gathered_events
        prediction.metadata = prediction.metadata or {}
        prediction.metadata[ACTIVE_FLOW_METADATA_KEY] = tracker.active_flow
        prediction.metadata[STEP_ID_METADATA_KEY] = tracker.current_step_id
        return prediction
    else:
        structlogger.warning("flow.step.execution.no_action")
        return FlowActionPrediction(None, 0.0, events=gathered_events)


def validate_collect_step(
    step: CollectInformationFlowStep,
    stack: DialogueStack,
    available_actions: List[str],
    slots: Dict[str, Slot],
    flow_name: str,
) -> bool:
    """Validate that a collect step can be executed.

    A collect step can be executed if either the `utter_ask` or the `action_ask` is
    defined in the domain. If neither is defined, the collect step can still be
    executed if the slot has an initial value defined in the domain, which would cause
    the step to be skipped.
    """
    slot = slots.get(step.collect)
    slot_has_initial_value_defined = slot and slot.initial_value is not None
    if (
        slot_has_initial_value_defined
        or step.utter in available_actions
        or step.collect_action in available_actions
    ):
        return True

    structlogger.error(
        "flow.step.run.collect_missing_utter_or_collect_action",
        slot_name=step.collect,
    )

    cancel_flow_and_push_internal_error(stack, flow_name)

    return False


def cancel_flow_and_push_internal_error(stack: DialogueStack, flow_name: str) -> None:
    """Cancel the top user flow and push the internal error pattern."""
    from rasa.dialogue_understanding.commands import CancelFlowCommand

    top_frame = stack.top()

    if isinstance(top_frame, BaseFlowStackFrame):
        # we need to first cancel the top user flow
        # because we cannot collect one of its slots
        # and therefore should not proceed with the flow
        # after triggering pattern_internal_error
        canceled_frames = CancelFlowCommand.select_canceled_frames(stack)
        stack.push(
            CancelPatternFlowStackFrame(
                canceled_name=flow_name,
                canceled_frames=canceled_frames,
            )
        )
    stack.push(InternalErrorPatternFlowStackFrame())


def attach_stack_metadata_to_events(
    step_id: str,
    flow_id: str,
    events: List[Event],
) -> None:
    """Attach the stack metadata to the events."""
    for event in events:
        event.metadata[STEP_ID_METADATA_KEY] = step_id
        event.metadata[ACTIVE_FLOW_METADATA_KEY] = flow_id


async def run_step(
    step: FlowStep,
    flow: Flow,
    stack: DialogueStack,
    tracker: DialogueStateTracker,
    available_actions: List[str],
    flows: FlowsList,
    previous_step_id: str,
    slots: List[Slot],
    output_channel: Optional[OutputChannel] = None,
) -> FlowStepResult:
    """Run a single step of a flow.

    Returns the predicted action and a list of events that were generated
    during the step. The predicted action can be `None` if the step
    doesn't generate an action. The list of events can be empty if the
    step doesn't generate any events.

    Raises a `FlowException` if the step is invalid.

    Args:
        step: The step to run.
        flow: The flow that the step belongs to.
        stack: The stack that the flow is on.
        tracker: The tracker to run the step on.
        available_actions: The actions that are available in the domain.
        flows: All flows.
        previous_step_id: The ID of the previous step.
        slots: The slots that are available in the domain.
        output_channel: The output channel to use.

    Returns:
    A result of running the step describing where to transition to.
    """
    initial_events: List[Event] = []
    if previous_step_id == START_STEP:
        # if the previous step id is the start step, we need to add a flow
        # started event to the initial events.
        # we can't use the current step to check this, as the current step is the
        # first step in the flow -> other steps might link to this flow, so the
        # only reliable way to check if we are starting a new flow is checking for
        # the START_STEP meta step
        initial_events.append(FlowStarted(flow.id, metadata=stack.current_context()))

    # FLow does not start with collect step or we are not in collect information pattern
    if _first_step_is_not_collect(step, previous_step_id) and not (
        _in_collect_information_pattern(flow) or _in_pattern_user_silence(flow)
    ):
        _append_global_silence_timeout_event(initial_events, tracker)

    if isinstance(step, CollectInformationFlowStep):
        return _run_collect_information_step(
            available_actions,
            initial_events,
            stack,
            step,
            tracker,
            flow.readable_name(),
        )

    elif isinstance(step, ActionFlowStep):
        if not step.action:
            raise FlowException(f"Action not specified for step {step}")
        return _run_action_step(available_actions, initial_events, stack, step)

    elif isinstance(step, LinkFlowStep):
        return _run_link_step(initial_events, stack, step)

    elif isinstance(step, CallFlowStep):
        return await _run_call_step(
            initial_events,
            stack,
            step,
            tracker,
            slots,
            flows,
            output_channel=output_channel,
        )

    elif isinstance(step, SetSlotsFlowStep):
        return _run_set_slot_step(initial_events, step)

    elif isinstance(step, NoOperationFlowStep):
        structlogger.debug("flow.step.run.no_operation")
        return ContinueFlowWithNextStep(events=initial_events)

    elif isinstance(step, EndFlowStep):
        # If pattern collect information flow is ending,
        # we need to reset the silence timeout slot to its global value.
        if flow.id == FLOW_PATTERN_COLLECT_INFORMATION:
            _append_global_silence_timeout_event(initial_events, tracker)

        return _run_end_step(flow, flows, initial_events, stack, tracker)

    else:
        raise FlowException(f"Unknown flow step type {type(step)}")


def _first_step_is_not_collect(
    step: FlowStep,
    previous_step_id: str,
) -> bool:
    """Check if the first step is not a collect information step."""
    return (previous_step_id == START_STEP) and not isinstance(
        step, CollectInformationFlowStep
    )


def _in_collect_information_pattern(flow: Flow) -> bool:
    """Check if the current flow is a collect information pattern."""
    return flow.id == FLOW_PATTERN_COLLECT_INFORMATION


def _in_pattern_user_silence(flow: Flow) -> bool:
    """Check if the current flow is a user silence pattern."""
    return flow.id == FLOW_PATTERN_USER_SILENCE


def _run_end_step(
    flow: Flow,
    flows: FlowsList,
    initial_events: List[Event],
    stack: DialogueStack,
    tracker: DialogueStateTracker,
) -> FlowStepResult:
    # this is the end of the flow, so we'll pop it from the stack
    structlogger.debug("flow.step.run.flow_end")
    current_frame = stack.pop()
    trigger_pattern_completed(current_frame, stack, flows)
    trigger_pattern_continue_interrupted(current_frame, stack, flows, tracker)
    reset_events: List[Event] = reset_scoped_slots(current_frame, flow, tracker)
    return ContinueFlowWithNextStep(
        events=initial_events + reset_events, has_flow_ended=True
    )


def _run_set_slot_step(
    initial_events: List[Event], step: SetSlotsFlowStep
) -> FlowStepResult:
    structlogger.debug("flow.step.run.slot")
    slot_events: List[Event] = events_from_set_slots_step(step)
    return ContinueFlowWithNextStep(events=initial_events + slot_events)


async def _run_call_step(
    initial_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
    tracker: DialogueStateTracker,
    slots: List[Slot],
    flows: FlowsList,
    output_channel: Optional[OutputChannel] = None,
) -> FlowStepResult:
    structlogger.debug("flow.step.run.call")
    if step.is_calling_mcp_tool():
        return await call_mcp_tool(initial_events, stack, step, tracker)
    elif step.is_calling_agent():
        return await run_agent(
            initial_events,
            stack,
            step,
            tracker,
            slots,
            flows,
            output_channel=output_channel,
        )
    else:
        stack.push(
            UserFlowStackFrame(
                flow_id=step.call,
                frame_type=FlowStackFrameType.CALL,
            ),
        )
        return ContinueFlowWithNextStep(events=initial_events)


def _run_link_step(
    initial_events: List[Event], stack: DialogueStack, step: LinkFlowStep
) -> FlowStepResult:
    structlogger.debug("flow.step.run.link")

    if step.link == RASA_PATTERN_HUMAN_HANDOFF:
        linked_stack_frame: DialogueStackFrame = HumanHandoffPatternFlowStackFrame()
    else:
        linked_stack_frame = UserFlowStackFrame(
            flow_id=step.link,
            frame_type=FlowStackFrameType.LINK,
        )

    stack.push(
        linked_stack_frame,
        # push this below the current stack frame so that we can
        # complete the current flow first and then continue with the
        # linked flow
        index=-1,
    )

    return ContinueFlowWithNextStep(events=initial_events)


def _run_action_step(
    available_actions: List[str],
    initial_events: List[Event],
    stack: DialogueStack,
    step: ActionFlowStep,
) -> FlowStepResult:
    context = {"context": stack.current_context()}
    action_name = render_template_variables(step.action, context)

    if action_name in available_actions:
        structlogger.debug("flow.step.run.action", context=context)
        return PauseFlowReturnPrediction(
            FlowActionPrediction(action_name, 1.0, events=initial_events)
        )
    else:
        if step.action != "validate_{{context.collect}}":
            # do not log about non-existing validation actions of collect steps
            utter_action_name = render_template_variables("{{context.utter}}", context)
            if utter_action_name not in available_actions:
                structlogger.warning(
                    "flow.step.run.action.unknown",
                    action=action_name,
                    event_info=(
                        f"The action '{action_name}' is not defined in the domain but "
                        f"getting triggered by the flow '{step.flow_id}'."
                    ),
                )
        return ContinueFlowWithNextStep(events=initial_events)


def _run_collect_information_step(
    available_actions: List[str],
    initial_events: List[Event],
    stack: DialogueStack,
    step: CollectInformationFlowStep,
    tracker: DialogueStateTracker,
    flow_name: str,
) -> FlowStepResult:
    is_step_valid = validate_collect_step(
        step, stack, available_actions, tracker.slots, flow_name
    )

    if not is_step_valid:
        # if we return any other FlowStepResult, the assistant will stay silent
        # instead of triggering the internal error pattern
        return ContinueFlowWithNextStep(events=initial_events)

    structlogger.debug("flow.step.run.collect")
    trigger_pattern_ask_collect_information(
        step.collect, stack, step.rejections, step.utter, step.collect_action
    )

    _set_dtmf_state_if_available(step)

    events: List[Event] = _events_for_collect_step_execution(step, tracker)
    return ContinueFlowWithNextStep(events=initial_events + events)


def _set_dtmf_state_if_available(step: CollectInformationFlowStep) -> None:
    """Set DTMF state if call_state is available (voice channels only).

    Safely attempts to set DTMF collection state. If call_state is not initialized
    (e.g., in text channels), this function silently skips the operation.
    """
    if not step.dtmf:
        return

    try:
        call_state.is_collecting_dtmf = True
        call_state.dtmf_config = step.dtmf
    except (LookupError, RuntimeError):
        # call_state is not initialized - this is expected for non-voice channels
        structlogger.debug(
            "flow.step.run.collect.skip_dtmf",
            event_info="call_state not initialized (non-voice channel)",
        )


def _events_for_collect_step_execution(
    step: CollectInformationFlowStep, tracker: DialogueStateTracker
) -> List[Event]:
    """Create the events needed to prepare for the execution of a collect step."""
    # reset the slots that always need to be explicitly collected

    events = _silence_timeout_events_for_collect_step(step, tracker)

    slot = tracker.slots.get(step.collect, None)
    if slot and step.ask_before_filling:
        events.append(SlotSet(step.collect, None))

    return events


def _silence_timeout_events_for_collect_step(
    step: CollectInformationFlowStep, tracker: DialogueStateTracker
) -> List[Event]:
    events: List[Event] = []

    if step.silence_timeout:
        input_channel_name = tracker.get_latest_input_channel()

        structlogger.debug(
            "flow.step.run.using_step_silence_timeout",
            duration=step.silence_timeout,
            collect=step.collect,
        )

        silence_timeout = step.silence_timeout.get_silence_for_channel(
            input_channel_name
        )
    else:
        input_channel_name = tracker.get_latest_input_channel()
        credentials_config = Configuration.get_instance().credentials

        if credentials_config:
            channel_config = (
                credentials_config.channels.get(input_channel_name)
                if input_channel_name
                else None
            )

            silence_timeout = (
                channel_config.get(
                    SILENCE_TIMEOUT_CHANNEL_KEY, GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE
                )
                if channel_config
                else GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE
            )
        else:
            silence_timeout = GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE

        structlogger.debug(
            "flow.step.run.use_channel_silence_timeout",
            duration=silence_timeout,
            collect=step.collect,
        )

    current_silence_timeout = tracker.get_slot(SILENCE_TIMEOUT_SLOT)

    if current_silence_timeout != silence_timeout:
        events.append(SlotSet(SILENCE_TIMEOUT_SLOT, silence_timeout))

    return events


def _append_global_silence_timeout_event(
    events: List[Event], tracker: DialogueStateTracker
) -> None:
    current_silence_timeout = tracker.get_slot(SILENCE_TIMEOUT_SLOT)
    endpoints = Configuration.get_instance().endpoints
    global_silence_timeout = endpoints.interaction_handling.global_silence_timeout

    if current_silence_timeout != global_silence_timeout:
        events.append(
            SlotSet(
                SILENCE_TIMEOUT_SLOT,
                global_silence_timeout,
            )
        )
