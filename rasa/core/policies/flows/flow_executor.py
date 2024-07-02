from __future__ import annotations

from typing import Any, Dict, Text, List, Optional

from jinja2 import Template
from structlog.contextvars import (
    bound_contextvars,
)
from rasa.core.policies.flows.flow_exceptions import (
    FlowCircuitBreakerTrippedException,
    FlowException,
    NoNextStepInFlowException,
)
from rasa.core.policies.flows.flow_step_result import (
    FlowActionPrediction,
    ContinueFlowWithNextStep,
    FlowStepResult,
    PauseFlowReturnPrediction,
)
from rasa.dialogue_understanding.patterns.search import SearchPatternFlowStackFrame
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    BaseFlowStackFrame,
    DialogueStackFrame,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.completed import (
    CompletedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.continue_interrupted import (
    ContinueInterruptedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    FlowStackFrameType,
)
from rasa.dialogue_understanding.stack.utils import (
    top_user_flow_frame,
)

from pypred import Predicate

from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.events import (
    Event,
    FlowCompleted,
    FlowResumed,
    FlowStarted,
    SlotSet,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.flow_step_links import (
    StaticFlowStepLink,
    IfFlowStepLink,
    ElseFlowStepLink,
)
from rasa.shared.core.flows.steps import (
    ActionFlowStep,
    SetSlotsFlowStep,
    LinkFlowStep,
    ContinueFlowStep,
    EndFlowStep,
    CallFlowStep,
    CollectInformationFlowStep,
    NoOperationFlowStep,
)
from rasa.shared.core.flows.flow import (
    END_STEP,
    Flow,
    FlowStep,
)
from rasa.shared.core.flows.steps.collect import SlotRejection
from rasa.shared.core.trackers import (
    DialogueStateTracker,
)
import structlog

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
            document=document,
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
            tracker=tracker,
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
            tracker=tracker,
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


def events_for_collect_step_execution(
    step: CollectInformationFlowStep, tracker: DialogueStateTracker
) -> List[Event]:
    """Create the events needed to prepare for the execution of a collect step."""
    # reset the slots that always need to be explicitly collected
    slot = tracker.slots.get(step.collect, None)

    if slot and step.ask_before_filling:
        return [SlotSet(step.collect, None)]
    else:
        return []


def trigger_pattern_continue_interrupted(
    current_frame: DialogueStackFrame, stack: DialogueStack, flows: FlowsList
) -> List[Event]:
    """Trigger the pattern to continue an interrupted flow if needed."""
    events: List[Event] = []

    # get previously started user flow that will be continued
    interrupted_user_flow_frame = top_user_flow_frame(stack)
    interrupted_user_flow_step = (
        interrupted_user_flow_frame.step(flows) if interrupted_user_flow_frame else None
    )
    interrupted_user_flow = (
        interrupted_user_flow_frame.flow(flows) if interrupted_user_flow_frame else None
    )

    if (
        isinstance(current_frame, UserFlowStackFrame)
        and interrupted_user_flow_step is not None
        and interrupted_user_flow is not None
        and current_frame.frame_type == FlowStackFrameType.INTERRUPT
        and not is_step_end_of_flow(interrupted_user_flow_step)
    ):
        stack.push(
            ContinueInterruptedPatternFlowStackFrame(
                previous_flow_name=interrupted_user_flow.readable_name(),
            )
        )
        events.append(
            FlowResumed(interrupted_user_flow.id, interrupted_user_flow_step.id)
        )

    return events


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
        events.append(SlotSet(slot_name, initial_value))

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

    for step in current_flow.steps_with_calls_resolved:
        if isinstance(step, CollectInformationFlowStep):
            # reset all slots scoped to the flow
            if step.reset_after_flow_ends:
                _reset_slot(step.collect, tracker)
            else:
                not_resettable_slot_names.add(step.collect)

    # slots set by the set slots step should be reset after the flow ends
    # unless they are also used in a collect step where `reset_after_flow_ends`
    # is set to `False`
    resettable_set_slots = [
        slot["key"]
        for step in current_flow.steps_with_calls_resolved
        if isinstance(step, SetSlotsFlowStep)
        for slot in step.slots
        if slot["key"] not in not_resettable_slot_names
    ]

    for name in resettable_set_slots:
        _reset_slot(name, tracker)

    return events


def advance_flows(
    tracker: DialogueStateTracker, available_actions: List[str], flows: FlowsList
) -> FlowActionPrediction:
    """Advance the current flows until the next action.

    Args:
        tracker: The tracker to get the next action for.
        available_actions: The actions that are available in the domain.
        flows: All flows.

    Returns:
    The predicted action and the events to run.
    """
    stack = tracker.stack
    if stack.is_empty():
        # if there are no flows, there is nothing to do
        return FlowActionPrediction(None, 0.0)

    return advance_flows_until_next_action(tracker, available_actions, flows)


def advance_flows_until_next_action(
    tracker: DialogueStateTracker,
    available_actions: List[str],
    flows: FlowsList,
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
                step_result = run_step(
                    next_step,
                    current_flow,
                    step_stack,
                    tracker,
                    available_actions,
                    flows,
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
                tracker.update_stack(step_stack)
                tracker.update_with_events(new_events)

    gathered_events = list(tracker.events)[number_of_initial_events:]
    if isinstance(step_result, PauseFlowReturnPrediction):
        prediction = step_result.action_prediction
        # make sure we really return all events that got created during the
        # step execution of all steps (not only the last one)
        prediction.events = gathered_events
        return prediction
    else:
        structlogger.warning("flow.step.execution.no_action")
        return FlowActionPrediction(None, 0.0, events=gathered_events)


def run_step(
    step: FlowStep,
    flow: Flow,
    stack: DialogueStack,
    tracker: DialogueStateTracker,
    available_actions: List[str],
    flows: FlowsList,
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

    Returns:
    A result of running the step describing where to transition to.
    """
    initial_events: List[Event] = []
    if step == flow.first_step_in_flow():
        initial_events.append(FlowStarted(flow.id))

    if isinstance(step, CollectInformationFlowStep):
        structlogger.debug("flow.step.run.collect")
        trigger_pattern_ask_collect_information(
            step.collect, stack, step.rejections, step.utter, step.collect_action
        )

        events: List[Event] = events_for_collect_step_execution(step, tracker)
        return ContinueFlowWithNextStep(events=initial_events + events)

    elif isinstance(step, ActionFlowStep):
        if not step.action:
            raise FlowException(f"Action not specified for step {step}")

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
                utter_action_name = render_template_variables(
                    "{{context.utter}}", context
                )
                if utter_action_name not in available_actions:
                    structlogger.warning(
                        "flow.step.run.action.unknown", action=action_name
                    )
            return ContinueFlowWithNextStep(events=initial_events)

    elif isinstance(step, LinkFlowStep):
        structlogger.debug("flow.step.run.link")
        stack.push(
            UserFlowStackFrame(
                flow_id=step.link,
                frame_type=FlowStackFrameType.LINK,
            ),
            # push this below the current stack frame so that we can
            # complete the current flow first and then continue with the
            # linked flow
            index=-1,
        )
        return ContinueFlowWithNextStep(events=initial_events)

    elif isinstance(step, CallFlowStep):
        structlogger.debug("flow.step.run.call")
        stack.push(
            UserFlowStackFrame(
                flow_id=step.call,
                frame_type=FlowStackFrameType.CALL,
            ),
        )
        return ContinueFlowWithNextStep()

    elif isinstance(step, SetSlotsFlowStep):
        structlogger.debug("flow.step.run.slot")
        slot_events: List[Event] = events_from_set_slots_step(step)
        return ContinueFlowWithNextStep(events=initial_events + slot_events)

    elif isinstance(step, NoOperationFlowStep):
        structlogger.debug("flow.step.run.no_operation")
        return ContinueFlowWithNextStep(events=initial_events)

    elif isinstance(step, EndFlowStep):
        # this is the end of the flow, so we'll pop it from the stack
        structlogger.debug("flow.step.run.flow_end")
        current_frame = stack.pop()
        trigger_pattern_completed(current_frame, stack, flows)
        resumed_events = trigger_pattern_continue_interrupted(
            current_frame, stack, flows
        )
        reset_events: List[Event] = reset_scoped_slots(current_frame, flow, tracker)
        return ContinueFlowWithNextStep(
            events=initial_events + reset_events + resumed_events, has_flow_ended=True
        )

    else:
        raise FlowException(f"Unknown flow step type {type(step)}")
