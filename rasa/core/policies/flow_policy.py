from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Text, List, Optional, Union

from jinja2 import Template
from structlog.contextvars import (
    bound_contextvars,
)
from rasa.cdu.stack.dialogue_stack import DialogueStack
from rasa.cdu.stack.frames import (
    BaseFlowStackFrame,
    DialogueStackFrame,
    UserFlowStackFrame,
)
from rasa.cdu.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.cdu.patterns.completed import CompletedPatternFlowStackFrame
from rasa.cdu.patterns.continue_interrupted import (
    ContinueInterruptedPatternFlowStackFrame,
)
from rasa.cdu.stack.frames.flow_stack_frame import FlowStackFrameType
from rasa.cdu.stack.utils import top_user_flow_frame

from rasa.core.constants import (
    DEFAULT_POLICY_PRIORITY,
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
)
from pypred import Predicate

from rasa.shared.constants import FLOW_PREFIX
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_SEND_TEXT_NAME,
    DIALOGUE_STACK_SLOT,
)
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import (
    END_STEP,
    ActionFlowStep,
    BranchFlowStep,
    ContinueFlowStep,
    ElseFlowLink,
    EndFlowStep,
    Flow,
    FlowStep,
    FlowsList,
    GenerateResponseFlowStep,
    IfFlowLink,
    EntryPromptFlowStep,
    CollectInformationScope,
    StepThatCanStartAFlow,
    UserMessageStep,
    LinkFlowStep,
    SetSlotsFlowStep,
    CollectInformationFlowStep,
    StaticFlowLink,
)
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.policies.policy import Policy, PolicyPrediction, SupportedData
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.slots import Slot
from rasa.shared.core.trackers import (
    DialogueStateTracker,
)
import structlog

structlogger = structlog.get_logger()


class FlowException(Exception):
    """Exception that is raised when there is a problem with a flow."""

    pass


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=False
)
class FlowPolicy(Policy):
    """A policy which handles the flow of the conversation based on flows.

    Flows are loaded from files during training. During prediction,
    the flows are applied.
    """

    @staticmethod
    def does_support_stack_frame(frame: DialogueStackFrame) -> bool:
        """Checks if the policy supports the given stack frame."""
        return isinstance(frame, BaseFlowStackFrame)

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""
        # please make sure to update the docs when changing a default parameter
        return {
            POLICY_PRIORITY: DEFAULT_POLICY_PRIORITY,
            POLICY_MAX_HISTORY: None,
        }

    @staticmethod
    def supported_data() -> SupportedData:
        """The type of data supported by this policy.

        By default, this is only ML-based training data. If policies support rule data,
        or both ML-based data and rule data, they need to override this method.

        Returns:
            The data type supported by this policy (ML-based training data).
        """
        return SupportedData.ML_DATA

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: Optional[TrackerFeaturizer] = None,
    ) -> None:
        """Constructs a new Policy object."""
        super().__init__(config, model_storage, resource, execution_context, featurizer)

        self.max_history = self.config.get(POLICY_MAX_HISTORY)
        self.resource = resource

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:
        """Trains a policy.

        Args:
            training_trackers: The story and rules trackers from the training data.
            domain: The model's domain.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to train itself.

        Returns:
            A policy must return its resource locator so that potential children nodes
            can load the policy from the resource.
        """
        # currently, nothing to do here. we have access to the flows during
        # prediction. we might want to store the flows in the future
        # or do some preprocessing here.
        return self.resource

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        flows: Optional[FlowsList] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: The tracker containing the conversation history up to now.
            domain: The model's domain.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to make predictions.

        Returns:
             The prediction.
        """
        if not self.supports_current_stack_frame(tracker):
            return self._prediction(self._default_predictions(domain))

        flows = flows or FlowsList([])
        executor = FlowExecutor.from_tracker(tracker, flows, domain)

        # create executor and predict next action
        prediction = executor.advance_flows(tracker)
        return self._create_prediction_result(
            prediction.action_name,
            domain,
            prediction.score,
            prediction.events,
            prediction.metadata,
        )

    def _create_prediction_result(
        self,
        action_name: Optional[Text],
        domain: Domain,
        score: float = 1.0,
        events: Optional[List[Event]] = None,
        action_metadata: Optional[Dict[Text, Any]] = None,
    ) -> PolicyPrediction:
        """Creates a prediction result.

        Args:
            action_name: The name of the predicted action.
            domain: The model's domain.
            score: The score of the predicted action.

        Resturns:
            The prediction result where the score is used for one hot encoding.
        """
        result = self._default_predictions(domain)
        if action_name:
            result[domain.index_for_action(action_name)] = score
        return self._prediction(
            result, optional_events=events, action_metadata=action_metadata
        )


@dataclass
class ActionPrediction:
    """Represents an action prediction."""

    action_name: Optional[Text]
    """The name of the predicted action."""
    score: float
    """The score of the predicted action."""
    metadata: Optional[Dict[Text, Any]] = None
    """The metadata of the predicted action."""
    events: Optional[List[Event]] = None
    """The events attached to the predicted action."""


class FlowExecutor:
    """Executes a flow."""

    def __init__(
        self, dialogue_stack: DialogueStack, all_flows: FlowsList, domain: Domain
    ) -> None:
        """Initializes the `FlowExecutor`.

        Args:
            dialogue_stack_frame: State of the flow.
            all_flows: All flows.
        """
        self.dialogue_stack = dialogue_stack
        self.all_flows = all_flows
        self.domain = domain

    @staticmethod
    def from_tracker(
        tracker: DialogueStateTracker, flows: FlowsList, domain: Domain
    ) -> FlowExecutor:
        """Creates a `FlowExecutor` from a tracker.

        Args:
            tracker: The tracker to create the `FlowExecutor` from.
            flows: The flows to use.

        Returns:
        The created `FlowExecutor`.
        """
        dialogue_stack = DialogueStack.from_tracker(tracker)
        return FlowExecutor(dialogue_stack, flows or FlowsList([]), domain)

    def find_startable_flow(self, tracker: DialogueStateTracker) -> Optional[Flow]:
        """Finds a flow which can be started.

        Args:
            tracker: The tracker containing the conversation history up to now.
            flows: The flows to use.

        Returns:
            The predicted action and the events to run.
        """
        if (
            not tracker.latest_message
            or tracker.latest_action_name != ACTION_LISTEN_NAME
        ):
            # flows can only be started automatically as a response to a user message
            return None

        for flow in self.all_flows.underlying_flows:
            first_step = flow.first_step_in_flow()
            if not first_step or not isinstance(first_step, StepThatCanStartAFlow):
                continue

            if first_step.is_triggered(tracker):
                return flow
        return None

    def is_condition_satisfied(
        self, predicate: Text, tracker: "DialogueStateTracker"
    ) -> bool:
        """Evaluate a predicate condition."""

        def get_value(
            initial_value: Union[Text, None]
        ) -> Union[Text, float, bool, None]:
            if initial_value is None or isinstance(initial_value, (bool, float)):
                return initial_value

            # if this isn't a bool or float, it's something else
            # the below is a best effort to convert it to something we can
            # use for the predicate evaluation
            initial_value = str(initial_value)  # make sure it's a string

            if initial_value.lower() in ["true", "false"]:
                return initial_value.lower() == "true"

            if initial_value.isnumeric():
                return float(initial_value)

            return initial_value

        # attach context to the predicate evaluation to allow coditions using it
        context = {"context": DialogueStack.from_tracker(tracker).current_context()}
        document: Dict[str, Any] = context.copy()
        for slot in self.domain.slots:
            document[slot.name] = get_value(tracker.get_slot(slot.name))
        p = Predicate(self.render_template_variables(predicate, context))
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

    def _select_next_step_id(
        self, current: FlowStep, tracker: "DialogueStateTracker"
    ) -> Optional[Text]:
        """Selects the next step id based on the current step."""
        next = current.next
        if len(next.links) == 1 and isinstance(next.links[0], StaticFlowLink):
            return next.links[0].target

        # evaluate if conditions
        for link in next.links:
            if isinstance(link, IfFlowLink) and link.condition:
                if self.is_condition_satisfied(link.condition, tracker):
                    return link.target

        # evaluate else condition
        for link in next.links:
            if isinstance(link, ElseFlowLink):
                return link.target

        if next.links:
            structlogger.error(
                "flow.link.failed_to_select_branch",
                current=current,
                links=next.links,
                tracker=tracker,
            )
            return None
        if current.id != END_STEP:
            # we've reached the end of the user defined steps in the flow.
            # every flow should end with an end step, so we add it here.
            return END_STEP
        else:
            # we are already at the very end of the flow. There is no next step.
            return None

    def _select_next_step(
        self,
        tracker: "DialogueStateTracker",
        current_step: FlowStep,
        flow: Flow,
    ) -> Optional[FlowStep]:
        """Get the next step to execute."""
        next_id = self._select_next_step_id(current_step, tracker)
        step = flow.step_by_id(next_id)
        structlogger.debug(
            "flow.step.next",
            next_id=step.id if step else None,
            current_id=current_step.id,
            flow_id=flow.id,
        )
        return step

    @staticmethod
    def render_template_variables(text: str, context: Dict[Text, Any]) -> str:
        """Replace context variables in a text."""
        return Template(text).render(context)

    def _slot_for_collect_information(self, collect_information: Text) -> Slot:
        """Find the slot for a collect information."""
        for slot in self.domain.slots:
            if slot.name == collect_information:
                return slot
        else:
            raise FlowException(
                f"Collect Information '{collect_information}' does not map to "
                f"an existing slot."
            )

    def _is_step_completed(
        self, step: FlowStep, tracker: "DialogueStateTracker"
    ) -> bool:
        """Check if a step is completed."""
        if isinstance(step, CollectInformationFlowStep):
            return tracker.get_slot(step.collect_information) is not None
        else:
            return True

    def consider_flow_switch(self, tracker: DialogueStateTracker) -> ActionPrediction:
        """Consider switching to a new flow.

        Args:
            tracker: The tracker to get the next action for.

        Returns:
        The predicted action and the events to run.
        """
        if new_flow := self.find_startable_flow(tracker):
            # there are flows available, but we are not in a flow
            # it looks like we can start a flow, so we'll predict the trigger action
            structlogger.debug("flow.startable", flow_id=new_flow.id)
            return ActionPrediction(FLOW_PREFIX + new_flow.id, 1.0)
        else:
            structlogger.debug("flow.nostartable")
            return ActionPrediction(None, 0.0)

    def advance_flows(self, tracker: DialogueStateTracker) -> ActionPrediction:
        """Advance the flows.

        Either start a new flow or advance the current flow.

        Args:
            tracker: The tracker to get the next action for.
            domain: The domain to get the next action for.

        Returns:
        The predicted action and the events to run.
        """
        prediction = self.consider_flow_switch(tracker)

        if prediction.action_name:
            # if a flow can be started, we'll start it
            return prediction
        if self.dialogue_stack.is_empty():
            # if there are no flows, there is nothing to do
            return ActionPrediction(None, 0.0)
        else:
            previous_stack = DialogueStack.get_persisted_stack(tracker)
            prediction = self._select_next_action(tracker)
            if previous_stack != self.dialogue_stack.as_dict():
                # we need to update dialogue stack to persist the state of the executor
                if not prediction.events:
                    prediction.events = []
                prediction.events.append(
                    SlotSet(
                        DIALOGUE_STACK_SLOT,
                        self.dialogue_stack.as_dict(),
                    )
                )
            return prediction

    def _select_next_action(
        self,
        tracker: DialogueStateTracker,
    ) -> ActionPrediction:
        """Select the next action to execute.

        Advances the current flow and returns the next action to execute. A flow
        is advanced until it is completed or until it predicts an action. If
        the flow is completed, the next flow is popped from the stack and
        advanced. If there are no more flows, the action listen is predicted.

        Args:
            tracker: The tracker to get the next action for.
            domain: The domain to get the next action for.

        Returns:
            The next action to execute, the events that should be applied to the
        tracker and the confidence of the prediction.
        """
        step_result: FlowStepResult = ContinueFlowWithNextStep()

        tracker = tracker.copy()

        number_of_initial_events = len(tracker.events)

        while isinstance(step_result, ContinueFlowWithNextStep):
            active_frame = self.dialogue_stack.top()
            if not isinstance(active_frame, BaseFlowStackFrame):
                # If there is no current flow, we assume that all flows are done
                # and there is nothing to do. The assumption here is that every
                # flow ends with an action listen.
                step_result = PauseFlowReturnPrediction(
                    ActionPrediction(ACTION_LISTEN_NAME, 1.0)
                )
            else:
                with bound_contextvars(flow_id=active_frame.flow_id):
                    structlogger.debug(
                        "flow.execution.loop", previous_step_id=active_frame.step_id
                    )
                    current_flow = active_frame.flow(self.all_flows)
                    current_step = self._select_next_step(
                        tracker, active_frame.step(self.all_flows), current_flow
                    )

                    if current_step:
                        self._advance_top_flow_on_stack(current_step.id)

                        with bound_contextvars(step_id=current_step.id):
                            step_result = self._run_step(
                                current_flow, current_step, tracker
                            )
                            tracker.update_with_events(step_result.events, self.domain)

        gathered_events = list(tracker.events)[number_of_initial_events:]
        if isinstance(step_result, PauseFlowReturnPrediction):
            prediction = step_result.action_prediction
            # make sure we really return all events that got created during the
            # step execution of all steps (not only the last one)
            prediction.events = gathered_events
            return prediction
        else:
            structlogger.warning("flow.step.execution.no_action")
            return ActionPrediction(None, 0.0)

    def _advance_top_flow_on_stack(self, updated_id: str) -> None:
        if (top := self.dialogue_stack.top()) and isinstance(top, BaseFlowStackFrame):
            top.step_id = updated_id

    def _reset_scoped_slots(
        self, current_flow: Flow, tracker: DialogueStateTracker
    ) -> List[Event]:
        """Reset all scoped slots."""
        events: List[Event] = []
        for step in current_flow.steps:
            # reset all slots scoped to the flow
            if (
                isinstance(step, CollectInformationFlowStep)
                and step.scope == CollectInformationScope.FLOW
            ):
                slot = tracker.slots.get(step.collect_information, None)
                initial_value = slot.initial_value if slot else None
                events.append(SlotSet(step.collect_information, initial_value))
        return events

    def _run_step(
        self,
        flow: Flow,
        step: FlowStep,
        tracker: DialogueStateTracker,
    ) -> FlowStepResult:
        """Run a single step of a flow.

        Returns the predicted action and a list of events that were generated
        during the step. The predicted action can be `None` if the step
        doesn't generate an action. The list of events can be empty if the
        step doesn't generate any events.

        Raises a `FlowException` if the step is invalid.

        Args:
            flow: The flow that the step belongs to.
            step: The step to run.
            tracker: The tracker to run the step on.

        Returns:
        A result of running the step describing where to transition to.
        """
        if isinstance(step, CollectInformationFlowStep):
            structlogger.debug("flow.step.run.collect_information")
            self.trigger_pattern_ask_collect_information(step.collect_information)

            # reset the slot if its already filled and the collect infomation shouldn't
            # be skipped
            slot = tracker.slots.get(step.collect_information, None)
            if slot and slot.has_been_set and step.ask_before_filling:
                events = [SlotSet(step.collect_information, slot.initial_value)]
            else:
                events = []

            return ContinueFlowWithNextStep(events=events)

        elif isinstance(step, ActionFlowStep):
            if not step.action:
                raise FlowException(f"Action not specified for step {step}")
            context = {"context": self.dialogue_stack.current_context()}
            action_name = self.render_template_variables(step.action, context)
            if action_name in self.domain.action_names_or_texts:
                structlogger.debug("flow.step.run.action", context=context)
                return PauseFlowReturnPrediction(ActionPrediction(action_name, 1.0))
            else:
                structlogger.warning("flow.step.run.action.unknown", action=action_name)
                return ContinueFlowWithNextStep()

        elif isinstance(step, LinkFlowStep):
            structlogger.debug("flow.step.run.link")
            self.dialogue_stack.push(
                UserFlowStackFrame(
                    flow_id=step.link,
                    frame_type=FlowStackFrameType.LINK,
                ),
                # push this below the current stack frame so that we can
                # complete the current flow first and then continue with the
                # linked flow
                index=-1,
            )
            return ContinueFlowWithNextStep()

        elif isinstance(step, SetSlotsFlowStep):
            structlogger.debug("flow.step.run.slot")
            return ContinueFlowWithNextStep(
                events=[SlotSet(slot["key"], slot["value"]) for slot in step.slots],
            )

        elif isinstance(step, UserMessageStep):
            structlogger.debug("flow.step.run.user_message")
            return ContinueFlowWithNextStep()

        elif isinstance(step, BranchFlowStep):
            structlogger.debug("flow.step.run.branch")
            return ContinueFlowWithNextStep()

        elif isinstance(step, EntryPromptFlowStep):
            structlogger.debug("flow.step.run.entry_prompt")
            return ContinueFlowWithNextStep()

        elif isinstance(step, GenerateResponseFlowStep):
            structlogger.debug("flow.step.run.generate_response")
            generated = step.generate(tracker)
            return PauseFlowReturnPrediction(
                ActionPrediction(
                    ACTION_SEND_TEXT_NAME,
                    1.0,
                    metadata={"message": {"text": generated}},
                )
            )

        elif isinstance(step, EndFlowStep):
            # this is the end of the flow, so we'll pop it from the stack
            structlogger.debug("flow.step.run.flow_end")
            current_frame = self.dialogue_stack.pop()
            self.trigger_pattern_continue_interrupted(current_frame)
            self.trigger_pattern_completed(current_frame)
            reset_events = self._reset_scoped_slots(flow, tracker)
            return ContinueFlowWithNextStep(events=reset_events)

        else:
            raise FlowException(f"Unknown flow step type {type(step)}")

    def trigger_pattern_continue_interrupted(
        self, current_frame: DialogueStackFrame
    ) -> None:
        """Trigger the pattern to continue an interrupted flow if needed."""
        # get previously started user flow that will be continued
        previous_user_flow_frame = top_user_flow_frame(self.dialogue_stack)
        previous_user_flow_step = (
            previous_user_flow_frame.step(self.all_flows)
            if previous_user_flow_frame
            else None
        )
        previous_user_flow = (
            previous_user_flow_frame.flow(self.all_flows)
            if previous_user_flow_frame
            else None
        )

        if (
            isinstance(current_frame, UserFlowStackFrame)
            and previous_user_flow_step
            and previous_user_flow
            and current_frame.frame_type == FlowStackFrameType.INTERRUPT
            and not self.is_step_end_of_flow(previous_user_flow_step)
        ):
            self.dialogue_stack.push(
                ContinueInterruptedPatternFlowStackFrame(
                    previous_flow_name=previous_user_flow.readable_name(),
                )
            )

    def trigger_pattern_completed(self, current_frame: DialogueStackFrame) -> None:
        """Trigger the pattern indicating that the stack is empty, if needed."""
        if self.dialogue_stack.is_empty() and isinstance(
            current_frame, UserFlowStackFrame
        ):
            completed_flow = current_frame.flow(self.all_flows)
            completed_flow_name = (
                completed_flow.readable_name() if completed_flow else None
            )
            self.dialogue_stack.push(
                CompletedPatternFlowStackFrame(
                    previous_flow_name=completed_flow_name,
                )
            )

    def trigger_pattern_ask_collect_information(self, collect_information: str) -> None:
        self.dialogue_stack.push(
            CollectInformationPatternFlowStackFrame(
                collect_information=collect_information
            )
        )

    @staticmethod
    def is_step_end_of_flow(step: FlowStep) -> bool:
        """Check if a step is the end of a flow."""
        return (
            step.id == END_STEP
            or
            # not quite at the end but almost, so we'll treat it as the end
            step.id == ContinueFlowStep.continue_step_for_id(END_STEP)
        )


class FlowStepResult:
    def __init__(self, events: Optional[List[Event]] = None) -> None:
        self.events = events or []


class ContinueFlowWithNextStep(FlowStepResult):
    def __init__(self, events: Optional[List[Event]] = None) -> None:
        super().__init__(events=events)


class PauseFlowReturnPrediction(FlowStepResult):
    def __init__(self, action_prediction: ActionPrediction) -> None:
        self.action_prediction = action_prediction
        super().__init__(events=action_prediction.events)
