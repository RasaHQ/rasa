from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Text, List, Optional, Union
from rasa.cdu.flow_stack import FlowStack, FlowStackFrame, StackFrameType

from rasa.core.constants import (
    DEFAULT_POLICY_PRIORITY,
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
)
from pypred import Predicate

from rasa.shared.constants import FLOW_PREFIX
from rasa.shared.nlu.constants import (
    ACTION_NAME,
)
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_SEND_TEXT_NAME,
    CORRECTED_SLOTS_SLOT,
    FLOW_STACK_SLOT,
    PREVIOUS_FLOW_SLOT,
)
from rasa.shared.core.events import ActiveLoop, Event, SlotSet
from rasa.shared.core.flows.flow import (
    END_STEP,
    START_STEP,
    ActionFlowStep,
    ElseFlowLink,
    EndFlowStep,
    Flow,
    FlowStep,
    FlowsList,
    GenerateResponseFlowStep,
    IfFlowLink,
    EntryPromptFlowStep,
    QuestionScope,
    StepThatCanStartAFlow,
    UserMessageStep,
    LinkFlowStep,
    SetSlotsFlowStep,
    QuestionFlowStep,
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
    def supported_stack_frames() -> List[StackFrameType]:
        return [
            StackFrameType.CORRECTION,
            StackFrameType.INTERRUPT,
            StackFrameType.LINK,
            StackFrameType.RESUME,
            StackFrameType.REGULAR,
        ]

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
        self, flow_stack: FlowStack, all_flows: FlowsList, domain: Domain
    ) -> None:
        """Initializes the `FlowExecutor`.

        Args:
            flow_stack_frame: State of the flow.
            all_flows: All flows.
        """
        self.flow_stack = flow_stack
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
        flow_stack = FlowStack.from_tracker(tracker)
        return FlowExecutor(flow_stack, flows or FlowsList([]), domain)

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

        text_slots = dict(
            {
                slot.name: get_value(tracker.get_slot(slot.name))
                for slot in self.domain.slots
            }
        )
        p = Predicate(predicate)
        evaluation, _ = p.analyze(text_slots)
        return evaluation

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
            raise ValueError(
                "No link was selected, but links are present. Links "
                "must cover all possible cases."
            )
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
        structlogger.debug("flow.step.next", next=step, current=current_step, flow=flow)
        return step

    def _slot_for_question(self, question: Text) -> Slot:
        """Find the slot for a question."""
        for slot in self.domain.slots:
            if slot.name == question:
                return slot
        else:
            raise FlowException(
                f"Question '{question}' does not map to an existing slot."
            )

    def _is_step_completed(
        self, step: FlowStep, tracker: "DialogueStateTracker"
    ) -> bool:
        """Check if a step is completed."""
        if isinstance(step, QuestionFlowStep):
            return tracker.get_slot(step.question) is not None
        else:
            return True

    def _find_earliest_updated_question(
        self, current_step: FlowStep, flow: Flow, updated_slots: List[Text]
    ) -> Optional[FlowStep]:
        """Find the question that was updated."""
        asked_question_steps = flow.previously_asked_questions(current_step.id)

        for question_step in reversed(asked_question_steps):
            if question_step.question in updated_slots:
                return question_step
        return None

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
        if self.flow_stack.is_empty():
            # if there are no flows, there is nothing to do
            return ActionPrediction(None, 0.0)
        else:
            prediction = self._select_next_action(tracker)
            if FlowStack.from_tracker(tracker).as_dict() != self.flow_stack.as_dict():
                # we need to update the flow stack to persist the state of the executor
                if not prediction.events:
                    prediction.events = []
                prediction.events.append(
                    SlotSet(
                        FLOW_STACK_SLOT,
                        self.flow_stack.as_dict(),
                    )
                )
            return prediction

    def _correct_flow_position(
        self,
        newly_set_slots: List[Text],
        step: FlowStep,
        flow: Flow,
        tracker: DialogueStateTracker,
    ) -> None:
        reset_point = self._find_earliest_updated_question(step, flow, newly_set_slots)

        if reset_point:
            structlogger.info(
                "flow.reset.slotupdate",
                step=step,
                flow=flow,
                reset_point=reset_point.id,
            )
            self.flow_stack.advance_top_flow(reset_point.id)

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
        predicted_action: Optional[ActionPrediction] = None

        tracker = tracker.copy()

        number_of_initial_events = len(tracker.events)

        while not predicted_action or predicted_action.score == 0.0:
            if not (current_flow := self.flow_stack.top_flow(self.all_flows)):
                # If there is no current flow, we assume that all flows are done
                # and there is nothing to do. The assumption here is that every
                # flow ends with an action listen.
                predicted_action = ActionPrediction(ACTION_LISTEN_NAME, 1.0)
                break

            if not (previous_step := self.flow_stack.top_flow_step(self.all_flows)):
                raise FlowException(
                    "The current flow is set, but there is no current step. "
                    "This should not happen, if a flow is started it should be set "
                    "to __start__ if it ended it should be popped from the stack."
                )

            structlogger.debug("flow.action.loop", previous_step=previous_step)
            predicted_action = self._wrap_up_previous_step(
                current_flow, previous_step, tracker
            )
            tracker.update_with_events(predicted_action.events or [], self.domain)

            if predicted_action.action_name:
                # if the previous step predicted an action, we'll stop here
                # the step is not completed yet and we need to predict the
                # action first before we can try again to wrap up this step and
                # advance to the next one
                break

            current_step = self._select_next_step(tracker, previous_step, current_flow)

            if current_step:
                # this can't be an else, because the previous if might change
                # this to "not None"
                self.flow_stack.advance_top_flow(current_step.id)

                predicted_action = self._run_step(current_flow, current_step, tracker)
                tracker.update_with_events(predicted_action.events or [], self.domain)

        gathered_events = list(tracker.events)[number_of_initial_events:]
        predicted_action.events = gathered_events
        return predicted_action

    def _reset_scoped_slots(
        self, current_flow: Flow, tracker: DialogueStateTracker
    ) -> List[Event]:
        """Reset all scoped slots."""
        events: List[Event] = []
        for step in current_flow.steps:
            # reset all slots scoped to the flow
            if isinstance(step, QuestionFlowStep) and step.scope == QuestionScope.FLOW:
                slot = tracker.slots.get(step.question, None)
                initial_value = slot.initial_value if slot else None
                events.append(SlotSet(step.question, initial_value))
        return events

    @staticmethod
    def _predict_question_loop(
        tracker: DialogueStateTracker, loop_name: Text
    ) -> Optional[Text]:

        is_finished = (
            tracker.latest_action
            and tracker.latest_action.get(ACTION_NAME) == loop_name
            and not tracker.active_loop
        )

        if is_finished:
            return None

        active_loop_rejected = tracker.is_active_loop_rejected
        should_predict_loop = (
            not active_loop_rejected
            and tracker.latest_action
            and tracker.latest_action.get(ACTION_NAME) != loop_name
        )

        if should_predict_loop:
            structlogger.debug("flow.question.loop", loop=loop_name)
            return loop_name
        else:
            structlogger.debug("flow.question.noloop")
            return ACTION_LISTEN_NAME

    def _wrap_up_previous_step(
        self,
        flow: Flow,
        step: FlowStep,
        tracker: DialogueStateTracker,
    ) -> ActionPrediction:
        """Try to wrap up the previous step.

        Args:
            current_flow: The current flow.
            step: The previous step.
            tracker: The tracker to run the step on.

        Returns:
        The predicted action and the events to run.
        """
        structlogger.debug("flow.step.wrapup", step=step, flow=flow)
        if isinstance(step, QuestionFlowStep):
            # the question is only finished once the slot is set and the loop
            # is finished
            loop_name = "question_" + step.question
            action_name = self._predict_question_loop(tracker, loop_name)

            if action_name:
                # loop is not yet done
                return ActionPrediction(action_name, 1.0)
            else:
                return ActionPrediction(None, 0.0)
        else:
            return ActionPrediction(None, 0.0)

    def _run_step(
        self,
        flow: Flow,
        step: FlowStep,
        tracker: DialogueStateTracker,
    ) -> ActionPrediction:
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
        A tuple of the predicted action and a list of events.
        """
        if isinstance(step, QuestionFlowStep):
            structlogger.debug("flow.step.run.question", step=step, flow=flow)
            slot = tracker.slots.get(step.question, None)
            initial_value = slot.initial_value if slot else None
            slot_value = slot.value if slot else None
            if step.skip_if_filled and slot_value != initial_value:
                return ActionPrediction(None, 0.0)

            question_action = ActionPrediction("question_" + step.question, 1.0)
            if slot_value != initial_value:
                question_action.events = [SlotSet(step.question, initial_value)]
            return question_action

        elif isinstance(step, ActionFlowStep):
            structlogger.debug("flow.step.run.action", step=step, flow=flow)
            if not step.action:
                raise FlowException(f"Action not specified for step {step}")
            return ActionPrediction(step.action, 1.0)
        elif isinstance(step, LinkFlowStep):
            structlogger.debug("flow.step.run.link", step=step, flow=flow)
            self.flow_stack.push(
                FlowStackFrame(
                    flow_id=step.link,
                    step_id=START_STEP,
                    frame_type=StackFrameType.LINK,
                )
            )
            if tracker.active_loop_name:
                return ActionPrediction(None, 0.0, events=[ActiveLoop(None)])
            else:
                return ActionPrediction(None, 0.0)
        elif isinstance(step, SetSlotsFlowStep):
            structlogger.debug("flow.step.run.slot", step=step, flow=flow)
            return ActionPrediction(
                None,
                0.0,
                events=[SlotSet(slot["key"], slot["value"]) for slot in step.slots],
            )
        elif isinstance(step, UserMessageStep):
            return ActionPrediction(None, 0.0)
        elif isinstance(step, EntryPromptFlowStep):
            return ActionPrediction(None, 0.0)
        elif isinstance(step, GenerateResponseFlowStep):
            generated = step.generate(tracker)
            return ActionPrediction(
                ACTION_SEND_TEXT_NAME, 1.0, metadata={"message": {"text": generated}}
            )
        elif isinstance(step, EndFlowStep):
            # this is the end of the flow, so we'll pop it from the stack
            events = self._reset_scoped_slots(flow, tracker)
            structlogger.debug("flow.step.run.flowend", flow=flow)
            if current_frame := self.flow_stack.pop():
                previous_flow = self.flow_stack.top_flow(self.all_flows)
                previous_flow_step = self.flow_stack.top_flow_step(self.all_flows)
                if current_frame.frame_type == StackFrameType.INTERRUPT:
                    # get stack frame that is below the current one and which will
                    # be continued now that this one has ended.
                    previous_flow_name = (
                        previous_flow.name or previous_flow.id
                        if previous_flow
                        else None
                    )

                    # TODO: rather than triggering the flow-trigger-action this should
                    #   directly put the flow onto the stack and start it that way.
                    return ActionPrediction(
                        FLOW_PREFIX + "pattern_continue_interrupted",
                        1.0,
                        metadata={"slots": {PREVIOUS_FLOW_SLOT: previous_flow_name}},
                        events=events,
                    )
                elif (
                    previous_flow
                    and previous_flow_step
                    and current_frame.frame_type == StackFrameType.CORRECTION
                ):
                    # TODO: we need to figure out how to actually
                    #    "undo" the changed slots
                    corrected_slots = tracker.get_slot(CORRECTED_SLOTS_SLOT)
                    if corrected_slots:
                        self._correct_flow_position(
                            corrected_slots, previous_flow_step, previous_flow, tracker
                        )
                    else:
                        # TODO: we need to figure out how to actually "undo" the
                        #    changed slots
                        pass
            return ActionPrediction(None, 0.0, events=events)
        else:
            raise FlowException(f"Unknown flow step type {type(step)}")
