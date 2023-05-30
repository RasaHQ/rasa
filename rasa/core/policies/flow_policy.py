from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Text, List, Optional, Tuple, Union
import logging

from rasa.core.constants import (
    DEFAULT_POLICY_PRIORITY,
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
)
from pypred import Predicate
from rasa.shared.constants import FLOW_PREFIX
from rasa.shared.nlu.constants import ENTITY_ATTRIBUTE_TYPE, INTENT_NAME_KEY
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    FLOW_STACK_SLOT,
    FLOW_STATE_SLOT,
)
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import (
    ActionFlowStep,
    ElseFlowLink,
    Flow,
    FlowLinks,
    FlowStep,
    FlowsList,
    IfFlowLink,
    UserMessageStep,
    LinkFlowStep,
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


logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=False
)
class FlowPolicy(Policy):
    """A policy which handles the flow of the conversation based on flows.

    Flows are loaded from files during training. During prediction,
    the flows are applied.
    """

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
        if not flows:
            # there are no flows available
            logger.debug("No flows available. Skipping prediction.")
            predicted_action = None
            predicted_score = 0.0
            events: List[Event] = []
        elif tracker.active_loop:
            # we are in a loop - we don't want to handle flows in this case
            logger.debug("We are in a loop. Skipping prediction.")
            predicted_action = None
            predicted_score = 0.0
            events = []
        elif current_state_dump := tracker.get_slot(FLOW_STATE_SLOT):
            flow_state = FlowState.from_dict(current_state_dump)
            logger.debug(f"Found flow state: {flow_state}")
            executor = FlowExecutor(flow_state, flows)
            predicted_action, events = executor.select_next_action(tracker, domain)
            predicted_score = 1.0
        elif new_flow := self.find_startable_flow(tracker, flows):
            # there are flows available, but we are not in a flow
            # it looks like we can start a flow, so we'll predict the trigger action
            logger.debug(f"Found startable flow: {new_flow.id}")
            predicted_action = FLOW_PREFIX + new_flow.id
            predicted_score = 1.0
            events = []
        else:
            logger.debug("No startable flow found. Skipping prediction.")
            predicted_action = None
            predicted_score = 0.0
            events = []

        result = self._prediction_result(predicted_action, domain, predicted_score)

        return self._prediction(result, optional_events=events)

    def find_startable_flow(
        self, tracker: DialogueStateTracker, flows: FlowsList
    ) -> Optional[Flow]:
        """Finds a flow which can be started.

        Args:
            tracker: The tracker containing the conversation history up to now.
            domain: The model's domain.
            flows: The flows to use.

        Returns:
            The predicted action and the events to run.
        """
        latest_intent = tracker.latest_message.intent.get(INTENT_NAME_KEY)
        latest_entities = [
            e.get(ENTITY_ATTRIBUTE_TYPE) for e in tracker.latest_message.entities
        ]

        for flow in flows.underlying_flows:
            first_step = flow.start_step()
            if not first_step or not isinstance(first_step, UserMessageStep):
                continue

            if first_step.is_triggered(latest_intent, latest_entities):
                return flow
        return None

    def _prediction_result(
        self, action_name: Optional[Text], domain: Domain, score: Optional[float] = 1.0
    ) -> List[float]:
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
        return result


@dataclass
class FlowState:
    """Represents the current flow step."""

    flow_id: Text
    step_id: Optional[Text] = None

    @staticmethod
    def from_dict(data: Dict[Text, Any]) -> FlowState:
        """Creates a `CurrentFlowStep` from a dictionary.

        Args:
            data: The dictionary to create the `CurrentFlowStep` from.

        Returns:
            The created `CurrentFlowStep`.
        """
        return FlowState(data["flow_id"], data["step_id"])

    def as_dict(self) -> Dict[Text, Any]:
        """Returns the `CurrentFlowStep` as a dictionary.

        Returns:
            The `CurrentFlowStep` as a dictionary.
        """
        return {"flow_id": self.flow_id, "step_id": self.step_id}

    def with_updated_id(self, step_id: Text) -> FlowState:
        """Creates a copy of the `CurrentFlowStep` with the given step id.

        Args:
            step_id: The step id to use for the copy.

        Returns:
            The copy of the `CurrentFlowStep` with the given step id.
        """
        return FlowState(self.flow_id, step_id)

    def __repr__(self) -> Text:
        return f"FlowState(flow_id: {self.flow_id}, step_id: {self.step_id})"


class FlowExecutor:
    """Executes a flow."""

    def __init__(self, flow_state: FlowState, all_flows: FlowsList) -> None:
        """Initializes the `FlowExecutor`.

        Args:
            flow_state: State of the flow.
            all_flows: All flows.
        """
        self.flow_state = flow_state
        self.all_flows = all_flows

    @staticmethod
    def is_condition_satisfied(
        predicate: Text, domain: Domain, tracker: "DialogueStateTracker"
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
            {slot.name: get_value(tracker.get_slot(slot.name)) for slot in domain.slots}
        )
        p = Predicate(predicate)
        evaluation, _ = p.analyze(text_slots)
        return evaluation

    def _evaluate_flow_links(
        self, next: FlowLinks, domain: Domain, tracker: "DialogueStateTracker"
    ) -> Optional[Text]:
        """Evaluate the flow links of a step."""
        if len(next.links) == 1 and isinstance(next.links[0], StaticFlowLink):
            return next.links[0].target

        # evaluate if conditions
        for link in next.links:
            if isinstance(link, IfFlowLink) and link.condition:
                if self.is_condition_satisfied(link.condition, domain, tracker):
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
        return None

    def _get_next_step_from_link(
        self, step: LinkFlowStep, tracker: "DialogueStateTracker", domain: Domain
    ) -> Optional[FlowStep]:
        """Get the next step from a link."""
        if next_step_id := self._evaluate_flow_links(step.next, domain, tracker):
            return self.all_flows.step_by_id(next_step_id, self.flow_state.flow_id)
        else:
            return None

    def _get_next_step(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
        current_step: FlowStep,
        flow_id: Text,
    ) -> Optional[FlowStep]:
        """Get the next step to execute."""
        # If the next step is not specified, we assume that the flow is done
        if not current_step.has_next():
            None

        next_id = self._evaluate_flow_links(current_step.next, domain, tracker)
        if next_id is None:
            return None
        return self.all_flows.step_by_id(next_id, flow_id)

    def _slot_for_question(self, question: Text, domain: Domain) -> Slot:
        """Find the slot for a question."""
        for slot in domain.slots:
            if slot.name == question:
                return slot
        else:
            raise Exception(f"Question '{question}' does not map to an existing slot.")

    def _is_step_completed(
        self, step: FlowStep, tracker: "DialogueStateTracker"
    ) -> bool:
        """Check if a step is completed."""
        if isinstance(step, QuestionFlowStep):
            return tracker.get_slot(step.question) is not None
        elif isinstance(step, LinkFlowStep):
            # The flow step can't be completed this way, it get's completed
            # when the linked flow wraps up and returns to this flow
            return False
        else:
            return True

    def _get_current_flow(self) -> Optional[Flow]:
        if self.flow_state.step_id is None:
            return None

        return self.all_flows.flow_by_id(self.flow_state.flow_id)

    def _get_current_step(self) -> Optional[FlowStep]:
        """Get the current step."""
        current_flow = self._get_current_flow()
        if not current_flow:
            return None

        return current_flow.step_for_id(self.flow_state.step_id)

    def start_flow(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Tuple[Optional[Text], List[Event]]:
        """Start the flow."""
        first_step = self.all_flows.first_step(self.flow_state.flow_id)

        if not first_step:
            return (None, [])

        if isinstance(first_step, UserMessageStep):
            return self._get_next_step(
                tracker, domain, first_step, self.flow_state.flow_id
            )
        else:
            return first_step

    def select_next_action(
        self,
        tracker: "DialogueStateTracker",
        domain: Domain,
    ) -> Tuple[Optional[Text], List[Event]]:
        """Request the next slot and response if needed, else return `None`."""
        if not (current_step := self._get_current_step()):
            # If the next step is not set, we return the first step
            # if there is one
            next_step = self.start_flow(tracker, domain)

        elif not self._is_step_completed(current_step, tracker):
            # TODO: figure out
            raise Exception("Not quite sure what to do here yet.")
        else:
            # If the step is completed, we get the next step
            next_step = self._get_next_step(
                tracker, domain, current_step, self.flow_state.flow_id
            )

        if next_step:
            action, events = self._get_action_for_next_step(next_step, tracker, domain)
            return (action, events)

        if not (current_flow := self._get_current_flow()):
            raise Exception(
                "No current flow, but no next step either. "
                "This should not happen. If there wouldn't be a flow, "
                "the current step should be None."
            )

        # this flow is finished. let's clean up
        events = self._reset_ephemeral_slots(current_flow, tracker)

        # there is no immediate next step, so we check if there is a stack
        # and if there is, we go one level up the stack
        if not (current_stack := tracker.get_slot(FLOW_STACK_SLOT)):
            # If there is no stack, we assume that the flow is done
            # and there is nothing to do. We reset the flow state
            # and return action listen. The assumption here is that every
            # flow ends with an action listen.
            events.append(SlotSet(FLOW_STATE_SLOT, None))
            return (ACTION_LISTEN_NAME, events)
        else:
            # If there is a stack, we pop the last item and return it
            stack_step_dump = current_stack.pop()
            stack_step_info = FlowState.from_dict(stack_step_dump)
            stack_step = self.all_flows.step_by_id(
                stack_step_info.step_id, stack_step_info.flow_id
            )
            next_step = self._get_next_step(
                tracker, domain, stack_step, stack_step_info.flow_id
            )
            action, action_events = self._get_action_for_next_step(
                next_step, tracker, domain
            )
            updated_state = FlowState(
                stack_step_info.flow_id,
                next_step.id if next_step else None,
            )
            events.extend(action_events)
            events.append(SlotSet(FLOW_STACK_SLOT, current_stack))
            events.append(SlotSet(FLOW_STATE_SLOT, updated_state.as_dict()))
            return (action, events)

    def _reset_ephemeral_slots(
        self, current_flow: Flow, tracker: DialogueStateTracker
    ) -> List[Event]:
        """Reset all ephemeral slots."""
        events = []
        for step in current_flow.steps:
            # reset all ephemeral slots
            if isinstance(step, QuestionFlowStep) and step.ephemeral:
                slot = tracker.slots.get(step.question, None)
                initial_value = slot.initial_value if slot else None
                events.append(SlotSet(step.question, initial_value))
        return events

    def _get_action_for_next_step(
        self,
        next_step: FlowStep,
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> Tuple[Optional[Text], List[Event]]:
        """Get the action for the next step."""
        if isinstance(next_step, QuestionFlowStep):
            events = []
            slot = tracker.slots.get(next_step.question, None)
            initial_value = slot.initial_value if slot else None
            if next_step.skip_if_filled:
                if slot != initial_value:
                    # TODO this needs more thought. we can't predict an action
                    # here as we need to go to the next step instead as we should
                    # skip this one

                    # this might currently actually work due to forms
                    pass
            elif slot != initial_value:
                events.append(SlotSet(next_step.question, initial_value))

            events.append(
                SlotSet(
                    FLOW_STATE_SLOT,
                    self.flow_state.with_updated_id(next_step.id).as_dict(),
                )
            )
            return ("question_" + next_step.question, events)
        elif isinstance(next_step, ActionFlowStep):
            if not (action_name := next_step.action):
                raise Exception(f"Action not specified for step {next_step}")
            return (
                action_name,
                [
                    SlotSet(
                        FLOW_STATE_SLOT,
                        self.flow_state.with_updated_id(next_step.id).as_dict(),
                    )
                ],
            )
        elif isinstance(next_step, LinkFlowStep):
            link_id = next_step.link
            current_stack = tracker.get_slot(FLOW_STACK_SLOT) or []
            # TODO: double check, are id's unique across flows or only within a flow?
            #  if they are only unique within a flow, we need to add the flow id
            current_stack.append(
                self.flow_state.with_updated_id(next_step.id).as_dict()
            )
            events = [SlotSet(FLOW_STACK_SLOT, current_stack)]
            sub_flow_action, sub_flow_events = FlowExecutor(
                FlowState(flow_id=link_id), self.all_flows
            ).select_next_action(tracker, domain)
            return (sub_flow_action, events + sub_flow_events)
        else:
            raise Exception(f"Unknown flow step type {type(next_step)}")
