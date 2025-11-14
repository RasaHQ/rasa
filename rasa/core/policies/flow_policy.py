from __future__ import annotations

from typing import Any, Dict, List, Optional, Text

import structlog

from rasa.core.channels.channel import OutputChannel
from rasa.core.constants import (
    FLOW_POLICY_PRIORITY,
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
)
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.policies.flows import flow_executor
from rasa.core.policies.flows.flow_exceptions import FlowCircuitBreakerTrippedException
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames import (
    BaseFlowStackFrame,
    DialogueStackFrame,
)
from rasa.dialogue_understanding.stack.utils import (
    end_top_user_flow,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import (
    DialogueStateTracker,
)

structlogger = structlog.get_logger()


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
        """Checks if the policy supports the topmost frame on the dialogue stack.

        If `False` is returned, the policy will abstain from making a prediction.

        Args:
            frame: The frame to check.

        Returns:
        `True` if the policy supports the frame, `False` otherwise.
        """
        return isinstance(frame, BaseFlowStackFrame)

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""
        # please make sure to update the docs when changing a default parameter
        return {
            POLICY_PRIORITY: FLOW_POLICY_PRIORITY,
            POLICY_MAX_HISTORY: None,
        }

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
        return self.resource

    async def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        flows: Optional[FlowsList] = None,
        output_channel: Optional[OutputChannel] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: The tracker containing the conversation history up to now.
            domain: The model's domain.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.
            flows: The flows to use.
            output_channel: The output channel to use.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to make predictions.

        Returns:
             The prediction.
        """
        if not self.supports_current_stack_frame(
            tracker
        ) or self.should_abstain_in_coexistence(tracker, True):
            # if the policy doesn't support the current stack frame, we'll abstain
            return self._prediction(self._default_predictions(domain))

        flows = flows or FlowsList(underlying_flows=[])

        # create executor and predict next action
        try:
            prediction = await flow_executor.advance_flows(
                tracker,
                domain.action_names_or_texts,
                flows,
                domain.slots,
                output_channel=output_channel,
            )
            return self._create_prediction_result(
                prediction.action_name,
                domain,
                prediction.score,
                prediction.events,
                prediction.metadata,
            )
        except FlowCircuitBreakerTrippedException as e:
            structlogger.error(
                "flow.circuit_breaker",
                number_of_steps_taken=e.number_of_steps_taken,
                event_info=(
                    "The flow circuit breaker tripped. "
                    "There appears to be an infinite loop in the flows."
                ),
                error=str(e),
            )
            # end the current flow and start the internal error flow
            updated_stack = tracker.stack
            updated_stack = end_top_user_flow(updated_stack)
            updated_stack.push(InternalErrorPatternFlowStackFrame())
            # we retry, with the internal error frame on the stack
            events = tracker.create_stack_updated_events(updated_stack)
            tracker.update_with_events(events)
            prediction = await flow_executor.advance_flows(
                tracker,
                domain.action_names_or_texts,
                flows,
                domain.slots,
                output_channel=output_channel,
            )
            collected_events = events + (prediction.events or [])
            return self._create_prediction_result(
                prediction.action_name,
                domain,
                prediction.score,
                collected_events,
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
            events: The events to return.
            action_metadata: The metadata of the predicted action.

        Returns:
            The prediction result where the score is used for one hot encoding.
        """
        result = self._default_predictions(domain)
        if action_name:
            result[domain.index_for_action(action_name)] = score
        return self._prediction(
            result, optional_events=events, action_metadata=action_metadata
        )
