from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Text

from rasa.core.policies.policy import PolicyPrediction
from rasa.engine.graph import GraphComponent
from rasa.engine.runner.interface import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    ActionExecutionRejected,
    DefinePrevUserUtteredFeaturization,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import InvalidConfigException, RasaException

logger = logging.getLogger(__name__)


def is_not_in_training_data(
    policy_name: Optional[Text], max_confidence: Optional[float] = None
) -> bool:
    """Checks whether the prediction is empty or by a policy which did not memoize data.

    Args:
        policy_name: The name of the policy.
        max_confidence: The max confidence of the policy's prediction.

    Returns:
        `False` if and only if an action was predicted (i.e. `max_confidence` > 0) by
        a `MemoizationPolicy`
    """
    from rasa.core.policies.memoization import (
        AugmentedMemoizationPolicy,
        MemoizationPolicy,
    )
    from rasa.core.policies.rule_policy import RulePolicy

    if not policy_name:
        return True

    memorizing_policies = [
        RulePolicy.__name__,
        MemoizationPolicy.__name__,
        AugmentedMemoizationPolicy.__name__,
    ]
    is_memorized = any(
        policy_name.endswith(f"_{memoizing_policy}")
        for memoizing_policy in memorizing_policies
    )

    # also check if confidence is 0, than it cannot be count as prediction
    return not is_memorized or max_confidence == 0.0


class InvalidPolicyEnsembleConfig(RasaException):
    """Exception that can be raised when the policy ensemble is not valid."""


class PolicyPredictionEnsemble(ABC):
    """Interface for any policy prediction ensemble.

    Given a list of predictions from policies, which include some meta data about the
    policies themselves, an "ensemble" decides what the final prediction should be, in
    the following way:
    1. If the previously predicted action was rejected, then the ensemble sets the
       probability for this action to 0.0 (in all given predictions).
    2. It combines the information from the single predictions, which include some
       meta data about the policies (e.g. priority), into a final prediction.
    3. If the sequence of events given at the time of prediction ends with a user
       utterance, then the ensemble adds a special event to the event-list included in
       the final prediction that indicates whether the final prediction was made based
       on the actual text of that user utterance.

    Observe that policies predict "mandatory" as well as "optional"
    events. The ensemble decides which of the optional events should
    be passed on.
    """

    def combine_predictions_from_kwargs(
        self, tracker: DialogueStateTracker, domain: Domain, **kwargs: Any
    ) -> PolicyPrediction:
        """Derives a single prediction from predictions given as kwargs.

        Args:
            tracker: dialogue state tracker holding the state of the conversation,
              which may influence the combination of predictions as well
            domain: the common domain
            **kwargs: arbitrary keyword arguments. All policy predictions passed as
              kwargs will be combined.

        Returns:
            a single prediction
        """
        predictions = [
            value for value in kwargs.values() if isinstance(value, PolicyPrediction)
        ]
        return self.combine_predictions(
            predictions=predictions, tracker=tracker, domain=domain
        )

    @abstractmethod
    def combine_predictions(
        self,
        predictions: List[PolicyPrediction],
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> PolicyPrediction:
        """Derives a single prediction from the given list of predictions.

        Args:
            predictions: a list of policy predictions that include "confidence scores"
              which are non-negative but *do not* necessarily up to 1
            tracker: dialogue state tracker holding the state of the conversation,
              which may influence the combination of predictions as well
            domain: the common domain

        Returns:
            a single prediction
        """
        ...


class DefaultPolicyPredictionEnsemble(PolicyPredictionEnsemble, GraphComponent):
    """An ensemble that picks the "best" prediction and combines events from all.

    The following rules determine which prediction is the "best":
    1. "No user" predictions overrule all other predictions.

    2. End-to-end predictions overrule all other predictions based on
        user input - if and only if *no* "no user" prediction is present in the
        given ensemble.

    3. Given two predictions, if the maximum confidence of one prediction is
        strictly larger than that of the other, then the prediction with the
        strictly larger maximum confidence is considered to be "better".
        The priorities of the policies that made these predictions does not matter.

    4. Given two predictions of policies that are equally confident, the
        prediction of the policy with the higher priority is considered to be
        "better".

    Observe that this comparison is *not* symmetric if the priorities are allowed to
    coincide (i.e. if we cannot distinguish two predictions using 1.-4., then
    the first prediction is considered to be "better").

    The list of events in the final prediction will contain all mandatory
    events contained in the given predictions, the optional events given in the
    "best" prediction, and `DefinePrevUserUtteredFeaturization` event (if the
    prediction was made for a sequence of events ending with a user utterance).
    """

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DefaultPolicyPredictionEnsemble:
        """Creates a new instance (see parent class for full docstring)."""
        return cls()

    def __str__(self) -> Text:
        return f"{self.__class__.__name__}()"

    @staticmethod
    def _pick_best_policy(predictions: List[PolicyPrediction]) -> PolicyPrediction:
        """Picks the best policy prediction based on probabilities and policy priority.

        Args:
            predictions: a list containing policy predictions

        Returns:
            The index of the best prediction
        """
        best_confidence = (-1.0, -1)
        best_index = -1

        # different type of predictions have different priorities
        # No user predictions overrule all other predictions.
        is_no_user_prediction = any(
            prediction.is_no_user_prediction for prediction in predictions
        )
        # End-to-end predictions overrule all other predictions based on user input.
        is_end_to_end_prediction = any(
            prediction.is_end_to_end_prediction for prediction in predictions
        )

        policy_events = []
        for idx, prediction in enumerate(predictions):
            policy_events += prediction.events

            # No user predictions (e.g. happy path loop predictions)
            # overrule all other predictions.
            if prediction.is_no_user_prediction != is_no_user_prediction:
                continue

            # End-to-end predictions overrule all other predictions based on user input.
            if (
                not is_no_user_prediction
                and prediction.is_end_to_end_prediction != is_end_to_end_prediction
            ):
                continue

            confidence = (prediction.max_confidence, prediction.policy_priority)
            if confidence > best_confidence:
                # pick the best policy
                best_confidence = confidence
                best_index = idx

        if best_index < 0:
            raise InvalidConfigException(
                "No best prediction found. Please check your model configuration."
            )

        best_prediction = predictions[best_index]
        policy_events += best_prediction.optional_events

        return PolicyPrediction(
            best_prediction.probabilities,
            best_prediction.policy_name,
            best_prediction.policy_priority,
            policy_events,
            is_end_to_end_prediction=best_prediction.is_end_to_end_prediction,
            is_no_user_prediction=best_prediction.is_no_user_prediction,
            diagnostic_data=best_prediction.diagnostic_data,
            hide_rule_turn=best_prediction.hide_rule_turn,
            action_metadata=best_prediction.action_metadata,
        )

    @staticmethod
    def _best_policy_prediction(
        predictions: List[PolicyPrediction],
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> PolicyPrediction:
        """Finds the best policy prediction.

        Args:
            predictions: a list of policy predictions that include "confidence scores"
              which are non-negative but *do not* necessarily up to 1
            tracker: dialogue state tracker holding the state of the conversation,
              which may influence the combination of predictions as well
            domain: the common domain

        Returns:
            The winning policy prediction.
        """
        last_action_event = next(
            (
                event
                for event in reversed(tracker.events)
                if isinstance(event, (ActionExecutionRejected, ActionExecuted))
            ),
            None,
        )

        rejected_action_name = None
        if len(tracker.events) > 0 and isinstance(
            last_action_event, ActionExecutionRejected
        ):
            rejected_action_name = last_action_event.action_name

        if rejected_action_name:
            logger.debug(
                f"Execution of '{rejected_action_name}' was rejected. "
                f"Setting its confidence to 0.0 in all predictions."
            )
            index_of_rejected_action = domain.index_for_action(rejected_action_name)
            for prediction in predictions:
                prediction.probabilities[index_of_rejected_action] = 0.0

        return DefaultPolicyPredictionEnsemble._pick_best_policy(predictions)

    def combine_predictions(
        self,
        predictions: List[PolicyPrediction],
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> PolicyPrediction:
        """Derives a single prediction from the given list of predictions.

        Note that you might get unexpected results if the priorities are non-unique.
        Moreover, the order of events in the result is determined by the order of the
        predictions passed to this method.

        Args:
            predictions: a list of policy predictions that include "probabilities"
              which are non-negative but *do not* necessarily up to 1
            tracker: dialogue state tracker holding the state of the conversation
            domain: the common domain

        Returns:
            The "best" prediction.
        """
        if not predictions:
            raise InvalidConfigException(
                "Expected at least one prediction. Please check your model "
                "configuration."
            )
        # Reminder: If just a single policy is given, we do *not* just return it because
        # it is expected that the final prediction contains mandatory and optional
        # events in the `events` attribute and no optional events.

        winning_prediction = self._best_policy_prediction(
            predictions=predictions, domain=domain, tracker=tracker
        )

        if tracker.latest_action_name == ACTION_LISTEN_NAME:
            if winning_prediction.is_end_to_end_prediction:
                logger.debug("Made e2e prediction using user text.")
                logger.debug("Added `DefinePrevUserUtteredFeaturization(True)` event.")
                winning_prediction.events.append(
                    DefinePrevUserUtteredFeaturization(True)
                )
            else:
                logger.debug("Made prediction using user intent.")
                logger.debug("Added `DefinePrevUserUtteredFeaturization(False)` event.")
                winning_prediction.events.append(
                    DefinePrevUserUtteredFeaturization(False)
                )

        logger.debug(f"Predicted next action using {winning_prediction.policy_name}.")
        return winning_prediction
