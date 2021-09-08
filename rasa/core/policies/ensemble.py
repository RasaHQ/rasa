from abc import abstractmethod
from typing import Text, Optional, Tuple, List, Dict, Any
import logging
import copy

from rasa.engine.graph import GraphComponent
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.runner.interface import ExecutionContext
import rasa.core
import rasa.core.training.training
from rasa.core.policies.policy import (
    InvalidPolicyConfig,
    Policy,
    SupportedData,
    PolicyPrediction,
)
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.policies._ensemble import SimplePolicyEnsemble, PolicyEnsemble
from rasa.shared.exceptions import RasaException, InvalidConfigException
import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.shared.constants import (
    DOCS_URL_RULES,
    DOCS_URL_POLICIES,
    DEFAULT_CONFIG_PATH,
    DOCS_URL_DEFAULT_ACTIONS,
)
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    USER_INTENT_BACK,
    USER_INTENT_RESTART,
    ACTION_RESTART_NAME,
    ACTION_BACK_NAME,
)
from rasa.shared.core.domain import InvalidDomain, Domain
from rasa.shared.core.events import (
    ActionExecutionRejected,
    ActionExecuted,
    DefinePrevUserUtteredFeaturization,
    Event,
)
from rasa.shared.core.trackers import DialogueStateTracker
import rasa.utils.io

logger = logging.getLogger(__name__)

# TODO: This is a workaround around until we have all components migrated to
# `GraphComponent`.
SimplePolicyEnsemble = SimplePolicyEnsemble
PolicyEnsemble = PolicyEnsemble


class InvalidPolicyEnsembleConfig(RasaException):
    """Exception that can be raised when the policy ensemble is not valid."""


class PolicyPredictionEnsemble:
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

    def predict(
        self,
        predictions: List[PolicyPrediction],
        domain: Domain,
        tracker: DialogueStateTracker,
    ) -> PolicyPrediction:
        """Derives a final prediction from the given list of predictions.

        Args:
            predictions: a list of predictions made by policies
            tracker: dialogue state tracker holding the state of the conversation
            domain: the common domain

        Returns:
            a single prediction
        """
        # apply side constraints and modify the given predictions accordingly
        self._exclude_last_action_from_predictions_if_it_was_rejected(
            tracker, predictions, domain
        )

        # combine the resulting predictions (with probabilities that do not
        # necessarily sum up to one anymore)
        final_prediction = self.combine_predictions(predictions, tracker)

        logger.debug(f"Predicted next action using {final_prediction.policy_name}.")
        return final_prediction

    @staticmethod
    def _exclude_last_action_from_predictions_if_it_was_rejected(
        tracker: DialogueStateTracker,
        predictions: List[PolicyPrediction],
        domain: Domain,
    ) -> None:
        """Sets the probability for the last action to 0 if it was just rejected.

        Args:
          tracker:  dialogue state tracker holding the state of the conversation
          predictions: a list of predictions from policies
          domain: the common domain
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

    @abstractmethod
    def combine_predictions(
        self, predictions: List[PolicyPrediction], tracker: DialogueStateTracker
    ) -> PolicyPrediction:
        """Derives a single prediction from the given list of predictions.

        Args:
            predictions: a list of policy predictions that include "confidence scores"
              which are non-negative but *do not* necessarily up to 1
            tracker: dialogue state tracker holding the state of the conversation,
              which may influence the combination of predictions as well

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
    ) -> GraphComponent:
        """Creates a new instance (see parent class for full docstring)."""
        return cls()

    def __str__(self) -> Text:
        return f"{self.__class__.__name__}()"

    def combine_predictions(
        self, predictions: List[PolicyPrediction], tracker: DialogueStateTracker
    ) -> PolicyPrediction:
        """Derives a single prediction from the given list of predictions.

        Note that you might get unexpected results if the priorities are non-unique.
        Moreover, the order of events in the result is determined by the order of the
        predictions passed to this method.

        Args:
            predictions: a list of policy predictions that include "probabilities"
              which are non-negative but *do not* necessarily up to 1
            tracker: dialogue state tracker holding the state of the conversation

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

        best_idx = self._choose_best_prediction(predictions)
        events = self._choose_events(
            predictions=predictions, best_idx=best_idx, tracker=tracker
        )

        # make a copy and swap out the events - and drop action metadata
        final = copy.copy(predictions[best_idx])
        final.events = events
        final.optional_events = []
        final.action_metadata = None

        return final

    @staticmethod
    def _choose_best_prediction(predictions: List[PolicyPrediction]) -> int:
        """Chooses the "best" prediction out of the given predictions.

        Note that this comparison is *not* symmetric if the priorities are allowed to
        coincide (i.e. if we cannot distinguish two predictions using 1.-4., then
        the first prediction is considered to be "better").

        Args:
          predictions: all predictions made by an ensemble of policies
        Returns:
          index of the best prediction
        """
        contains_no_form_predictions = all(
            not prediction.is_no_user_prediction for prediction in predictions
        )

        def scores(
            prediction: PolicyPrediction,
        ) -> Tuple[bool, bool, float, Tuple[int, ...]]:
            """Extracts scores ordered by importance where larger is better."""
            return (
                prediction.is_no_user_prediction,
                contains_no_form_predictions and prediction.is_end_to_end_prediction,
                prediction.max_confidence,
                prediction.policy_priority,  # Note: this is a tuple.
            )

        # grab the index of a prediction whose tuple of scores is >= all the
        # tuples of scores for any other prediction
        arg_max = max(
            list(range(len(predictions))), key=lambda idx: scores(predictions[idx])
        )
        return arg_max

    @staticmethod
    def _choose_events(
        predictions: List[PolicyPrediction],
        best_idx: int,
        tracker: DialogueStateTracker,
    ) -> List[Event]:
        """Chooses the events to be added to the final prediction.

        Args:
          predictions: all predictions made by an ensemble of policies
          best_idx: index of the prediction that is considered to be the "best" one
          tracker: dialogue state tracker holding the state of the conversation
        """
        if best_idx < 0 or best_idx >= len(predictions):
            raise ValueError(
                f"Expected given index to be pointing towards the best prediction "
                f"among the given predictions. Received index {best_idx} and a list "
                f"of predictions of length {len(predictions)}"
            )

        # combine all mandatory events in the given order
        events = [event for prediction in predictions for event in prediction.events]

        # append optional events from the "best prediction"
        best = predictions[best_idx]
        events += best.optional_events

        # if the sequence of events ends with a user utterance, append an event
        # defining the featurization
        if tracker.latest_action_name == ACTION_LISTEN_NAME:
            was_end_to_end_prediction = best.is_end_to_end_prediction
            if was_end_to_end_prediction:
                logger.debug("Made prediction using user text.")
            else:
                logger.debug("Made prediction without user text")
            event = DefinePrevUserUtteredFeaturization(was_end_to_end_prediction)
            events.append(event)
            logger.debug(f"Added `{event}` event.")
        return events
