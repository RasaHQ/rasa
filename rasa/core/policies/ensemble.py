"""Notes.

NOTE: *not* covered here / other changes required
-  `validate` contains checks that where previously done during intialisation of
    the ensemble
    --> trigger in graph validation
    --> see: `validate` functions in ensembles
-  `set_rule_data` was done during intialisation and as last step of train
    --> e.g. persist rule_only_data separately during RulePolicy persist and
        create a graph component to the prediction graph that loads this specific
        resource during prediction and has access to all policies and can modify those
        graph components after they've been loaded, or maybe add a component that
        after training loads the rule data and adds that resource to all policies
        by persisting it to each policy resource (?) (FIXME)
    --> see: `propagate_specification_of_rule_only_data` below
-  final prediction requires access to domain
   --> add domain provider in predict graph
-  final prediction needs to add events to the tracker if and only if the sequence of
   events seen so far ends with a `UserUttered` event
   --> add tracker provider to predict graph
- specific unit tests related to action unlikely intent which basically test whether the
  *semantic* / idea behind the simple ensemble makes sense
   --> should replace this with some test in unexpecTEDintentPolicy which test that
       the probability, priority, ... whatever is chosen such that the tests that were
       previously here would pass (there is no real need to actually use the ensemble
       in these tests if no one decides to change the meaning of the simple policy
       instead of parametrising it's behavior)
NOTE: differences to previous version
-  previously there was a comment there saying
    # find rejected action before running the policies
    # because some of them might add events
   this seems to be outdated because the policy prediction contains the event lists
   and because I could not find policies adding events while they predict
   --> we do *not* move this to a component that is executed before the policies
   but leave this snippet in the ensemble's predict
-  refactored the way the "best prediction" is determined because that decision
   process was a bit scattered and in-explicit and changed PolicyPrediction such that
   priority is an int not a tuple (?!)
- `is_not_in_training_data` has been removed
   -->  can be replaced by an check that determines whether the policy is
        some kind of memoization policy *inside the tests* (via its name...)
NOTE/FIXME: questions
-  Where are the events from the (winning) predictions applied? We only add things to
   the final `PolicyPrediction` here.
"""

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


def propagate_specification_of_rule_only_data(ensemble: List[Policy]) -> None:
    """Extracts information on rule only data and propagates it to all policies.

    TODO: remove this once there's a corresponding new graph component taking
    care of this
    """
    rule_policy = next(
        (policy for policy in ensemble if isinstance(policy, RulePolicy)), None
    )
    if rule_policy:
        rule_only_data = rule_policy.get_rule_only_data()
    else:
        rule_only_data = dict()
    for policy in ensemble:
        policy.set_shared_policy_states(rule_only_data=rule_only_data)


def validate_warn_if_rule_based_data_unused_or_missing(
    ensemble: List[Policy], training_trackers: List[DialogueStateTracker]
) -> None:
    """Emit `UserWarning`s about missing or unused rule-based data.

    TODO: remove this once there's a corresponding check in `RulePolicy`

    Args:
        ensemble: a list of policies
        training_trackers: trackers to be used for training
    """
    consuming_rule_data = any(
        policy.supported_data()
        in [SupportedData.RULE_DATA, SupportedData.ML_AND_RULE_DATA]
        for policy in ensemble
    )
    contains_rule_tracker = any(
        tracker.is_rule_tracker for tracker in training_trackers
    )

    if consuming_rule_data and not contains_rule_tracker:
        rasa.shared.utils.io.raise_warning(
            f"Found a rule-based policy in your pipeline but "
            f"no rule-based training data. Please add rule-based "
            f"stories to your training data or "
            f"remove the rule-based policy (`{RulePolicy.__name__}`) from your "
            f"your pipeline.",
            docs=DOCS_URL_RULES,
        )
    elif not consuming_rule_data and contains_rule_tracker:
        rasa.shared.utils.io.raise_warning(
            f"Found rule-based training data but no policy supporting rule-based "
            f"data. Please add `{RulePolicy.__name__}` or another rule-supporting "
            f"policy to the `policies` section in `{DEFAULT_CONFIG_PATH}`.",
            docs=DOCS_URL_RULES,
        )


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

    @classmethod
    def validate_ensemble_valid(cls, ensemble: List[Policy], domain: Domain) -> None:
        """Checks that predictions of the given policy ensemble can be used as input.

        # TODO: replace the ensemble (List[Policy]) with a list of policy classes and
        # their configurations
        # TODO: add tests once all policies have been migrated

        Args:
          ensemble: a list of policies
          domain: the common domain

        Raises:
          `InvalidPolicyEnsembleConfig`: if this ensemble cannot be applied to
            the given ensemble of policies
        """
        cls.warn_if_rule_policy_not_contained(ensemble=ensemble)
        cls.assert_compatibility_with_domain(ensemble=ensemble, domain=domain)

    @staticmethod
    def warn_if_rule_policy_not_contained(ensemble: List[Policy]) -> None:
        """Checks that a rule policy is present.

        # TODO: replace the ensemble (List[Policy]) with a list of policy classes and
        # their configurations
        # TODO: add tests once all policies have been migrated

        Args:
          ensemble: list of policies
        """
        if not any(isinstance(policy, RulePolicy) for policy in ensemble):
            rasa.shared.utils.io.raise_warning(
                f"'{RulePolicy.__name__}' is not included in the model's "
                f"policy configuration. Default intents such as "
                f"'{USER_INTENT_RESTART}' and '{USER_INTENT_BACK}' will "
                f"not trigger actions '{ACTION_RESTART_NAME}' and "
                f"'{ACTION_BACK_NAME}'.",
                docs=DOCS_URL_DEFAULT_ACTIONS,
            )

    @staticmethod
    def assert_compatibility_with_domain(
        ensemble: List[Policy], domain: Optional[Domain]
    ) -> None:
        """Check for elements that only work with certain policy/domain combinations.

        # TODO: replace the ensemble (List[Policy]) with a list of policy classes and
        # their configurations
        # TODO: add tests once all policies have been migrated
        # TODO: make sure this check is also called if there is just a single policy
        # and *no* ensemble node in the graph

        Args:
            ensemble: list of policies
            domain: a domain

        Raises:
            `InvalidDomain` exception if the given domain is incompatible with the
            given ensemble
        """
        # TODO: adapt validate_against_domain to work with list of policies/configs
        # RulePolicy.validate_against_domain(ensemble, domain)

        contains_rule_policy = any(
            isinstance(policy, RulePolicy) for policy in ensemble
        )
        if domain.form_names and not contains_rule_policy:
            raise InvalidDomain(
                "You have defined a form action, but have not added the "
                f"'{RulePolicy.__name__}' to your policy ensemble. Either "
                f"remove all forms from your domain or add the '{RulePolicy.__name__}' "
                f"to your policy configuration."
            )

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

    def validate(self, ensemble: List[Policy], domain: Domain) -> None:
        """Checks whether this ensemble can be applied to the given policies.

        Args:
          ensemble: a list of policies
          domain: the common domain
        Raises:
          `InvalidPolicyEnsembleConfig`: if this ensemble cannot be applied to
            the given ensemble of policies
        """
        super().validate(ensemble=ensemble, domain=domain)
        # TODO: shouldn't we raise in case priorities are not unique since this might
        # lead to unexpected results in edge cases (swap order in your config)
        self.warn_if_priorities_not_unique(ensemble=ensemble)
        # TODO: shouldn't we assert that there's at most one rule policy here?

    @staticmethod
    def warn_if_priorities_not_unique(ensemble: List[Policy]) -> None:
        """Checks for duplicate policy priorities.

        Only raises a warning if two policies have the same priority.

        Args:
          ensemble: list of policies

        Raises:
           nothing
        """
        priority_dict: Dict[int, List[Text]] = dict()
        for p in ensemble:
            priority_dict.setdefault(p.priority, []).append(type(p).__name__)
        for k, v in priority_dict.items():
            if len(v) > 1:
                rasa.shared.utils.io.raise_warning(
                    f"Found policies {v} with same priority {k} "
                    f"in PolicyEnsemble. When personalizing "
                    f"priorities, be sure to give all policies "
                    f"different priorities.",
                    docs=DOCS_URL_POLICIES,
                )

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
        # (TODO: what is action meta data and why was it ignored?)
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

        def scores(prediction: PolicyPrediction) -> Tuple[bool, bool, float, int]:
            """Extracts scores ordered by importance where larger is better."""
            return (
                prediction.is_no_user_prediction,
                contains_no_form_predictions and prediction.is_end_to_end_prediction,
                prediction.max_confidence,
                prediction.policy_priority[0],  # because this is a tuple.
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
