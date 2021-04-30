import logging
from typing import Any, List, Dict, Text, Optional, Set, Tuple, TYPE_CHECKING

from rasa.shared.constants import DOCS_URL_RULES
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.io
from rasa.shared.core.events import (
    LoopInterrupted,
)
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.policy import SupportedData, PolicyPrediction
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    get_active_loop_name,
    is_prev_action_listen_in_state,
)
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.constants import (
    DEFAULT_CORE_FALLBACK_THRESHOLD,
    RULE_POLICY_PRIORITY,
)
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_SESSION_START,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_BACK_NAME,
)
from rasa.shared.core.domain import InvalidDomain, State, Domain
from rasa.shared.nlu.constants import ACTION_NAME, INTENT_NAME_KEY
import rasa.core.test
import rasa.core.training.training

from rasa.shared.nlu.state_machine.state_machine_state import (
    Response,
    StateMachineState,
    Transition,
)

from rasa.shared.nlu.state_machine.student_life_state_machine import (
    student_life_state_machine,
)

from rasa.core.actions.state_machine_action import StateMachineAction

if TYPE_CHECKING:
    from rasa.core.policies.ensemble import PolicyEnsemble

logger = logging.getLogger(__name__)

# These are Rasa Open Source default actions and overrule everything at any time.
DEFAULT_ACTION_MAPPINGS = {
    USER_INTENT_RESTART: ACTION_RESTART_NAME,
    USER_INTENT_BACK: ACTION_BACK_NAME,
    USER_INTENT_SESSION_START: ACTION_SESSION_START_NAME,
}

RULES = "rules"
RULES_FOR_LOOP_UNHAPPY_PATH = "rules_for_loop_unhappy_path"
RULES_NOT_IN_STORIES = "rules_not_in_stories"

LOOP_WAS_INTERRUPTED = "loop_was_interrupted"
DO_NOT_PREDICT_LOOP_ACTION = "do_not_predict_loop_action"

DEFAULT_RULES = "predicting default action with intent "
LOOP_RULES = "handling active loops and forms - "
LOOP_RULES_SEPARATOR = " - "


class InvalidRule(RasaException):
    """Exception that can be raised when rules are not valid."""

    def __init__(self, message: Text) -> None:
        super().__init__()
        self.message = message

    def __str__(self) -> Text:
        return self.message + (
            f"\nYou can find more information about the usage of "
            f"rules at {DOCS_URL_RULES}. "
        )


class StateMachinePolicy(MemoizationPolicy):
    """Policy which handles all the rules"""

    # rules use explicit json strings
    ENABLE_FEATURE_STRING_COMPRESSION = False

    # number of user inputs that is allowed in case rules are restricted
    ALLOWED_NUMBER_OF_USER_INPUTS = 1

    def _metadata(self) -> Dict[Text, Any]:
        return {
            "priority": self.priority,
            "lookup": self.lookup,
            "core_fallback_threshold": self._core_fallback_threshold,
            "core_fallback_action_name": self._fallback_action_name,
            "enable_fallback_prediction": self._enable_fallback_prediction,
        }

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "rule_policy.json"

    @staticmethod
    def supported_data() -> SupportedData:
        """The type of data supported by this policy.

        Returns:
            The data type supported by this policy (ML and rule data).
        """
        return SupportedData.ML_AND_RULE_DATA

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = RULE_POLICY_PRIORITY,
        lookup: Optional[Dict] = None,
        core_fallback_threshold: float = DEFAULT_CORE_FALLBACK_THRESHOLD,
        core_fallback_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME,
        enable_fallback_prediction: bool = True,
        restrict_rules: bool = True,
        check_for_contradictions: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a `StateMachinePolicy` object.

        Args:
            featurizer: `Featurizer` which is used to convert conversation states to
                features.
            priority: Priority of the policy which is used if multiple policies predict
                actions with the same confidence.
            lookup: Lookup table which is used to pick matching rules for a conversation
                state.
            core_fallback_threshold: Confidence of the prediction if no rule matched
                and de-facto threshold for a core fallback.
            core_fallback_action_name: Name of the action which should be predicted
                if no rule matched.
            enable_fallback_prediction: If `True` `core_fallback_action_name` is
                predicted in case no rule matched.
            restrict_rules: If `True` rules are restricted to contain a maximum of 1
                user message. This is used to avoid that users build a state machine
                using the rules.
            check_for_contradictions: Check for contradictions.
        """
        self._core_fallback_threshold = core_fallback_threshold
        self._fallback_action_name = core_fallback_action_name
        self._enable_fallback_prediction = enable_fallback_prediction
        self._restrict_rules = restrict_rules
        self._check_for_contradictions = check_for_contradictions

        self._rules_sources = None

        # max history is set to `None` in order to capture any lengths of rule stories
        super().__init__(
            featurizer=featurizer,
            priority=priority,
            max_history=None,
            lookup=lookup,
            **kwargs,
        )

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        """Trains the policy on given training trackers.

        Args:
            training_trackers: The list of the trackers.
            domain: The domain.
            interpreter: Interpreter which can be used by the polices for featurization.
        """
        pass

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> "PolicyPrediction":
        """Predicts the next action (see parent class for more information)."""
        prediction, _ = self._predict(tracker, domain)
        return prediction

    def _check_if_current_state_has_actions(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> bool:
        """
        Predict StateMachineAction if any are true:
        If any slots have been filled
        If any response conditions are met
        If any transitions conditions are met
        """

        # Get current state info
        state_machine_state: StateMachineState = student_life_state_machine  # TODO

        # Check if there are slots to fill
        if StateMachineAction._get_slot_values(
            slots=state_machine_state.slots, tracker=tracker
        ):
            return True

        # Check if any response conditions are met
        if StateMachineAction._get_response_action_names(
            responses=state_machine_state.responses, tracker=tracker
        ):
            return True

        # TODO: Check if any transitions conditions are met
        # if StateMachineAction._get_response_action_names(
        #     responses=state_machine_state.responses, tracker=tracker
        # ):
        #     return True

        return False

    def _predict(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Tuple[PolicyPrediction, Text]:
        action_name: Optional[str] = None

        if self._check_if_current_state_has_actions(tracker, domain):
            action_name = "action_state_machine_action"

        return (
            self._rule_prediction(
                self._prediction_result(action_name, tracker, domain),
                None,
                returning_from_unhappy_path=False,
                is_end_to_end_prediction=False,
            ),
            None,
        )

    def _rule_prediction(
        self,
        probabilities: List[float],
        prediction_source: Text,
        returning_from_unhappy_path: bool = False,
        is_end_to_end_prediction: bool = False,
        is_no_user_prediction: bool = False,
    ) -> PolicyPrediction:
        return PolicyPrediction(
            probabilities,
            self.__class__.__name__,
            self.priority,
            events=[LoopInterrupted(True)] if returning_from_unhappy_path else [],
            is_end_to_end_prediction=is_end_to_end_prediction,
            is_no_user_prediction=is_no_user_prediction,
            hide_rule_turn=(
                True
                if prediction_source in self.lookup.get(RULES_NOT_IN_STORIES, [])
                else False
            ),
        )

    def _default_predictions(self, domain: Domain) -> List[float]:
        result = super()._default_predictions(domain)

        if self._enable_fallback_prediction:
            result[
                domain.index_for_action(self._fallback_action_name)
            ] = self._core_fallback_threshold
        return result
