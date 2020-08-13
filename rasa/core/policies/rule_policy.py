import logging
from typing import List, Dict, Text, Optional, Any, Set, TYPE_CHECKING

import json

from rasa.core.events import FormValidation
from rasa.core.domain import (
    Domain,
    InvalidDomain,
    STATE,
)
from rasa.core.featurizers import TrackerFeaturizer
from rasa.core.interpreter import NaturalLanguageInterpreter
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.policy import SupportedData
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.generator import TrackerWithCachedStates
from rasa.core.constants import (
    FORM_POLICY_PRIORITY,
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_SESSION_START,
    FORM,
    SHOULD_NOT_BE_SET,
    PREVIOUS_ACTION,
)
from rasa.core.actions.action import (
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_BACK_NAME,
    ACTION_SESSION_START_NAME,
    RULE_SNIPPET_ACTION_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
)
from rasa.nlu.constants import ACTION_NAME

if TYPE_CHECKING:
    from rasa.core.policies.ensemble import PolicyEnsemble  # pytype: disable=pyi-error

logger = logging.getLogger(__name__)

# These are Rasa Open Source default actions and overrule everything at any time.
DEFAULT_ACTION_MAPPINGS = {
    USER_INTENT_RESTART: ACTION_RESTART_NAME,
    USER_INTENT_BACK: ACTION_BACK_NAME,
    USER_INTENT_SESSION_START: ACTION_SESSION_START_NAME,
}

RULES = "rules"
RULES_FOR_FORM_UNHAPPY_PATH = "rules_for_form_unhappy_path"
DO_NOT_VALIDATE_FORM = "do_not_validate_form"
DO_NOT_PREDICT_FORM_ACTION = "do_not_predict_form_action"


class RulePolicy(MemoizationPolicy):
    """Policy which handles all the rules"""

    ENABLE_FEATURE_STRING_COMPRESSION = False

    @staticmethod
    def supported_data() -> SupportedData:
        """The type of data supported by this policy.

        Returns:
            The data type supported by this policy (rule data).
        """
        return SupportedData.ML_AND_RULE_DATA

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = FORM_POLICY_PRIORITY,
        lookup: Optional[Dict] = None,
        core_fallback_threshold: float = 0.3,
        core_fallback_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME,
        enable_fallback_prediction: bool = True,
    ) -> None:
        """Create a `RulePolicy` object.

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
        """

        self._core_fallback_threshold = core_fallback_threshold
        self._fallback_action_name = core_fallback_action_name
        self._enable_fallback_prediction = enable_fallback_prediction

        # max history is set to `None` in order to capture any lengths of rule stories
        super().__init__(
            featurizer=featurizer, priority=priority, max_history=None, lookup=lookup
        )

    @classmethod
    def validate_against_domain(
        cls, ensemble: Optional["PolicyEnsemble"], domain: Optional[Domain]
    ) -> None:
        if ensemble is None:
            return

        rule_policy = next(
            (p for p in ensemble.policies if isinstance(p, RulePolicy)), None
        )
        if not rule_policy or not rule_policy._enable_fallback_prediction:
            return

        if (
            domain is None
            or rule_policy._fallback_action_name not in domain.action_names
        ):
            raise InvalidDomain(
                f"The fallback action '{rule_policy._fallback_action_name}' which was "
                f"configured for the {RulePolicy.__name__} must be present in the "
                f"domain."
            )

    @staticmethod
    def _is_rule_snippet_state(state: STATE) -> bool:
        prev_action_name = state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
        return prev_action_name == RULE_SNIPPET_ACTION_NAME

    def _create_feature_key(self, states: List[STATE]) -> Optional[Text]:
        new_states = []
        for state in reversed(states):
            if self._is_rule_snippet_state(state):
                # remove all states before `...`
                break
            new_states.insert(0, state)

        if not new_states or new_states == [{}]:
            return

        return json.dumps(new_states, sort_keys=True)

    @staticmethod
    def _get_active_form_name(state: STATE) -> Optional[Text]:
        return state.get(FORM, {}).get("name")

    @staticmethod
    def _prev_action_listen_in_state(state: STATE) -> bool:
        prev_action_name = state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
        return prev_action_name == ACTION_LISTEN_NAME

    @staticmethod
    def _modified_states(states: List[STATE]) -> List[STATE]:
        """Modifies the states to create feature keys for form unhappy path conditions.

        Args:
            states: a representation of a tracker
                as a list of dictionaries containing features

        Returns:
            modified states
        """

        # leave only last 2 dialogue turns to
        # - capture previous meaningful action before action_listen
        # - ignore previous intent
        if len(states) == 1 or not states[-2].get(PREVIOUS_ACTION):
            return [states[-1]]
        else:
            return [{PREVIOUS_ACTION: states[-2][PREVIOUS_ACTION]}, states[-1]]

    @staticmethod
    def _clean_feature_keys(lookup: Dict[Text, Text]) -> Dict[Text, Text]:
        # remove action_listens that were added after conditions
        updated_lookup = lookup.copy()
        for feature_key, action in lookup.items():
            # Delete rules if it would predict the `...` action
            if action == RULE_SNIPPET_ACTION_NAME:
                del updated_lookup[feature_key]

        return updated_lookup

    def _create_form_unhappy_lookup_from_states(
        self,
        trackers_as_states: List[List[STATE]],
        trackers_as_actions: List[List[Text]],
    ) -> Dict[Text, Text]:
        """Creates lookup dictionary from the tracker represented as states.

        Args:
            trackers_as_states: representation of the trackers as a list of states
            trackers_as_actions: representation of the trackers as a list of actions

        Returns:
            lookup dictionary
        """

        lookup = {}
        for states, actions in zip(trackers_as_states, trackers_as_actions):
            action = actions[0]
            active_form = self._get_active_form_name(states[-1])
            # even if there are two identical feature keys
            # their form will be the same
            # because of `active_form_...` feature
            if active_form and active_form != SHOULD_NOT_BE_SET:
                states = self._modified_states(states)
                feature_key = self._create_feature_key(states)
                if not feature_key:
                    continue

                # Since rule snippets and stories inside the form contain
                # only unhappy paths, notify the form that
                # it was predicted after an answer to a different question and
                # therefore it should not validate user input for requested slot
                if (
                    # form is predicted after action_listen in unhappy path,
                    # therefore no validation is needed
                    self._prev_action_listen_in_state(states[-1])
                    and action == active_form
                ):
                    lookup[feature_key] = DO_NOT_VALIDATE_FORM
                elif (
                    # some action other than action_listen and active_form
                    # is predicted in unhappy path,
                    # therefore active_form shouldn't be predicted by the rule
                    not self._prev_action_listen_in_state(states[-1])
                    and action not in {ACTION_LISTEN_NAME, active_form}
                ):
                    lookup[feature_key] = DO_NOT_PREDICT_FORM_ACTION
        return lookup

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:

        # only consider original trackers (no augmented ones)
        training_trackers = [
            t
            for t in training_trackers
            if not hasattr(t, "is_augmented") or not t.is_augmented
        ]
        # only use trackers from rule-based training data
        rule_trackers = [t for t in training_trackers if t.is_rule_tracker]
        (
            rule_trackers_as_states,
            rule_trackers_as_actions,
        ) = self.featurizer.training_states_and_actions(rule_trackers, domain)

        rules_lookup = self._create_lookup_from_states(
            rule_trackers_as_states, rule_trackers_as_actions
        )
        self.lookup[RULES] = self._clean_feature_keys(rules_lookup)

        story_trackers = [t for t in training_trackers if not t.is_rule_tracker]
        (
            story_trackers_as_states,
            story_trackers_as_actions,
        ) = self.featurizer.training_states_and_actions(story_trackers, domain)

        # use all trackers to find negative rules in unhappy paths
        trackers_as_states = rule_trackers_as_states + story_trackers_as_states
        trackers_as_actions = rule_trackers_as_actions + story_trackers_as_actions

        # negative rules are not anti-rules, they are auxiliary to actual rules
        self.lookup[
            RULES_FOR_FORM_UNHAPPY_PATH
        ] = self._create_form_unhappy_lookup_from_states(
            trackers_as_states, trackers_as_actions
        )

        # TODO use story_trackers and rule_trackers
        #  to check that stories don't contradict rules

        logger.debug(f"Memorized '{len(self.lookup[RULES])}' unique rules.")

    @staticmethod
    def _check_rule_state(rule_state: STATE, state: STATE) -> bool:

        for state_type, rule_sub_state in rule_state.items():
            sub_state = state.get(state_type, {})
            for key, value in rule_sub_state.items():
                if isinstance(value, list):
                    # json dumps and loads tuples as lists,
                    # so we need to convert them back
                    value = tuple(value)

                if (
                    # value should be set, therefore
                    # check whether it is the same as in the state
                    value
                    and value != SHOULD_NOT_BE_SET
                    and sub_state.get(key) != value
                ) or (
                    # value shouldn't be set, therefore
                    # it should be None or non existent in the state
                    value == SHOULD_NOT_BE_SET
                    and sub_state.get(key)
                ):
                    return False

        return True

    @staticmethod
    def _rule_key_to_state(rule_key: Text) -> List[STATE]:
        return json.loads(rule_key)

    def _rule_is_good(self, rule_key: Text, turn_index: int, state: STATE) -> bool:
        """Check if rule is satisfied with current state at turn."""

        # turn_index goes back in time
        reversed_rule_states = list(reversed(self._rule_key_to_state(rule_key)))

        return bool(
            # rule is shorter than current turn index
            turn_index >= len(reversed_rule_states)
            # current rule and state turns are empty
            or (not reversed_rule_states[turn_index] and not state)
            # check that current rule turn features are present in current state turn
            or (
                reversed_rule_states[turn_index]
                and state
                and self._check_rule_state(reversed_rule_states[turn_index], state)
            )
        )

    def _get_possible_keys(
        self, lookup: Dict[Text, Text], states: List[STATE]
    ) -> Set[Text]:
        possible_keys = set(lookup.keys())
        for i, state in enumerate(reversed(states)):
            # find rule keys that correspond to current state
            possible_keys = set(
                filter(lambda _key: self._rule_is_good(_key, i, state), possible_keys)
            )
        return possible_keys

    @staticmethod
    def _find_action_from_default_actions(
        tracker: DialogueStateTracker,
    ) -> Optional[Text]:
        if (
            not tracker.latest_action.get(ACTION_NAME) == ACTION_LISTEN_NAME
            or not tracker.latest_message
        ):
            return None

        default_action_name = DEFAULT_ACTION_MAPPINGS.get(
            tracker.latest_message.intent.get("name")
        )

        if default_action_name:
            logger.debug(f"Predicted default action '{default_action_name}'.")

        return default_action_name

    @staticmethod
    def _find_action_from_form_happy_path(
        tracker: DialogueStateTracker,
    ) -> Optional[Text]:

        active_form_name = tracker.active_form_name()
        active_form_rejected = tracker.active_loop.get("rejected")
        should_predict_form = (
            active_form_name
            and not active_form_rejected
            and tracker.latest_action.get(ACTION_NAME) != active_form_name
        )
        should_predict_listen = (
            active_form_name
            and not active_form_rejected
            and tracker.latest_action.get(ACTION_NAME) == active_form_name
        )

        if should_predict_form:
            logger.debug(f"Predicted form '{active_form_name}'.")
            return active_form_name

        # predict `action_listen` if form action was run successfully
        if should_predict_listen:
            logger.debug(
                f"Predicted '{ACTION_LISTEN_NAME}' after form '{active_form_name}'."
            )
            return ACTION_LISTEN_NAME

    def _find_action_from_rules(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Optional[Text]:
        tracker_as_states = self.featurizer.prediction_states([tracker], domain)
        states = tracker_as_states[0]

        logger.debug(f"Current tracker state: {states}")

        rule_keys = self._get_possible_keys(self.lookup[RULES], states)
        predicted_action_name = None
        best_rule_key = ""
        if rule_keys:
            # if there are several rules,
            # it should mean that some rule is a subset of another rule
            # therefore we pick a rule of maximum length
            best_rule_key = max(rule_keys, key=len)
            predicted_action_name = self.lookup[RULES].get(best_rule_key)

        active_form_name = tracker.active_form_name()
        if active_form_name:
            # find rules for unhappy path of the form
            form_unhappy_keys = self._get_possible_keys(
                self.lookup[RULES_FOR_FORM_UNHAPPY_PATH], states
            )
            # there could be several unhappy path conditions
            unhappy_path_conditions = [
                self.lookup[RULES_FOR_FORM_UNHAPPY_PATH].get(key)
                for key in form_unhappy_keys
            ]

            # Check if a rule that predicted action_listen
            # was applied inside the form.
            # Rules might not explicitly switch back to the `Form`.
            # Hence, we have to take care of that.
            predicted_listen_from_general_rule = (
                predicted_action_name == ACTION_LISTEN_NAME
                and not self._get_active_form_name(
                    self._rule_key_to_state(best_rule_key)[-1]
                )
            )
            if predicted_listen_from_general_rule:
                if DO_NOT_PREDICT_FORM_ACTION not in unhappy_path_conditions:
                    # negative rules don't contain a key that corresponds to
                    # the fact that active_form shouldn't be predicted
                    logger.debug(
                        f"Predicted form '{active_form_name}' by overwriting "
                        f"'{ACTION_LISTEN_NAME}' predicted by general rule."
                    )
                    return active_form_name

                # do not predict anything
                predicted_action_name = None

            if DO_NOT_VALIDATE_FORM in unhappy_path_conditions:
                logger.debug("Added `FormValidation(False)` event.")
                tracker.update(FormValidation(False))

        if predicted_action_name is not None:
            logger.debug(
                f"There is a rule for the next action '{predicted_action_name}'."
            )
        else:
            logger.debug("There is no applicable rule.")

        return predicted_action_name

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> List[float]:

        result = self._default_predictions(domain)

        # Rasa Open Source default actions overrule anything. If users want to achieve
        # the same, they need to write a rule or make sure that their form rejects
        # accordingly.
        default_action_name = self._find_action_from_default_actions(tracker)
        if default_action_name:
            return self._prediction_result(default_action_name, tracker, domain)

        # A form has priority over any other rule.
        # The rules or any other prediction will be applied only if a form was rejected.
        # If we are in a form, and the form didn't run previously or rejected, we can
        # simply force predict the form.
        form_happy_path_action_name = self._find_action_from_form_happy_path(tracker)
        if form_happy_path_action_name:
            return self._prediction_result(form_happy_path_action_name, tracker, domain)

        rules_action_name = self._find_action_from_rules(tracker, domain)
        if rules_action_name:
            return self._prediction_result(rules_action_name, tracker, domain)

        return result

    def _default_predictions(self, domain: Domain) -> List[float]:
        result = super()._default_predictions(domain)

        if self._enable_fallback_prediction:
            result[
                domain.index_for_action(self._fallback_action_name)
            ] = self._core_fallback_threshold
        return result
