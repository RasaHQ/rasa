import logging
from typing import List, Dict, Text, Optional, Any, Set, TYPE_CHECKING

import re
from collections import defaultdict

from rasa.core.events import FormValidation
from rasa.core.domain import PREV_PREFIX, ACTIVE_FORM_PREFIX, Domain, InvalidDomain
from rasa.core.featurizers import TrackerFeaturizer
from rasa.core.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.policy import SupportedData
from rasa.core.trackers import DialogueStateTracker
from rasa.core.constants import (
    FORM_POLICY_PRIORITY,
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_SESSION_START,
)
from rasa.core.actions.action import (
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_BACK_NAME,
    ACTION_SESSION_START_NAME,
    RULE_SNIPPET_ACTION_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
)

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
        if not featurizer:
            # max history is set to `None` in order to capture lengths of rule stories
            featurizer = self._standard_featurizer()
            featurizer.max_history = None

        self._core_fallback_threshold = core_fallback_threshold
        self._fallback_action_name = core_fallback_action_name
        self._enable_fallback_prediction = enable_fallback_prediction

        super().__init__(featurizer=featurizer, priority=priority, lookup=lookup)

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

    def _create_feature_key(self, states: List[Dict]) -> Text:

        feature_str = ""
        for state in states:
            if state:
                feature_str += "|"
                for feature in state.keys():
                    feature_str += feature + " "
                feature_str = feature_str.strip()

        return feature_str

    @staticmethod
    def _get_active_form_name(state: Dict[Text, float]) -> Optional[Text]:
        # by construction there is only one active form
        return next(
            (
                state_name[len(ACTIVE_FORM_PREFIX) :]
                for state_name, prob in state.items()
                if ACTIVE_FORM_PREFIX in state_name
                and state_name != ACTIVE_FORM_PREFIX + "None"
                and prob > 0
            ),
            None,
        )

    @staticmethod
    def _prev_action_listen_in_state(state: Dict[Text, float]) -> bool:
        return any(
            PREV_PREFIX + ACTION_LISTEN_NAME in state_name and prob > 0
            for state_name, prob in state.items()
        )

    @staticmethod
    def _modified_states(
        states: List[Dict[Text, float]]
    ) -> List[Optional[Dict[Text, float]]]:
        """Modifies the states to create feature keys for form unhappy path conditions.

        Args:
            states: a representation of a tracker
                as a list of dictionaries containing features

        Returns:
            modified states
        """

        indicator = PREV_PREFIX + RULE_SNIPPET_ACTION_NAME
        state_only_with_action = {indicator: 1}
        # leave only last 2 dialogue turns to
        # - capture previous meaningful action before action_listen
        # - ignore previous intent
        if len(states) > 2 and states[-2] is not None:
            state_only_with_action = {
                state_name: prob
                for state_name, prob in states[-2].items()
                if PREV_PREFIX in state_name and prob > 0
            }

        # add `prev_...` to show that it should not be a first turn
        if indicator not in state_only_with_action and indicator not in states[-1]:
            return [{indicator: 1}, state_only_with_action, states[-1]]

        return [state_only_with_action, states[-1]]

    @staticmethod
    def _clean_feature_keys(lookup: Dict[Text, Text]) -> Dict[Text, Text]:
        # remove action_listens that were added after conditions
        updated_lookup = lookup.copy()
        for feature_key, action in lookup.items():
            # Delete rules if there is no prior action or if it would predict
            # the `...` action
            if PREV_PREFIX not in feature_key or action == RULE_SNIPPET_ACTION_NAME:
                del updated_lookup[feature_key]
            elif RULE_SNIPPET_ACTION_NAME in feature_key:
                # If the previous action is `...` -> remove any specific state
                # requirements for that state (anything can match this state)
                new_feature_key = re.sub(
                    rf".*{PREV_PREFIX}\.\.\.[^|]*", "", feature_key
                )

                if new_feature_key:
                    if new_feature_key.startswith("|"):
                        new_feature_key = new_feature_key[1:]
                    if new_feature_key.endswith("|"):
                        new_feature_key = new_feature_key[:-1]
                    updated_lookup[new_feature_key] = action

                del updated_lookup[feature_key]

        return updated_lookup

    def _create_form_unhappy_lookup_from_states(
        self,
        trackers_as_states: List[List[Dict]],
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
            if active_form:
                states = self._modified_states(states)
                feature_key = self._create_feature_key(states)

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
        training_trackers: List[DialogueStateTracker],
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
        form_unhappy_lookup = self._create_form_unhappy_lookup_from_states(
            trackers_as_states, trackers_as_actions
        )
        self.lookup[RULES_FOR_FORM_UNHAPPY_PATH] = self._clean_feature_keys(
            form_unhappy_lookup
        )

        # TODO use story_trackers and rule_trackers
        #  to check that stories don't contradict rules

        logger.debug(f"Memorized '{len(self.lookup[RULES])}' unique rules.")

    @staticmethod
    def _features_in_state(features: List[Text], state: Dict[Text, float]) -> bool:

        state_slots = defaultdict(set)
        for s in state.keys():
            if s.startswith("slot"):
                state_slots[s[: s.rfind("_")]].add(s)

        f_slots = defaultdict(set)
        for f in features:
            # TODO: this is a hack to make a rule know
            #  that slot or form should not be set;
            #  `_None` is added inside domain to indicate that
            #  the feature should not be present
            if f.endswith("_None"):
                if any(f[: f.rfind("_")] in key for key in state.keys()):
                    return False
            elif f not in state:
                return False
            elif f.startswith("slot"):
                f_slots[f[: f.rfind("_")]].add(f)

        for k, v in f_slots.items():
            if state_slots[k] != v:
                return False

        return True

    def _rule_is_good(
        self, rule_key: Text, turn_index: int, state: Dict[Text, float]
    ) -> bool:
        """Check if rule is satisfied with current state at turn."""

        # turn_index goes back in time
        rule_turns = list(reversed(rule_key.split("|")))

        return bool(
            # rule is shorter than current turn index
            turn_index >= len(rule_turns)
            # current rule and state turns are empty
            or (not rule_turns[turn_index] and not state)
            # check that current rule turn features are present in current state turn
            or (
                rule_turns[turn_index]
                and state
                and self._features_in_state(rule_turns[turn_index].split(), state)
            )
        )

    def _get_possible_keys(
        self, lookup: Dict[Text, Text], states: List[Dict[Text, float]]
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
            not tracker.latest_action_name == ACTION_LISTEN_NAME
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

        active_form_name = tracker.active_loop_name()
        active_form_rejected = tracker.active_loop.get("rejected")
        should_predict_form = (
            active_form_name
            and not active_form_rejected
            and tracker.latest_action_name != active_form_name
        )
        should_predict_listen = (
            active_form_name
            and not active_form_rejected
            and tracker.latest_action_name == active_form_name
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
            # TODO check that max is correct
            # if there are several rules,
            # it should mean that some rule is a subset of another rule
            best_rule_key = max(rule_keys, key=len)
            predicted_action_name = self.lookup[RULES].get(best_rule_key)

        active_form_name = tracker.active_loop_name()
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
                and ACTIVE_FORM_PREFIX + active_form_name not in best_rule_key
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
        interpreter: NaturalLanguageInterpreter = RegexInterpreter(),
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
