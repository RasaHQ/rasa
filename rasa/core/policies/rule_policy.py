import logging
from typing import List, Dict, Text, Optional, Any

import re
from collections import defaultdict

from rasa.core.events import FormValidation
from rasa.core.domain import PREV_PREFIX, ACTIVE_FORM_PREFIX, Domain
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
)

logger = logging.getLogger(__name__)

# These are Rasa Open Source default actions and overrule everything at any time.
DEFAULT_ACTION_MAPPINGS = {
    USER_INTENT_RESTART: ACTION_RESTART_NAME,
    USER_INTENT_BACK: ACTION_BACK_NAME,
    USER_INTENT_SESSION_START: ACTION_SESSION_START_NAME,
}

RULES = "rules"
NEGATIVE_RULES = "negative_rules"
NO_VALIDATION = "no_validation"
NO_ACTIVE_FORM = "no_active_form"


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
        negative_lookup: Optional[Dict] = None,
    ) -> None:
        if not featurizer:
            # max history is set to `None` in order to capture lengths of rule stories
            featurizer = self._standard_featurizer()
            featurizer.max_history = None

        super().__init__(featurizer=featurizer, priority=priority, lookup=lookup)

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
        found_forms = [
            state_name[len(ACTIVE_FORM_PREFIX) :]
            for state_name, prob in state.items()
            if ACTIVE_FORM_PREFIX in state_name
            and state_name != ACTIVE_FORM_PREFIX + "None"
            and prob > 0
        ]
        # by construction there is only one active form
        return found_forms[0] if found_forms else None

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
        """Modify the states to
            - capture previous meaningful action before action_listen
            - ignore previous intent
        """
        if states[0] is None:
            action_before_listen = None
        else:
            action_before_listen = {
                state_name: prob
                for state_name, prob in states[0].items()
                if PREV_PREFIX in state_name and prob > 0
            }
        # add `prev_...` to show that it should not be a first turn
        indicator = PREV_PREFIX + RULE_SNIPPET_ACTION_NAME
        if indicator not in action_before_listen.keys():
            return [{indicator: 1}, action_before_listen, states[-1]]
        return [action_before_listen, states[-1]]

    @staticmethod
    def _clean_feature_keys(lookup: Optional[Dict]) -> Optional[Dict]:
        # remove action_listens that were added after conditions
        updated_lookup = lookup.copy()
        for key in lookup.keys():
            # Delete rules if there is no prior action or if it would predict
            # the `...` action
            if PREV_PREFIX not in key or lookup[key] == RULE_SNIPPET_ACTION_NAME:
                del updated_lookup[key]
            elif RULE_SNIPPET_ACTION_NAME in key:
                # If the previous action is `...` -> remove any specific state
                # requirements for that state (anything can match this state)
                new_key = re.sub(rf".*{PREV_PREFIX}\.\.\.[^|]*", "", key)

                if new_key:
                    if new_key.startswith("|"):
                        new_key = new_key[1:]
                    if new_key.endswith("|"):
                        new_key = new_key[:-1]
                    updated_lookup[new_key] = lookup[key]

                del updated_lookup[key]

        return updated_lookup

    def _create_negative_lookup_from_states(
        self,
        trackers_as_states: List[List[Dict]],
        trackers_as_actions: List[List[Text]],
    ) -> Dict[Text, Text]:
        negative_lookup = {}
        for states, actions in zip(trackers_as_states, trackers_as_actions):
            active_form = self._get_active_form_name(states[-1])
            # even if there are two identical feature keys
            # their form will be the same
            # because of `active_form_...` feature
            if active_form:
                # leave only last 2 dialogue turns
                states = self._modified_states(states[-2:])
                feature_key = self._create_feature_key(states)

                if (
                    # form is predicted after action_listen in unhappy path,
                    # therefore no validation is needed
                    self._prev_action_listen_in_state(states[-1])
                    and actions[0] == active_form
                ):
                    negative_lookup[feature_key] = NO_VALIDATION
                elif (
                    # some action other than action_listen and active_form
                    # is predicted in unhappy path,
                    # therefore active_form shouldn't be predicted by the rule
                    not self._prev_action_listen_in_state(states[-1])
                    and actions[0] not in {ACTION_LISTEN_NAME, active_form}
                ):
                    negative_lookup[feature_key] = NO_ACTIVE_FORM
        return negative_lookup

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        """Trains the policy on given training trackers."""

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

        rule_lookup = self._create_lookup_from_states(
            rule_trackers_as_states, rule_trackers_as_actions
        )

        self.lookup[RULES] = self._clean_feature_keys(rule_lookup)

        story_trackers = [t for t in training_trackers if not t.is_rule_tracker]
        (
            story_trackers_as_states,
            story_trackers_as_actions,
        ) = self.featurizer.training_states_and_actions(story_trackers, domain)

        # use all trackers to find negative rules in unhappy paths
        trackers_as_states = rule_trackers_as_states + story_trackers_as_states
        trackers_as_actions = rule_trackers_as_actions + story_trackers_as_actions

        # negative rules are not anti-rules, they are auxiliary to actual rules
        negative_lookup = self._create_negative_lookup_from_states(
            trackers_as_states, trackers_as_actions
        )
        self.lookup[NEGATIVE_RULES] = self._clean_feature_keys(negative_lookup)

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

    @staticmethod
    def _predict_default_action(tracker: DialogueStateTracker) -> Optional[Text]:
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
    def _predict_form_action(tracker: DialogueStateTracker) -> Optional[Text]:

        active_form_name = tracker.active_form_name()
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

    def _predict_rule_action(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Optional[Text]:
        tracker_as_states = self.featurizer.prediction_states([tracker], domain)
        states = tracker_as_states[0]

        logger.debug(f"Current tracker state: {states}")

        possible_keys = set(self.lookup[RULES].keys())
        negative_keys = set(self.lookup[NEGATIVE_RULES].keys())
        for i, state in enumerate(reversed(states)):
            # find rule keys that correspond to current state
            possible_keys = set(
                filter(lambda _key: self._rule_is_good(_key, i, state), possible_keys)
            )
            negative_keys = set(
                filter(lambda _key: self._rule_is_good(_key, i, state), negative_keys)
            )

        recalled = None
        key = ""
        if possible_keys:
            # TODO check that max is correct
            # if there are several rules,
            # it should mean that some rule is a subset of another rule
            key = max(possible_keys, key=len)
            recalled = self.lookup[RULES].get(key)

        # there could be several negative rules
        negative_recalled = [
            self.lookup[NEGATIVE_RULES].get(key) for key in negative_keys
        ]

        active_form_name = tracker.active_form_name()
        if active_form_name:
            # Check if a rule that predicted action_listen
            # was applied inside the form.
            # Rules might not explicitly switch back to the `Form`.
            # Hence, we have to take care of that.
            predicted_listen_from_general_rule = (
                recalled == ACTION_LISTEN_NAME
                and ACTIVE_FORM_PREFIX + active_form_name not in key
            )
            if predicted_listen_from_general_rule:
                if NO_ACTIVE_FORM not in negative_recalled:
                    # negative rules don't contain a key that corresponds to
                    # the fact that active_form shouldn't be predicted
                    logger.debug(
                        f"Predicted form '{active_form_name}' by overwriting "
                        f"'{ACTION_LISTEN_NAME}' predicted by general rule."
                    )
                    return active_form_name

                # do not predict anything
                recalled = None

            # Since rule snippets and stories inside the form contain
            # only unhappy paths, notify the form that
            # it was predicted after an answer to a different question and
            # therefore it should not validate user input for requested slot
            if NO_VALIDATION in negative_recalled:
                logger.debug("Added `FormValidation(False)` event.")
                tracker.update(FormValidation(False))

        if recalled is not None:
            logger.debug(f"There is a rule for the next action '{recalled}'.")
        else:
            logger.debug("There is no applicable rule.")

        return recalled

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter = RegexInterpreter(),
        **kwargs: Any,
    ) -> List[float]:
        """Predicts the next action the bot should take after seeing the tracker.

        Returns the list of probabilities for the next actions.
        If memorized action was found returns 1 for its index,
        else returns 0 for all actions.
        """
        result = self._default_predictions(domain)

        if not self.is_enabled:
            return result

        # Rasa Open Source default actions overrule anything. If users want to achieve
        # the same, they need to write a rule or make sure that their form rejects
        # accordingly.
        predicted_default_action_name = self._predict_default_action(tracker)
        if predicted_default_action_name:
            return self._prediction_result(
                predicted_default_action_name, tracker, domain
            )

        # A form has priority over any other rule.
        # The rules or any other prediction will be applied only if a form was rejected.
        # If we are in a form, and the form didn't run previously or rejected, we can
        # simply force predict the form.
        predicted_form_action_name = self._predict_form_action(tracker)
        if predicted_form_action_name:
            return self._prediction_result(predicted_form_action_name, tracker, domain)

        predicted_rule_action_name = self._predict_rule_action(tracker, domain)
        if predicted_rule_action_name:
            return self._prediction_result(predicted_rule_action_name, tracker, domain)

        return result
