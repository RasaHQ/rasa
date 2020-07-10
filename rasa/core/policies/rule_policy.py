import logging
from typing import List, Dict, Text, Optional, Any

import re
from collections import defaultdict

from rasa.core.events import FormValidation
from rasa.core.domain import Domain
from rasa.core.featurizers import TrackerFeaturizer
from rasa.core.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.policy import SupportedData
from rasa.core.trackers import DialogueStateTracker
from rasa.core.constants import (
    FORM_POLICY_PRIORITY,
    RULE_SNIPPET_ACTION_NAME,
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_SESSION_START,
)
from rasa.core.actions.action import (
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_BACK_NAME,
    ACTION_SESSION_START_NAME,
)

logger = logging.getLogger(__name__)

# These are Rasa Open Source default actions and overrule everything at any time.
DEFAULT_ACTION_MAPPINGS = {
    USER_INTENT_RESTART: ACTION_RESTART_NAME,
    USER_INTENT_BACK: ACTION_BACK_NAME,
    USER_INTENT_SESSION_START: ACTION_SESSION_START_NAME,
}


class RulePolicy(MemoizationPolicy):
    """Policy which handles all the rules"""

    ENABLE_FEATURE_STRING_COMPRESSION = False

    @staticmethod
    def supported_data() -> SupportedData:
        """The type of data supported by this policy.

        Returns:
            The data type supported by this policy (rule data).
        """
        return SupportedData.RULE_DATA

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = FORM_POLICY_PRIORITY,
        lookup: Optional[Dict] = None,
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

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        """Trains the policy on given training trackers."""
        self.lookup = {}

        # only use trackers from rule-based training data
        training_trackers = [t for t in training_trackers if t.is_rule_tracker]

        # only consider original trackers (no augmented ones)
        training_trackers = [
            t
            for t in training_trackers
            if not hasattr(t, "is_augmented") or not t.is_augmented
        ]
        (
            trackers_as_states,
            trackers_as_actions,
        ) = self.featurizer.training_states_and_actions(training_trackers, domain)

        self._add_states_to_lookup(trackers_as_states, trackers_as_actions, domain)

        # remove action_listens that were added after conditions
        updated_lookup = self.lookup.copy()
        for key in self.lookup.keys():
            # Delete rules if there is no prior action or if it would predict
            # the `...` action
            if "prev" not in key or self.lookup[key] == domain.index_for_action(
                RULE_SNIPPET_ACTION_NAME
            ):
                del updated_lookup[key]
            elif RULE_SNIPPET_ACTION_NAME in key:
                # If the previous action is `...` -> remove any specific state
                # requirements for that state (anything can match this state)
                new_key = re.sub(r".*prev_\.\.\.[^|]*", "", key)

                if new_key:
                    if new_key.startswith("|"):
                        new_key = new_key[1:]
                    if new_key.endswith("|"):
                        new_key = new_key[:-1]
                    updated_lookup[new_key] = self.lookup[key]

                del updated_lookup[key]

        self.lookup = updated_lookup
        logger.debug("Memorized {} unique examples.".format(len(self.lookup)))

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
        # the same, they need to a rule or make sure that their form rejects
        # accordingly.
        rasa_default_action_name = _should_run_rasa_default_action(tracker)
        if rasa_default_action_name:
            result[domain.index_for_action(rasa_default_action_name)] = 1
            return result

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

        # A form has priority over any other rule.
        # The rules or any other prediction will be applied only if a form was rejected.
        # If we are in a form, and the form didn't run previously or rejected, we can
        # simply force predict the form.
        if should_predict_form:
            logger.debug(f"Predicted form '{active_form_name}'.")
            result[domain.index_for_action(active_form_name)] = 1
            return result

        # predict `action_listen` if form action was run successfully
        if should_predict_listen:
            logger.debug(
                f"Predicted '{ACTION_LISTEN_NAME}' after form '{active_form_name}'."
            )
            result[domain.index_for_action(ACTION_LISTEN_NAME)] = 1
            return result

        possible_keys = set(self.lookup.keys())

        tracker_as_states = self.featurizer.prediction_states([tracker], domain)
        states = tracker_as_states[0]

        logger.debug(f"Current tracker state: {states}")

        for i, state in enumerate(reversed(states)):
            possible_keys = set(
                filter(lambda _key: self._rule_is_good(_key, i, state), possible_keys)
            )

        if possible_keys:
            # TODO rethink that
            key = max(possible_keys, key=len)

            recalled = self.lookup.get(key)

            if active_form_name:
                # Check if a rule that predicted action_listen
                # was applied inside the form.
                # Rules might not explicitly switch back to the `Form`.
                # Hence, we have to take care of that.
                predicted_listen_from_general_rule = recalled is None or (
                    domain.action_names[recalled] == ACTION_LISTEN_NAME
                    and f"active_form_{active_form_name}" not in key
                )
                if predicted_listen_from_general_rule:
                    logger.debug(f"Predicted form '{active_form_name}'.")
                    result[domain.index_for_action(active_form_name)] = 1
                    return result

                # Since rule snippets inside the form contain only unhappy paths,
                # notify the form that
                # it was predicted after an answer to a different question and
                # therefore it should not validate user input for requested slot
                predicted_form_from_form_rule = (
                    domain.action_names[recalled] == active_form_name
                    and f"active_form_{active_form_name}" in key
                )
                if predicted_form_from_form_rule:
                    logger.debug("Added `FormValidation(False)` event.")
                    tracker.update(FormValidation(False))

            if recalled is not None:
                logger.debug(
                    f"There is a rule for next action "
                    f"'{domain.action_names[recalled]}'."
                )

                result[recalled] = 1
            else:
                logger.debug("There is no applicable rule.")

        return result


def _should_run_rasa_default_action(tracker: DialogueStateTracker) -> Optional[Text]:
    if (
        not tracker.latest_action_name == ACTION_LISTEN_NAME
        or not tracker.latest_message
    ):
        return None

    return DEFAULT_ACTION_MAPPINGS.get(tracker.latest_message.intent.get("name"))
