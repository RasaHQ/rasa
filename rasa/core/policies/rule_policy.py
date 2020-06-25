import logging
import typing
from typing import List, Dict, Text, Optional, Any

import re
from collections import defaultdict, deque

from rasa.core.events import FormValidation
from rasa.core.domain import Domain
from rasa.core.featurizers import TrackerFeaturizer
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.trackers import DialogueStateTracker
from rasa.core.constants import FORM_POLICY_PRIORITY, RULE_SNIPPET_ACTION_NAME
from rasa.core.actions.action import ACTION_LISTEN_NAME

logger = logging.getLogger(__name__)


class RulePolicy(MemoizationPolicy):
    """Policy which handles all the rules"""

    ENABLE_FEATURE_STRING_COMPRESSION = False

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = FORM_POLICY_PRIORITY,
        lookup: Optional[Dict] = None,
    ) -> None:

        # max history is set to None in order to capture lengths of rule stories
        super().__init__(
            featurizer=featurizer, priority=priority, max_history=None, lookup=lookup
        )

    def _create_feature_key(self, states: List[Dict]):

        feature_str = ""
        for state in states:
            if state:
                feature_str += "|"
                for feature in state.keys():
                    feature_str += feature + " "
                feature_str = feature_str.strip()

        return feature_str

    @staticmethod
    def _features_in_state(fs, state):

        state_slots = defaultdict(set)
        for s in state.keys():
            if s.startswith("slot"):
                state_slots[s[: s.rfind("_")]].add(s)

        f_slots = defaultdict(set)
        for f in fs:
            if f.endswith("_None"):
                # TODO: this is a hack to make a rule know
                #  that slot or form should not be set
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

    def _rule_is_good(self, key, i, state):
        return (
            i >= len(key.split("|"))
            or (not list(reversed(key.split("|")))[i] and not state)
            or (
                list(reversed(key.split("|")))[i]
                and state
                and self._features_in_state(
                    list(reversed(key.split("|")))[i].split(), state
                )
            )
        )

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
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
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        """Predicts the next action the bot should take after seeing the tracker.

        Returns the list of probabilities for the next actions.
        If memorized action was found returns 1 for its index,
        else returns 0 for all actions.
        """
        result = self._default_predictions(domain)

        if not self.is_enabled:
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
            logger.debug(f"Predicted '{active_form_name}'")
            result[domain.index_for_action(active_form_name)] = 1
            return result
        # predict action_listen if form action was run successfully
        elif should_predict_listen:
            logger.debug(f"Predicted '{ACTION_LISTEN_NAME}' after form '{active_form_name}'.")
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
