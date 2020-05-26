import logging
import typing
from typing import List, Dict, Text, Optional, Any
from tqdm import tqdm
import json
import re
from collections import defaultdict, deque

import rasa.utils.io

from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted, Form, SlotSet, ActionExecutionRejected
from rasa.core.featurizers import TrackerFeaturizer, MaxHistoryTrackerFeaturizer
from rasa.core.policies.form_policy import FormPolicy
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.common import is_logging_disabled
from rasa.core.constants import (
    MEMOIZATION_POLICY_PRIORITY,
    REQUESTED_SLOT,
    RULE_SNIPPET_ACTION_NAME,
)

from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.domain import PREV_PREFIX, ACTIVE_FORM_PREFIX, Domain, InvalidDomain
from rasa.core.events import FormValidation
from rasa.core.featurizers import TrackerFeaturizer
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.trackers import DialogueStateTracker
from rasa.core.constants import FORM_POLICY_PRIORITY

if typing.TYPE_CHECKING:
    from rasa.core.policies.ensemble import PolicyEnsemble


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
            if f not in state:
                return False

            if f.startswith("slot"):
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
            if "prev" not in key or self.lookup[key] == domain.index_for_action("..."):
                del updated_lookup[key]
            elif "..." in key:
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

    @staticmethod
    def _form_training_trackers(domain: Domain) -> List[DialogueStateTracker]:
        """Add a rule for every form which triggers the `FormAction` whenever we are
           in a form.

        Args:
            domain: The current domain.

        Returns:
            The additional training trackers.
        """
        return [
            RulePolicy._form_trigger_rule(form_name, domain)
            for form_name in domain.form_names
        ]

    @staticmethod
    def _form_trigger_rule(form_name: Text, domain: Domain) -> DialogueStateTracker:
        from rasa.core.training.generator import TrackerWithCachedStates

        return TrackerWithCachedStates.from_events(
            "bla",
            slots=domain.slots,
            evts=[
                # When we are in a form
                Form(form_name),
                SlotSet(REQUESTED_SLOT, "some value"),
                # We don't mind about previous conversation context
                ActionExecuted(RULE_SNIPPET_ACTION_NAME),
                ActionExecuted(ACTION_LISTEN_NAME),
                # Trigger form when we have an active form with that name
                ActionExecuted(form_name),
                ActionExecuted(ACTION_LISTEN_NAME),
            ],
            domain=domain,
        )

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

        states = [
            domain.get_active_states(tr)
            for tr in tracker.generate_all_prior_trackers_for_rules()
        ]
        states = deque(frozenset(s.items()) for s in states)
        bin_states = []
        for state in states:
            # copy state dict to preserve internal order of keys
            bin_state = dict(state)
            best_intent = None
            best_intent_prob = -1.0
            for state_name, prob in state:
                if state_name.startswith("intent_"):
                    if prob > best_intent_prob:
                        # finding the maximum confidence intent
                        if best_intent is not None:
                            # delete previous best intent
                            del bin_state[best_intent]
                        best_intent = state_name
                        best_intent_prob = prob
                    else:
                        # delete other intents
                        del bin_state[state_name]

            if best_intent is not None:
                # set the confidence of best intent to 1.0
                bin_state[best_intent] = 1.0

            bin_states.append(bin_state)
        states = bin_states

        logger.debug(f"Current tracker state {states}")

        possible_keys = set(self.lookup.keys())
        for i, state in enumerate(reversed(states)):
            possible_keys = set(
                filter(lambda _key: self._rule_is_good(_key, i, state), possible_keys)
            )

        active_form_name = tracker.active_form_name()
        if active_form_name and tracker.events[-1] == ActionExecutionRejected(
            active_form_name
        ):
            possible_keys = self._remove_keys_which_trigger_form_action_again(
                possible_keys, domain, active_form_name
            )

        if possible_keys:
            key = max(possible_keys, key=len)

            recalled = self.lookup.get(key)
            if recalled is not None:
                logger.debug(
                    f"There is a memorised next action '{domain.action_names[recalled]}'"
                )

                if self.USE_NLU_CONFIDENCE_AS_SCORE:
                    # the memoization will use the confidence of NLU on the latest
                    # user message to set the confidence of the action
                    score = tracker.latest_message.intent.get("confidence", 1.0)
                else:
                    score = 1.0

                result[recalled] = score
            else:
                logger.debug("There is no memorised next action")

        return result

    def _remove_keys_which_trigger_form_action_again(
        self, possible_keys: typing.Set[Text], domain: Domain, current_form_name: Text
    ) -> typing.Set[Text]:
        """Remove any matching rules which would predict the `FormAction`.

        This is used when the Form rejected its execution and we are entering an
        unhappy path.

        Args:
            possible_keys: Possible rule keys which match the current state.
            domain: The current domain.
            current_form_name: The currently active form.

        Returns:
            Possible keys without keys which predict the `FormAction`.
        """
        form_action_idx = domain.index_for_action(current_form_name)

        return {key for key in possible_keys if self.lookup[key] != form_action_idx}
