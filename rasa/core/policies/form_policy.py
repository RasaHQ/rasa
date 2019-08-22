import logging
import typing
from typing import List, Optional, Dict, Text, Optional

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


class FormPolicy(MemoizationPolicy):
    """Policy which handles prediction of Forms"""

    ENABLE_FEATURE_STRING_COMPRESSION = True

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = FORM_POLICY_PRIORITY,
        lookup: Optional[Dict] = None,
    ) -> None:

        # max history is set to 2 in order to capture
        # previous meaningful action before action listen
        super(FormPolicy, self).__init__(
            featurizer=featurizer, priority=priority, max_history=2, lookup=lookup
        )

    @classmethod
    def validate_against_domain(
        cls, ensemble: Optional["PolicyEnsemble"], domain: Optional[Domain]
    ) -> None:
        if not domain:
            return

        has_form_policy = ensemble is not None and any(
            isinstance(p, cls) for p in ensemble.policies
        )
        if domain.form_names and not has_form_policy:
            raise InvalidDomain(
                "You have defined a form action, but haven't added the "
                "FormPolicy to your policy ensemble. Either remove all "
                "forms from your domain or exclude the FormPolicy from your "
                "policy configuration."
            )

    @staticmethod
    def _get_active_form_name(state):
        found_forms = [
            state_name[len(ACTIVE_FORM_PREFIX) :]
            for state_name, prob in state.items()
            if ACTIVE_FORM_PREFIX in state_name and prob > 0
        ]
        # by construction there is only one active form
        return found_forms[0] if found_forms else None

    @staticmethod
    def _prev_action_listen_in_state(state):
        return any(
            PREV_PREFIX + ACTION_LISTEN_NAME in state_name and prob > 0
            for state_name, prob in state.items()
        )

    @staticmethod
    def _modified_states(states):
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

        return [action_before_listen, states[-1]]

    def _add_states_to_lookup(
        self, trackers_as_states, trackers_as_actions, domain, online=False
    ):
        """Add states to lookup dict"""
        for states in trackers_as_states:
            active_form = self._get_active_form_name(states[-1])
            if active_form and self._prev_action_listen_in_state(states[-1]):
                # modify the states
                states = self._modified_states(states)
                feature_key = self._create_feature_key(states)
                # even if there are two identical feature keys
                # their form will be the same
                # because of `active_form_...` feature
                self.lookup[feature_key] = active_form

    def recall(
        self,
        states: List[Dict[Text, float]],
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> Optional[int]:
        # modify the states
        return self._recall_states(self._modified_states(states))

    def state_is_unhappy(self, tracker, domain):
        # since it is assumed that training stories contain
        # only unhappy paths, notify the form that
        # it should not be validated if predicted by other policy
        tracker_as_states = self.featurizer.prediction_states([tracker], domain)
        states = tracker_as_states[0]

        memorized_form = self.recall(states, tracker, domain)

        state_is_unhappy = (
            memorized_form is not None
            and memorized_form == tracker.active_form.get("name")
        )

        if state_is_unhappy:
            logger.debug(
                "There is a memorized tracker state {}, "
                "added `FormValidation(False)` event"
                "".format(self._modified_states(states))
            )

        return state_is_unhappy

    def predict_action_probabilities(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        """Predicts the corresponding form action if there is an active form"""
        result = [0.0] * domain.num_actions

        if tracker.active_form.get("name"):
            logger.debug(
                "There is an active form '{}'".format(tracker.active_form["name"])
            )
            if tracker.latest_action_name == ACTION_LISTEN_NAME:
                # predict form action after user utterance

                if tracker.active_form.get("rejected"):
                    if self.state_is_unhappy(tracker, domain):
                        tracker.update(FormValidation(False))
                        return result

                idx = domain.index_for_action(tracker.active_form["name"])
                result[idx] = 1.0

            elif tracker.latest_action_name == tracker.active_form.get("name"):
                # predict action_listen after form action
                idx = domain.index_for_action(ACTION_LISTEN_NAME)
                result[idx] = 1.0
        else:
            logger.debug("There is no active form")

        return result
