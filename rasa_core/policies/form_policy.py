from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import typing

from typing import List, Optional, Dict

from rasa_core.constants import FORM_SCORE
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.events import FormValidation

from rasa_core.domain import PREV_PREFIX, ACTIVE_FORM_PREFIX
from rasa_core.actions.action import ACTION_LISTEN_NAME

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.featurizers import TrackerFeaturizer

logger = logging.getLogger(__name__)


class FormPolicy(MemoizationPolicy):
    """Policy which handles deterministic prediction of Forms"""

    ENABLE_FEATURE_STRING_COMPRESSION = True

    def __init__(self,
                 featurizer=None,  # type: Optional[TrackerFeaturizer]
                 lookup=None  # type: Optional[Dict]
                 ):
        # type: (...) -> None

        # max history is set to 2 in order to capture
        # previous meaningful action before action listen
        super(FormPolicy, self).__init__(featurizer=featurizer,
                                         max_history=2,
                                         lookup=lookup)

    @staticmethod
    def _active_form_in_state(state):
        return any(ACTIVE_FORM_PREFIX in k and v > 0 for k, v in state.items())

    @staticmethod
    def _prev_action_listen_in_state(state):
        return any(PREV_PREFIX + ACTION_LISTEN_NAME in k and v > 0
                   for k, v in state.items())

    @staticmethod
    def _modified_states(states):
        """Modify the states to
            - capture previous meaningful action before action_listen
            - ignore previous intent
        """
        prev_prev_action = {k: v
                            for k, v in states[0].items()
                            if PREV_PREFIX in k and v > 0}
        return [prev_prev_action, states[-1]]

    def _add(self, trackers_as_states, trackers_as_actions,
             domain, online=False):
        """Add states to lookup dict"""
        for states in trackers_as_states:
            if (self._active_form_in_state(states[-1]) and
                    self._prev_action_listen_in_state(states[-1])):
                # by construction there is only one active form
                form = [k[len(ACTIVE_FORM_PREFIX):]
                        for k, v in states[-1].items()
                        if ACTIVE_FORM_PREFIX in k and v > 0][0]
                # modify the states
                states = self._modified_states(states)
                feature_key = self._create_feature_key(states)
                # even if there are two identical feature keys
                # their form will be the same
                # because of `active_form_...` feature
                self.lookup[feature_key] = form

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts a form action if form is active"""
        result = [0.0] * domain.num_actions

        if tracker.active_form.get('name'):
            logger.debug("There is an active form '{}'"
                         "".format(tracker.active_form['name']))
            if tracker.latest_action_name == ACTION_LISTEN_NAME:
                # predict form action after user utterance

                if tracker.active_form.get('rejected'):
                    # since it is assumed that training stories contain
                    # only unhappy paths, notify the form that
                    # it should not be validated if predicted by other policy
                    tracker_as_states = self.featurizer.prediction_states(
                            [tracker], domain)
                    # modify the states
                    states = self._modified_states(tracker_as_states[0])
                    memorized_form = self._recall_states(states)

                    if memorized_form == tracker.active_form['name']:
                        logger.debug("There is a memorized tracker state {}, "
                                     "added `FormValidation(False)` event"
                                     "".format(states))
                        tracker.update(FormValidation(False))
                        return result

                idx = domain.index_for_action(tracker.active_form['name'])
                result[idx] = FORM_SCORE

            elif tracker.latest_action_name == tracker.active_form.get('name'):
                # predict action_listen after form action
                idx = domain.index_for_action(ACTION_LISTEN_NAME)
                result[idx] = FORM_SCORE
        else:
            logger.debug("There is no active form")

        return result
