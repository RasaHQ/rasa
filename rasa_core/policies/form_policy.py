from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import typing

from typing import Any, List, Optional, Dict

from rasa_core.constants import FORM_SCORE
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.events import (ActionExecuted,
                              ActionExecutionRejected,
                              NoFormValidation)

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

        super(FormPolicy, self).__init__(featurizer=featurizer,
                                         max_history=1,
                                         lookup=lookup)

    @staticmethod
    def _active_form_in(state):
        return any(ACTIVE_FORM_PREFIX in k and v > 0 for k, v in state.items())

    @staticmethod
    def _prev_action_listen_in(state):
        return any(PREV_PREFIX + ACTION_LISTEN_NAME in k and v > 0
                   for k, v in state.items())

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: Any
              ):
        # type: (...) -> None
        """Finds intents that shouldn't be validated.
            Assumes that stories contain only unhappy paths"""

        self.lookup = {}

        trackers_as_states, _ = self.featurizer.training_states_and_actions(
                training_trackers, domain)

        for states in trackers_as_states:
            state = states[0]
            if (self._active_form_in(state) and
                    self._prev_action_listen_in(state)):
                # by construction there is only one active form
                form = [k[len(ACTIVE_FORM_PREFIX):]
                        for k, v in state.items()
                        if ACTIVE_FORM_PREFIX in k and v > 0][0]

                feature_key = self._create_feature_key(states)
                # even if there are two identical feature keys
                # their form will be the same
                # because of `active_form_...` feature
                self.lookup[feature_key] = form

    @staticmethod
    def _form_was_rejected(tracker):
        # type: (DialogueStateTracker) -> bool
        """Check whether previous call to the form was rejected"""
        for event in reversed(tracker.applied_events()):
            if (isinstance(event, ActionExecuted) and
                    event.action_name == tracker.active_form):
                return False
            elif (isinstance(event, ActionExecutionRejected) and
                    event.action_name == tracker.active_form):
                return True

        return False

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts a form action if form is active"""
        result = [0.0] * domain.num_actions

        if tracker.active_form:
            logger.debug("There is an active form '{}'"
                         "".format(tracker.active_form))
            if tracker.latest_action_name == ACTION_LISTEN_NAME:
                # predict form action after user utterance

                if self._form_was_rejected(tracker):
                    # since it is assumed that training stories contain
                    # only unhappy paths, notify the form that
                    # it should not be validated if predicted by other policy
                    tracker_as_states = self.featurizer.prediction_states(
                            [tracker], domain)
                    states = tracker_as_states[0]
                    memorized_form = self._recall_states(states)

                    if memorized_form == tracker.active_form:
                        logger.debug("There is a memorized tracker state {}, "
                                     "added `NoFormValidation` event"
                                     "".format(states))
                        tracker.update(NoFormValidation())
                        return result

                idx = domain.index_for_action(tracker.active_form)
                result[idx] = FORM_SCORE

            elif tracker.latest_action_name == tracker.active_form:
                # predict action_listen after form action
                idx = domain.index_for_action(ACTION_LISTEN_NAME)
                result[idx] = FORM_SCORE
        else:
            logger.debug("There is no active form")

        return result
