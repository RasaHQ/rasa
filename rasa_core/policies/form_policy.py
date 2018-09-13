from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import typing

from typing import Any, List, Text

from rasa_core.constants import FORM_SCORE, EXTRACTED_SLOT
from rasa_core.policies import Policy

from rasa_core.actions.action import ACTION_LISTEN_NAME

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


class FormPolicy(Policy):
    @staticmethod
    def _standard_featurizer():
        return None

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: Any
              ):
        # type: (...) -> None
        """Does nothing. This policy is deterministic."""

        pass

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts a form action if form is active"""
        result = [0.0] * domain.num_actions

        if tracker.active_form:
            logger.debug("Form '{}' is active".format(tracker.active_form))
            if tracker.latest_action_name == ACTION_LISTEN_NAME:
                # predict form action after user utterance
                idx = domain.index_for_action(tracker.active_form)
                result[idx] = FORM_SCORE
            elif tracker.latest_action_name == tracker.active_form:
                # predict action_listen after form action
                idx = domain.index_for_action(ACTION_LISTEN_NAME)
                result[idx] = FORM_SCORE
        else:
            logger.debug("There is no active form")

        return result

    def persist(self, path):
        # type: (Text) -> None
        """Persists the policy to storage."""
        pass

    @classmethod
    def load(cls, path):
        # type: (Text) -> FormPolicy
        return cls()
