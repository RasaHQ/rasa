from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing

from typing import Any, List, Text

from rasa_core.constants import FORM_SCORE, EXTRACTED_SLOT
from rasa_core.policies import Policy


if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.trackers import DialogueStateTracker


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
            intent = tracker.latest_message.intent.get('name', '')
            if intent == EXTRACTED_SLOT:
                idx = domain.index_for_action(tracker.active_form)
                result[idx] = FORM_SCORE

        return result
