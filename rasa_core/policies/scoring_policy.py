from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import typing
from typing import List

import numpy as np

from rasa_core.policies.memoization import MemoizationPolicy

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.domain import Domain


class ScoringPolicy(MemoizationPolicy):
    SUPPORTS_ONLINE_TRAINING = True

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]

        tracker_as_states = self.featurizer.prediction_states([tracker], domain)
        states = tracker_as_states[0]
        logger.debug('got features {}'.format(states))

        result = [0.0] * domain.num_actions

        memorised = self._recall(states)
        if memorised is not None and self.is_enabled:
            logger.debug("Used memorised next action '{}'".format(memorised))
            result[memorised] = 1.0
            return result

        # trying to recall with cleared history
        augmented_states = list(states)  # COPY!
        for i in range(self.max_history - 1):
            augmented_states[i] = None
            logger.debug('trying to recall {}'.format(augmented_states))
            memorised = self._recall(augmented_states)
            if memorised is not None:
                logger.debug("Used memorised next "
                             "action '{}'".format(memorised))
                result[memorised] = 1.0
                return result

        # trying to recall with cleared slots and entities
        augmented_states = list(states)  # COPY!
        for i in range(self.max_history):
            if augmented_states[i]:
                for slot in domain.slot_features:
                    augmented_states[i][slot] = 0.0
                for entity in domain.entity_features:
                    if augmented_states[i].get(entity):
                        augmented_states[i][entity] = 0.0

            logger.debug('trying to recall {}'.format(augmented_states))
            memorised = self._recall(augmented_states)
            if memorised is not None:
                logger.info("Used memorised next action '{}'".format(memorised))
                result[memorised] = 1.0
                return result

        # trying to recall with cleared slots and entities and history
        augmented_states = list(states)  # COPY!
        for i in range(self.max_history):
            if augmented_states[i]:
                for slot in domain.slot_features:
                    augmented_states[i][slot] = 0.0
                for entity in domain.entity_features:
                    if augmented_states[i].get(entity):
                        augmented_states[i][entity] = 0.0

            if i > 0:
                augmented_states[i - 1] = None
            logger.debug('trying to recall {}'.format(augmented_states))
            memorised = self._recall(augmented_states)
            if memorised is not None:
                logger.info("Used memorised next action '{}'".format(memorised))
                result[memorised] = 1.0
                return result

        return result
