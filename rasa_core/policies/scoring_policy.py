from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np

from rasa_core.policies.memoization import MemoizationPolicy

logger = logging.getLogger(__name__)


class ScoringPolicy(MemoizationPolicy):
    SUPPORTS_ONLINE_TRAINING = True

    def predict_action_probabilities(self, tracker, domain):
        result = [0.0] * domain.num_actions
        slot_feature_idxs = [domain.index_of_feature(f)
                             for f in domain.slot_features]
        entity_feature_idxs = [domain.index_of_feature(f)
                               for f in domain.entity_features]
        x = self.featurize(tracker, domain)
        logger.debug('got features {}'.format(
                self.featurizer.decode(x, domain.input_features)))
        x_orig = np.array(x)  # COPY!
        for i in range(self.max_history):
            if i > 0:
                x[i - 1] = -1
            logger.debug('trying to recall {}'.format(
                    self.featurizer.decode(x, domain.input_features)))
            memorised = self.recall(x, domain)
            if memorised is not None:
                logger.debug("Used memorised next "
                             "action '{}'".format(memorised))
                result[memorised] = 1.0
                return result

        x = np.array(x_orig)
        # trying to recall with cleared slots and entities
        for i in range(self.max_history):
            x[i, slot_feature_idxs] = 0
            x[i, entity_feature_idxs] = 0
            logger.debug('trying to recall {}'.format(
                    self.featurizer.decode(x, domain.input_features)))
            memorised = self.recall(x, domain)
            if memorised is not None:
                logger.info("Used memorised next action '{}'".format(memorised))
                result[memorised] = 1.0
                return result

        # trying to recall with cleared slots and entities and history
        x = np.array(x_orig)
        for i in range(self.max_history):
            x[i, slot_feature_idxs] = 0
            x[i, entity_feature_idxs] = 0
            if i > 0:
                x[i - 1] = -1
            logger.debug('trying to recall {}'.format(
                    self.featurizer.decode(x, domain.input_features)))
            memorised = self.recall(x, domain)
            if memorised is not None:
                logger.info("Used memorised next action '{}'".format(memorised))
                result[memorised] = 1.0
                return result

        return result
