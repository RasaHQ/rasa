from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
from builtins import object
from numpy.core.records import ndarray
from typing import Any
from typing import List
from typing import Optional
from typing import Text

from rasa_core.domain import Domain
from rasa_core.featurizers import Featurizer
from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


class Policy(object):
    SUPPORTS_ONLINE_TRAINING = False
    MAX_HISTORY_DEFAULT = 3

    def __init__(self, featurizer=None, max_history=None):
        # type: (Optional[Featurizer]) -> None

        self.featurizer = featurizer
        self.max_history = max_history

    def featurize(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> ndarray
        """Transform tracker into a vector representation.

        The tracker, consisting of multiple turns, will be transformed
        into a float vector which can be used by a ML model."""

        x = domain.feature_vector_for_tracker(self.featurizer, tracker,
                                              self.max_history)
        return np.array(x)

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]

        return []

    def prepare(self, featurizer, max_history):
        self.featurizer = featurizer
        self.max_history = max_history

    def train(self, X, y, domain, **kwargs):
        # type: (ndarray, List[int], Domain, **Any) -> None
        """Trains the policy on given training data."""

        raise NotImplementedError

    def continue_training(self, X, y, domain, **kwargs):
        """Continues training an already trained policy.

        This doesn't need to be supported by every policy. If it is supported,
        the policy can be used for online training and the implementation for
        the continued training should be put into this function."""
        pass

    def persist(self, path):
        # type: (Text) -> None
        """Persists the policy to storage."""

        pass

    @classmethod
    def load(cls, path, featurizer, max_history):
        raise NotImplementedError
