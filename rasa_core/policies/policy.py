from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from builtins import object
from typing import Any
from typing import List
from typing import Optional
from typing import Text

from rasa_core.domain import Domain
from rasa_core.featurizers import Featurizer
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training.data import DialogueTrainingData

logger = logging.getLogger(__name__)


class Policy(object):
    SUPPORTS_ONLINE_TRAINING = False
    MAX_HISTORY_DEFAULT = 3

    def __init__(self, featurizer=None):
        # type: (Optional[Featurizer]) -> None

        self.featurizer = featurizer

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]

        return []

    def prepare(self, featurizer):
        # type: (Featurizer) -> None
        self.featurizer = featurizer

    def train(self, training_data, domain, **kwargs):
        # type: (DialogueTrainingData, Domain, **Any) -> None
        """Trains the policy on given training data."""

        raise NotImplementedError

    def continue_training(self, training_data, domain, **kwargs):
        # type: (DialogueTrainingData, Domain, **Any) -> None
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
    def load(cls, path, featurizer):
        raise NotImplementedError
