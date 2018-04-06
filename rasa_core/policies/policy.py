from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import typing

from builtins import object
from typing import Any, List, Optional, Text, Dict
from rasa_core.featurizers import \
    FullDialogueFeaturizer, BinaryFeaturizeMechanism

if typing.TYPE_CHECKING:
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

    def prepare(self, featurizer):
        # type: (Featurizer) -> None

        if self.featurizer is None:
            self.featurizer = featurizer
        else:
            logger.warning("Trying to reset featurizer {} "
                           "for policy {} by agent featurizer {}. "
                           "Agent featurizer is ignored."
                           "".format(type(self.featurizer),
                                     type(self),
                                     type(featurizer)))

    @staticmethod
    def _standard_featurizer():
        return FullDialogueFeaturizer(BinaryFeaturizeMechanism())

    def featurize_for_training(
            self,
            trackers,  # type: List[DialogueStateTracker]
            domain,  # type: Domain
            max_training_samples=None  # type: Optional[int]
    ):
        # type: (...) -> DialogueTrainingData
        """Transform training trackers into a vector representation.
        The trackers, consisting of multiple turns, will be transformed
        into a float vector which can be used by a ML model."""

        if self.featurizer is None:
            self.featurizer = self._standard_featurizer()

        training_data, _ = self.featurizer.featurize_trackers(trackers,
                                                              domain)
        if max_training_samples:
            training_data.limit_training_data_to(max_training_samples)

        return training_data

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]

        return []

    def train(self, training_trackers, domain, **kwargs):
        # type: (List[DialogueStateTracker], Domain, **Any) -> Dict[Text: Any]
        """Trains the policy on given training data.
        And returns training metadata."""

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
        """Persists the policy to a storage."""

        pass

    @classmethod
    def load(cls, path, featurizer):
        # type: (Text, Featurizer) -> Policy
        """Loads a policy from the storage."""

        raise NotImplementedError
