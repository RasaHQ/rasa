from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import typing

from inspect import signature
from builtins import object
from typing import \
    Any, List, Optional, Text, Dict, Callable

from copy import deepcopy
from rasa_core.featurizers import \
    MaxHistoryTrackerFeaturizer, BinarySingleStateFeaturizer

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.featurizers import Featurizer
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.training.data import DialogueTrainingData

logger = logging.getLogger(__name__)


class Policy(object):
    SUPPORTS_ONLINE_TRAINING = False
    MAX_HISTORY_DEFAULT = 3

    @classmethod
    def _standard_featurizer(cls):
        return MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(),
                                    cls.MAX_HISTORY_DEFAULT)

    @classmethod
    def _create_featurizer(cls, featurizer=None):
        return featurizer if featurizer else cls._standard_featurizer()

    def __init__(self, featurizer=None):
        # type: (Optional[Featurizer]) -> None
        self.__featurizer = self._create_featurizer(featurizer)

    @property
    def featurizer(self):
        return self.__featurizer

    @staticmethod
    def _get_valid_params(func, **kwargs):
        # type: (Callable, **Any) -> Dict
        # filter out kwargs that cannot be passed to func
        valid_keys = signature(func).parameters.keys()
        params = {key: kwargs.get(key)
                  for key in valid_keys if kwargs.get(key)}
        ignored_params = {key: kwargs.get(key)
                          for key in kwargs.keys()
                          if not params.get(key)}
        logger.debug("Ignored parameters: {}"
                     "".format(ignored_params))
        return params

    def featurize_for_training(
            self,
            trackers,  # type: List[DialogueStateTracker]
            domain,  # type: Domain
            **kwargs  # type: **Any
    ):
        # type: (...) -> DialogueTrainingData
        """Transform training trackers into a vector representation.
        The trackers, consisting of multiple turns, will be transformed
        into a float vector which can be used by a ML model."""

        max_history = kwargs.get('max_history')
        if max_history:
            logger.warning("Passing `max_history` through agent is "
                           "deprecated. Pass appropriate featurizer "
                           "to the policy instead.")

        training_data = self.featurizer.featurize_trackers(trackers,
                                                           domain)

        max_training_samples = kwargs.get('max_training_samples')
        if max_training_samples:
            training_data.limit_training_data_to(max_training_samples)

        return training_data

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: **Any
              ):
        # type: (...) -> None
        """Trains the policy on given training trackers.

        Returns training metadata."""

        raise NotImplementedError("Policy must have the capacity "
                                  "to train.")

    def continue_training(self, trackers, domain, **kwargs):
        # type: (List[DialogueStateTracker], Domain, **Any) -> None
        """Continues training an already trained policy.

        This doesn't need to be supported by every policy. If it is supported,
        the policy can be used for online training and the implementation for
        the continued training should be put into this function."""

        pass

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts the next action the bot should take
        after seeing the tracker.

        Returns the list of probabilities for the next actions"""

        raise NotImplementedError("Policy must have the capacity "
                                  "to predict.")

    def persist(self, path):
        # type: (Text) -> None
        """Persists the policy to a storage."""
        if self.featurizer:
            self.featurizer.persist(path)

    @classmethod
    def load(cls, path):
        # type: (Text) -> Policy
        """Loads a policy from the storage.

        Needs to load its featurizer"""

        raise NotImplementedError("Policy must have the capacity "
                                  "to load itself.")
