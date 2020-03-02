import copy
import logging
from typing import Any, List, Optional, Text, Dict, Callable

import rasa.utils.common
from rasa.core.domain import Domain
from rasa.core.featurizers import (
    MaxHistoryTrackerFeaturizer,
    BinarySingleStateFeaturizer,
)
from rasa.core.featurizers import TrackerFeaturizer
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.data import DialogueTrainingData
from rasa.core.constants import DEFAULT_POLICY_PRIORITY


logger = logging.getLogger(__name__)


class Policy:
    SUPPORTS_ONLINE_TRAINING = False

    @staticmethod
    def _standard_featurizer() -> MaxHistoryTrackerFeaturizer:
        return MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer())

    @classmethod
    def _create_featurizer(cls, featurizer=None) -> TrackerFeaturizer:
        if featurizer:
            return copy.deepcopy(featurizer)
        else:
            return cls._standard_featurizer()

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
    ) -> None:
        self.__featurizer = self._create_featurizer(featurizer)
        self.priority = priority

    @property
    def featurizer(self):
        return self.__featurizer

    @staticmethod
    def _get_valid_params(func: Callable, **kwargs: Any) -> Dict:
        """Filters out kwargs that cannot be passed to func.

        Args:
            func: a callable function

        Returns:
            the dictionary of parameters
        """

        valid_keys = rasa.utils.common.arguments_of(func)

        params = {key: kwargs.get(key) for key in valid_keys if kwargs.get(key)}
        ignored_params = {
            key: kwargs.get(key) for key in kwargs.keys() if not params.get(key)
        }
        logger.debug(f"Parameters ignored by `model.fit(...)`: {ignored_params}")
        return params

    def featurize_for_training(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any,
    ) -> DialogueTrainingData:
        """Transform training trackers into a vector representation.

        The trackers, consisting of multiple turns, will be transformed
        into a float vector which can be used by a ML model.

        Args:
            training_trackers:
                the list of the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.core.domain.Domain`

        Returns:
            the :class:`rasa.core.training.data.DialogueTrainingData`
        """

        training_data = self.featurizer.featurize_trackers(training_trackers, domain)

        max_training_samples = kwargs.get("max_training_samples")
        if max_training_samples is not None:
            logger.debug(
                "Limit training data to {} training samples."
                "".format(max_training_samples)
            )
            training_data.limit_training_data_to(max_training_samples)

        return training_data

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any,
    ) -> None:
        """Trains the policy on given training trackers.

        Args:
            training_trackers:
                the list of the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.core.domain.Domain`
        """

        raise NotImplementedError("Policy must have the capacity to train.")

    def predict_action_probabilities(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.core.domain.Domain`

        Returns:
             the list of probabilities for the next actions
        """

        raise NotImplementedError("Policy must have the capacity to predict.")

    def persist(self, path: Text) -> None:
        """Persists the policy to a storage.

        Args:
            path: the path where to save the policy to
        """

        raise NotImplementedError("Policy must have the capacity to persist itself.")

    @classmethod
    def load(cls, path: Text) -> "Policy":
        """Loads a policy from the storage.

        Needs to load its featurizer.

        Args:
            path: the path from where to load the policy
        """

        raise NotImplementedError("Policy must have the capacity to load itself.")

    @staticmethod
    def _default_predictions(domain: Domain) -> List[float]:
        """Creates a list of zeros.

        Args:
            domain: the :class:`rasa.core.domain.Domain`
        Returns:
            the list of the length of the number of actions
        """

        return [0.0] * domain.num_actions


def confidence_scores_for(
    action_name: Text, value: float, domain: Domain
) -> List[float]:
    """Returns confidence scores if a single action is predicted.

    Args:
        action_name: the name of the action for which the score should be set
        value: the confidence for `action_name`
        domain: the :class:`rasa.core.domain.Domain`

    Returns:
        the list of the length of the number of actions
    """

    results = [0.0] * domain.num_actions
    idx = domain.index_for_action(action_name)
    results[idx] = value

    return results
