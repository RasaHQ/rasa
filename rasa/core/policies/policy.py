import copy
import logging
from enum import Enum
import typing
from typing import Any, List, Optional, Text, Dict, Callable, Type, Union, Tuple
import numpy as np

import rasa.utils.common
from rasa.shared.core.domain import Domain
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
)
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.constants import DEFAULT_POLICY_PRIORITY

if typing.TYPE_CHECKING:
    from rasa.shared.nlu.training_data.features import Features


logger = logging.getLogger(__name__)


class SupportedData(Enum):
    """Enumeration of a policy's supported training data type."""

    # policy only supports ML-based training data ("stories")
    ML_DATA = 1

    # policy only supports rule-based data ("rules")
    RULE_DATA = 2

    # policy supports both ML-based and rule-based data ("stories" as well as "rules")
    ML_AND_RULE_DATA = 3

    @staticmethod
    def trackers_for_policy(
        policy: Union["Policy", Type["Policy"]], trackers: List[DialogueStateTracker]
    ) -> List[DialogueStateTracker]:
        """Return trackers for a given policy.

        Args:
            policy: Policy or policy type to return trackers for.
            trackers: Trackers to split.

        Returns:
            Trackers from ML-based training data and/or rule-based data.
        """
        supported_data = policy.supported_data()

        if supported_data == SupportedData.RULE_DATA:
            return [tracker for tracker in trackers if tracker.is_rule_tracker]

        if supported_data == SupportedData.ML_DATA:
            return [tracker for tracker in trackers if not tracker.is_rule_tracker]

        # `supported_data` is `SupportedData.ML_AND_RULE_DATA`
        return trackers


class Policy:
    SUPPORTS_ONLINE_TRAINING = False

    @staticmethod
    def supported_data() -> SupportedData:
        """The type of data supported by this policy.

        By default, this is only ML-based training data. If policies support rule data,
        or both ML-based data and rule data, they need to override this method.

        Returns:
            The data type supported by this policy (ML-based training data).
        """
        return SupportedData.ML_DATA

    @staticmethod
    def _standard_featurizer() -> MaxHistoryTrackerFeaturizer:
        return MaxHistoryTrackerFeaturizer(SingleStateFeaturizer())

    @classmethod
    def _create_featurizer(
        cls, featurizer: Optional[TrackerFeaturizer] = None
    ) -> TrackerFeaturizer:
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
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> Tuple[List[List[Dict[Text, List["Features"]]]], np.ndarray]:
        """Transform training trackers into a vector representation.

        The trackers, consisting of multiple turns, will be transformed
        into a float vector which can be used by a ML model.

        Args:
            training_trackers:
                the list of the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.shared.core.domain.Domain`
            interpreter: the :class:`rasa.core.interpreter.NaturalLanguageInterpreter`

        Returns:
            - a dictionary of attribute (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
              ENTITIES, SLOTS, FORM) to a list of features for all dialogue turns in
              all training trackers
            - the label ids (e.g. action ids) for every dialuge turn in all training
              trackers
        """

        state_features, label_ids = self.featurizer.featurize_trackers(
            training_trackers, domain, interpreter
        )

        max_training_samples = kwargs.get("max_training_samples")
        if max_training_samples is not None:
            logger.debug(
                "Limit training data to {} training samples."
                "".format(max_training_samples)
            )
            state_features = state_features[:max_training_samples]
            label_ids = label_ids[:max_training_samples]

        return state_features, label_ids

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        """Trains the policy on given training trackers.

        Args:
            training_trackers:
                the list of the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.shared.core.domain.Domain`
            interpreter: Interpreter which can be used by the polices for featurization.
        """

        raise NotImplementedError("Policy must have the capacity to train.")

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> List[float]:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.shared.core.domain.Domain`
            interpreter: Interpreter which may be used by the policies to create
                additional features.

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
            domain: the :class:`rasa.shared.core.domain.Domain`
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
        domain: the :class:`rasa.shared.core.domain.Domain`

    Returns:
        the list of the length of the number of actions
    """

    results = [0.0] * domain.num_actions
    idx = domain.index_for_action(action_name)
    results[idx] = value

    return results
