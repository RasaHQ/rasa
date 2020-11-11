import copy
import json
import logging
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    List,
    Optional,
    Text,
    Dict,
    Callable,
    Type,
    Union,
    Tuple,
    TYPE_CHECKING,
)
import numpy as np
from rasa.shared.core.events import Event

import rasa.shared.utils.common
import rasa.utils.common
import rasa.shared.utils.io
from rasa.shared.core.domain import Domain
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
    FEATURIZER_FILE,
)
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.constants import DEFAULT_POLICY_PRIORITY
from rasa.shared.core.constants import USER, SLOTS, PREVIOUS_ACTION, ACTIVE_LOOP
from rasa.shared.nlu.constants import ENTITIES, INTENT, TEXT, ACTION_TEXT, ACTION_NAME

if TYPE_CHECKING:
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
        policy: Union["Policy", Type["Policy"]],
        trackers: Union[List[DialogueStateTracker], List[TrackerWithCachedStates]],
    ) -> Union[List[DialogueStateTracker], List[TrackerWithCachedStates]]:
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

        valid_keys = rasa.shared.utils.common.arguments_of(func)

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
            - the label ids (e.g. action ids) for every dialogue turn in all training
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
    ) -> "PolicyPrediction":
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.shared.core.domain.Domain`
            interpreter: Interpreter which may be used by the policies to create
                additional features.

        Returns:
             The policy's prediction (e.g. the probabilities for the actions).
        """
        raise NotImplementedError("Policy must have the capacity to predict.")

    def _prediction(
        self,
        probabilities: List[float],
        events: Optional[List[Event]] = None,
        optional_events: Optional[List[Event]] = None,
        is_end_to_end_prediction: bool = False,
    ) -> "PolicyPrediction":
        return PolicyPrediction(
            probabilities,
            self.__class__.__name__,
            self.priority,
            events,
            optional_events,
            is_end_to_end_prediction,
        )

    def _metadata(self) -> Optional[Dict[Text, Any]]:
        """Returns this policy's attributes that should be persisted.

        Policies following the default `persist()` and `load()` templates must
        implement the `_metadata()` method."

        Returns:
            The policy metadata.
        """
        pass

    @classmethod
    def _metadata_filename(cls) -> Optional[Text]:
        """Returns the filename of the persisted policy metadata.

        Policies following the default `persist()` and `load()` templates must
        implement the `_metadata_filename()` method.

        Returns:
            The filename of the persisted policy metadata.
        """
        pass

    def persist(self, path: Union[Text, Path]) -> None:
        """Persists the policy to storage.

        Args:
            path: Path to persist policy to.
        """
        # not all policies have a featurizer
        if self.featurizer is not None:
            self.featurizer.persist(path)

        file = Path(path) / self._metadata_filename()

        rasa.shared.utils.io.create_directory_for_file(file)
        rasa.shared.utils.io.dump_obj_as_json_to_file(file, self._metadata())

    @classmethod
    def load(cls, path: Union[Text, Path]) -> "Policy":
        """Loads a policy from path.

        Args:
            path: Path to load policy from.

        Returns:
            An instance of `Policy`.
        """
        metadata_file = Path(path) / cls._metadata_filename()

        if metadata_file.is_file():
            data = json.loads(rasa.shared.utils.io.read_file(metadata_file))

            if (Path(path) / FEATURIZER_FILE).is_file():
                featurizer = TrackerFeaturizer.load(path)
                data["featurizer"] = featurizer

            return cls(**data)

        logger.info(
            f"Couldn't load metadata for policy '{cls.__name__}'. "
            f"File '{metadata_file}' doesn't exist."
        )
        return cls()

    @staticmethod
    def _default_predictions(domain: Domain) -> List[float]:
        """Creates a list of zeros.

        Args:
            domain: the :class:`rasa.shared.core.domain.Domain`
        Returns:
            the list of the length of the number of actions
        """

        return [0.0] * domain.num_actions

    def format_tracker_states(self, states: List[Dict]) -> Text:
        """Format tracker states to human readable format on debug log.

        Args:
            states: list of tracker states dicts

        Returns:
            the string of the states with user intents and actions
        """
        # empty string to insert line break before first state
        formatted_states = [""]
        if states:
            for index, state in enumerate(states):
                state_messages = []
                if state:
                    if USER in state:
                        if TEXT in state[USER]:
                            state_messages.append(
                                f"user text: {str(state[USER][TEXT])}"
                            )
                        if INTENT in state[USER]:
                            state_messages.append(
                                f"user intent: {str(state[USER][INTENT])}"
                            )
                        if ENTITIES in state[USER]:
                            state_messages.append(
                                f"user entities: {str(state[USER][ENTITIES])}"
                            )
                    if PREVIOUS_ACTION in state:
                        if ACTION_NAME in state[PREVIOUS_ACTION]:
                            state_messages.append(
                                f"previous action name: {str(state[PREVIOUS_ACTION][ACTION_NAME])}"
                            )
                        if ACTION_TEXT in state[PREVIOUS_ACTION]:
                            state_messages.append(
                                f"previous action text: {str(state[PREVIOUS_ACTION][ACTION_TEXT])}"
                            )
                    if ACTIVE_LOOP in state:
                        state_messages.append(f"active loop: {str(state[ACTIVE_LOOP])}")
                    if SLOTS in state:
                        state_messages.append(f"slots: {str(state[SLOTS])}")
                    state_message_formatted = " | ".join(state_messages)
                    state_formatted = f"[state {str(index)}] {state_message_formatted}"
                    formatted_states.append(state_formatted)

        return "\n".join(formatted_states)


class PolicyPrediction:
    """Stores information about the prediction of a `Policy`."""

    def __init__(
        self,
        probabilities: List[float],
        policy_name: Optional[Text],
        policy_priority: int = 1,
        events: Optional[List[Event]] = None,
        optional_events: Optional[List[Event]] = None,
        is_end_to_end_prediction: bool = False,
    ) -> None:
        """Creates a `PolicyPrediction`.

        Args:
            probabilities: The probabilities for each action.
            policy_name: Name of the policy which made the prediction.
            policy_priority: The priority of the policy which made the prediction.
            events: Events which the `Policy` needs to have applied to the tracker
                after the prediction. These events are applied independent of whether
                the policy wins against other policies or not. Be careful which events
                you return as they can potentially influence the conversation flow.
            optional_events: Events which the `Policy` needs to have applied to the
                tracker after the prediction in case it wins. These events are only
                applied in case the policy's prediction wins. Be careful which events
                you return as they can potentially influence the conversation flow.
            is_end_to_end_prediction: `True` if the prediction used the text of the
                user message instead of the intent.
        """
        self.probabilities = probabilities
        self.policy_name = policy_name
        self.policy_priority = (policy_priority,)
        self.events = events or []
        self.optional_events = optional_events or []
        self.is_end_to_end_prediction = is_end_to_end_prediction

    @staticmethod
    def for_action_name(
        domain: Domain,
        action_name: Text,
        policy_name: Optional[Text] = None,
        confidence: float = 1.0,
    ) -> "PolicyPrediction":
        """Create a prediction for a given action.

        Args:
            domain: The current model domain
            action_name: The action which should be predicted.
            policy_name: The policy which did the prediction.
            confidence: The prediction confidence.

        Returns:
            The prediction.
        """
        probabilities = confidence_scores_for(action_name, confidence, domain)

        return PolicyPrediction(probabilities, policy_name)

    def __eq__(self, other: Any) -> bool:
        """Checks if the two objects are equal.

        Args:
            other: Any other object.

        Returns:
            `True` if other has the same type and the values are the same.
        """
        if not isinstance(other, PolicyPrediction):
            return False

        return (
            self.probabilities == other.probabilities
            and self.policy_name == other.policy_name
            and self.policy_priority == other.policy_priority
            and self.events == other.events
            and self.optional_events == other.events
            and self.is_end_to_end_prediction == other.is_end_to_end_prediction
        )

    @property
    def max_confidence_index(self) -> int:
        """Gets the index of the action prediction with the highest confidence.

        Returns:
            The index of the action with the highest confidence.
        """
        return self.probabilities.index(self.max_confidence)

    @property
    def max_confidence(self) -> float:
        """Gets the highest predicted probability.

        Returns:
            The highest predicted probability.
        """
        return max(self.probabilities, default=0.0)


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
