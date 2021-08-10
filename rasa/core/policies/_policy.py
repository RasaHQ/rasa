# WARNING: This module will be dropped before Rasa Open Source 3.0 is released.
#          Please don't do any changes in this module and rather adapt Policy2 from
#          the regular `rasa.core.policies.policy` module. This module is a workaround
#          to defer breaking changes due to the architecture revamp in 3.0.
import copy
import json
import logging
from pathlib import Path
from typing import (
    Optional,
    Any,
    List,
    Dict,
    Text,
    Tuple,
    Union,
    TYPE_CHECKING,
    Callable,
)

import numpy as np
from rasa.core.constants import DEFAULT_POLICY_PRIORITY
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import (
    MaxHistoryTrackerFeaturizer,
    TrackerFeaturizer,
    FEATURIZER_FILE,
)
import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.shared.core.constants import USER, PREVIOUS_ACTION, ACTIVE_LOOP, SLOTS
from rasa.shared.core.domain import Domain, State
from rasa.shared.core.events import Event
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import ENTITIES, TEXT, INTENT, ACTION_NAME, ACTION_TEXT
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter

if TYPE_CHECKING:
    from rasa.shared.nlu.training_data.features import Features
    from rasa.core.policies.policy import SupportedData, PolicyPrediction

logger = logging.getLogger(__name__)


class Policy:
    """Common parent class for all dialogue policies."""

    @staticmethod
    def supported_data() -> "SupportedData":
        """The type of data supported by this policy.

        By default, this is only ML-based training data. If policies support rule data,
        or both ML-based data and rule data, they need to override this method.

        Returns:
            The data type supported by this policy (ML-based training data).
        """
        from rasa.core.policies.policy import SupportedData

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
        should_finetune: bool = False,
        **kwargs: Any,
    ) -> None:
        """Constructs a new Policy object."""
        self.__featurizer = self._create_featurizer(featurizer)
        self.priority = priority
        self.finetune_mode = should_finetune
        self._rule_only_data = {}

    @property
    def featurizer(self) -> TrackerFeaturizer:
        """Returns the policy's featurizer."""
        return self.__featurizer

    def set_shared_policy_states(self, **kwargs: Any) -> None:
        """Sets policy's shared states for correct featurization."""
        self._rule_only_data = kwargs.get("rule_only_data", {})

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

    def _featurize_for_training(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        bilou_tagging: bool = False,
        **kwargs: Any,
    ) -> Tuple[
        List[List[Dict[Text, List["Features"]]]],
        np.ndarray,
        List[List[Dict[Text, List["Features"]]]],
    ]:
        """Transform training trackers into a vector representation.

        The trackers, consisting of multiple turns, will be transformed
        into a float vector which can be used by a ML model.

        Args:
            training_trackers:
                the list of the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.shared.core.domain.Domain`
            interpreter: the :class:`rasa.core.interpreter.NaturalLanguageInterpreter`
            bilou_tagging: indicates whether BILOU tagging should be used or not

        Returns:
            - a dictionary of attribute (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
              ENTITIES, SLOTS, FORM) to a list of features for all dialogue turns in
              all training trackers
            - the label ids (e.g. action ids) for every dialogue turn in all training
              trackers
            - A dictionary of entity type (ENTITY_TAGS) to a list of features
              containing entity tag ids for text user inputs otherwise empty dict
              for all dialogue turns in all training trackers
        """
        from rasa.core.policies.policy import SupportedData

        state_features, label_ids, entity_tags = self.featurizer.featurize_trackers(
            training_trackers,
            domain,
            interpreter,
            bilou_tagging,
            ignore_action_unlikely_intent=self.supported_data()
            == SupportedData.ML_DATA,
        )

        max_training_samples = kwargs.get("max_training_samples")
        if max_training_samples is not None:
            logger.debug(
                "Limit training data to {} training samples."
                "".format(max_training_samples)
            )
            state_features = state_features[:max_training_samples]
            label_ids = label_ids[:max_training_samples]
            entity_tags = entity_tags[:max_training_samples]

        return state_features, label_ids, entity_tags

    def _prediction_states(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        use_text_for_last_user_input: bool = False,
    ) -> List[State]:
        """Transforms tracker to states for prediction.

        Args:
            tracker: The tracker to be featurized.
            domain: The Domain.
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.

        Returns:
            A list of states.
        """
        from rasa.core.policies.policy import SupportedData

        return self.featurizer.prediction_states(
            [tracker],
            domain,
            use_text_for_last_user_input=use_text_for_last_user_input,
            ignore_rule_only_turns=self.supported_data() == SupportedData.ML_DATA,
            rule_only_data=self._rule_only_data,
            ignore_action_unlikely_intent=self.supported_data()
            == SupportedData.ML_DATA,
        )[0]

    def _featurize_for_prediction(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        use_text_for_last_user_input: bool = False,
    ) -> List[List[Dict[Text, List["Features"]]]]:
        """Transforms training tracker into a vector representation.

        The trackers, consisting of multiple turns, will be transformed
        into a float vector which can be used by a ML model.

        Args:
            tracker: The tracker to be featurized.
            domain: The Domain.
            interpreter: The NLU interpreter.
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.

        Returns:
            A list (corresponds to the list of trackers)
            of lists (corresponds to all dialogue turns)
            of dictionaries of state type (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
            ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
            turns in all trackers.
        """
        from rasa.core.policies.policy import SupportedData

        return self.featurizer.create_state_features(
            [tracker],
            domain,
            interpreter,
            use_text_for_last_user_input=use_text_for_last_user_input,
            ignore_rule_only_turns=self.supported_data() == SupportedData.ML_DATA,
            rule_only_data=self._rule_only_data,
            ignore_action_unlikely_intent=self.supported_data()
            == SupportedData.ML_DATA,
        )

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
            **kwargs: Additional keyword arguments.
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
        is_no_user_prediction: bool = False,
        diagnostic_data: Optional[Dict[Text, Any]] = None,
        action_metadata: Optional[Dict[Text, Any]] = None,
    ) -> "PolicyPrediction":
        from rasa.core.policies.policy import PolicyPrediction

        return PolicyPrediction(
            probabilities,
            self.__class__.__name__,
            self.priority,
            events,
            optional_events,
            is_end_to_end_prediction,
            is_no_user_prediction,
            diagnostic_data,
            action_metadata=action_metadata,
        )

    def _metadata(self) -> Optional[Dict[Text, Any]]:
        """Returns this policy's attributes that should be persisted.

        Policies using the default `persist()` and `load()` implementations must
        implement the `_metadata()` method."

        Returns:
            The policy metadata.
        """
        pass

    @classmethod
    def _metadata_filename(cls) -> Text:
        """Returns the filename of the persisted policy metadata.

        Policies using the default `persist()` and `load()` implementations must
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
    def load(cls, path: Union[Text, Path], **kwargs: Any) -> "Policy":
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

            data.update(kwargs)

            return cls(**data)

        logger.info(
            f"Couldn't load metadata for policy '{cls.__name__}'. "
            f"File '{metadata_file}' doesn't exist."
        )
        return cls()

    def _default_predictions(self, domain: Domain) -> List[float]:
        """Creates a list of zeros.

        Args:
            domain: the :class:`rasa.shared.core.domain.Domain`
        Returns:
            the list of the length of the number of actions
        """
        return [0.0] * domain.num_actions

    @staticmethod
    def format_tracker_states(states: List[Dict]) -> Text:
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
                                f"previous action name: "
                                f"{str(state[PREVIOUS_ACTION][ACTION_NAME])}"
                            )
                        if ACTION_TEXT in state[PREVIOUS_ACTION]:
                            state_messages.append(
                                f"previous action text: "
                                f"{str(state[PREVIOUS_ACTION][ACTION_TEXT])}"
                            )
                    if ACTIVE_LOOP in state:
                        state_messages.append(f"active loop: {str(state[ACTIVE_LOOP])}")
                    if SLOTS in state:
                        state_messages.append(f"slots: {str(state[SLOTS])}")
                    state_message_formatted = " | ".join(state_messages)
                    state_formatted = f"[state {str(index)}] {state_message_formatted}"
                    formatted_states.append(state_formatted)

        return "\n".join(formatted_states)
