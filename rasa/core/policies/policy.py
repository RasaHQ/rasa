from __future__ import annotations

import abc
import copy
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
    Tuple,
    TypeVar,
    TYPE_CHECKING,
)

import numpy as np

from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.events import Event
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
import rasa.utils.common
import rasa.shared.utils.io
from rasa.shared.exceptions import RasaException, FileIOException
from rasa.shared.nlu.constants import ENTITIES, INTENT, TEXT, ACTION_TEXT, ACTION_NAME
from rasa.shared.core.domain import Domain, State
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.constants import (
    DEFAULT_POLICY_PRIORITY,
    POLICY_PRIORITY,
    POLICY_MAX_HISTORY,
)
from rasa.shared.core.constants import USER, SLOTS, PREVIOUS_ACTION, ACTIVE_LOOP
import rasa.shared.utils.common

if TYPE_CHECKING:
    from rasa.shared.nlu.training_data.features import Features
    from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
    from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
    from rasa.dialogue_understanding.stack.frames import DialogueStackFrame

logger = logging.getLogger(__name__)

TrackerListTypeVar = TypeVar(
    "TrackerListTypeVar", List[DialogueStateTracker], List[TrackerWithCachedStates]
)


class SupportedData(Enum):
    """Enumeration of a policy's supported training data type."""

    # policy only supports ML-based training data ("stories")
    ML_DATA = 1

    # policy only supports rule-based data ("rules")
    RULE_DATA = 2

    # policy supports both ML-based and rule-based data ("stories" as well as "rules")
    ML_AND_RULE_DATA = 3

    @staticmethod
    def trackers_for_supported_data(
        supported_data: SupportedData,
        trackers: TrackerListTypeVar,
    ) -> TrackerListTypeVar:
        """Return trackers for a given policy.

        Args:
            supported_data: Supported data filter for the `trackers`.
            trackers: Trackers to split.

        Returns:
            Trackers from ML-based training data and/or rule-based data.
        """
        if supported_data == SupportedData.RULE_DATA:
            return [tracker for tracker in trackers if tracker.is_rule_tracker]

        if supported_data == SupportedData.ML_DATA:
            return [tracker for tracker in trackers if not tracker.is_rule_tracker]

        # `supported_data` is `SupportedData.ML_AND_RULE_DATA`
        return trackers


class Policy(GraphComponent):
    """Common parent class for all dialogue policies."""

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
    def does_support_stack_frame(frame: "DialogueStackFrame") -> bool:
        """Determine whether a stack frame is supported by the policy."""
        return False

    def supports_current_stack_frame(
        self,
        tracker: DialogueStateTracker,
        only_after_user_message: bool = True,
        empty_stack_behavior: bool = True,
    ) -> bool:
        """Check whether the policy is allowed to act.

        Args:
            tracker: The tracker.
            only_after_user_message: Whether to only allow actions after a user message.
            empty_stack_behavior: Whether to allow actions if the stack is empty.

        Returns:
            `True` if the policy is allowed to act, `False` otherwise.
        """
        if top_frame := tracker.stack.top():
            return self.does_support_stack_frame(top_frame)
        elif only_after_user_message and len(tracker.events) > 0:
            return not tracker.has_action_after_latest_user_message()
        else:
            return empty_stack_behavior

    def should_abstain_in_coexistence(
        self, tracker: DialogueStateTracker, is_calm_policy: bool
    ) -> bool:
        """Whether a policy should abstain making predictions in coexistence.

        A calm policy should run when the routing slot is set to True.
        A nlu-based policy should run when the routing slot is set to False or None.
        """
        if is_calm_policy:
            return tracker.has_coexistence_routing_slot and (
                tracker.get_slot(ROUTE_TO_CALM_SLOT) is False
                or tracker.get_slot(ROUTE_TO_CALM_SLOT) is None
            )

        return (
            tracker.has_coexistence_routing_slot
            and tracker.get_slot(ROUTE_TO_CALM_SLOT) is True
        )

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: Optional["TrackerFeaturizer"] = None,
    ) -> None:
        """Constructs a new Policy object."""
        self.config = config
        if featurizer is None:
            featurizer = self._create_featurizer()
        self.__featurizer = featurizer

        self.priority = config.get(POLICY_PRIORITY, DEFAULT_POLICY_PRIORITY)
        self.finetune_mode = execution_context.is_finetuning

        self._model_storage = model_storage
        self._resource = resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> Policy:
        """Creates a new untrained policy (see parent class for full docstring)."""
        return cls(config, model_storage, resource, execution_context)

    def _create_featurizer(self) -> "TrackerFeaturizer":
        from rasa.core.featurizers.tracker_featurizers import (
            MaxHistoryTrackerFeaturizer,
        )

        policy_config = copy.deepcopy(self.config)

        featurizer_configs = policy_config.get("featurizer")

        if not featurizer_configs:
            return self._standard_featurizer()

        featurizer_func = _get_featurizer_from_config(
            featurizer_configs,
            self.__class__.__name__,
            lookup_path="rasa.core.featurizers.tracker_featurizers",
        )
        featurizer_config = featurizer_configs[0]

        state_featurizer_configs = featurizer_config.pop("state_featurizer", None)
        if state_featurizer_configs:
            state_featurizer_func = _get_featurizer_from_config(
                state_featurizer_configs,
                self.__class__.__name__,
                lookup_path="rasa.core.featurizers.single_state_featurizer",
            )
            state_featurizer_config = state_featurizer_configs[0]

            featurizer_config["state_featurizer"] = state_featurizer_func(
                **state_featurizer_config
            )

        featurizer = featurizer_func(**featurizer_config)
        if (
            isinstance(featurizer, MaxHistoryTrackerFeaturizer)
            and POLICY_MAX_HISTORY in policy_config
            and POLICY_MAX_HISTORY not in featurizer_config
        ):
            featurizer.max_history = policy_config[POLICY_MAX_HISTORY]
        return featurizer

    def _standard_featurizer(self) -> "MaxHistoryTrackerFeaturizer":
        """Initializes the standard featurizer for this policy."""
        from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
        from rasa.core.featurizers.tracker_featurizers import (
            MaxHistoryTrackerFeaturizer,
        )

        return MaxHistoryTrackerFeaturizer(
            SingleStateFeaturizer(), self.config.get(POLICY_MAX_HISTORY)
        )

    @property
    def featurizer(self) -> "TrackerFeaturizer":
        """Returns the policy's featurizer."""
        return self.__featurizer

    @staticmethod
    def _get_valid_params(func: Callable, **kwargs: Any) -> Dict:
        """Filters out kwargs that cannot be passed to func.

        Args:
            func: a callable function
            **kwargs: the dictionary of parameters

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
        precomputations: Optional[MessageContainerForCoreFeaturization],
        bilou_tagging: bool = False,
        **kwargs: Any,
    ) -> Tuple[
        List[List[Dict[Text, List[Features]]]],
        np.ndarray,
        List[List[Dict[Text, List[Features]]]],
    ]:
        """Transform training trackers into a vector representation.

        The trackers, consisting of multiple turns, will be transformed
        into a float vector which can be used by a ML model.

        Args:
            training_trackers:
                the list of the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.shared.core.domain.Domain`
            precomputations: Contains precomputed features and attributes.
            bilou_tagging: indicates whether BILOU tagging should be used or not
            **kwargs: Additional arguments.

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
        state_features, label_ids, entity_tags = self.featurizer.featurize_trackers(
            training_trackers,
            domain,
            precomputations=precomputations,
            bilou_tagging=bilou_tagging,
            ignore_action_unlikely_intent=self.supported_data()
            == SupportedData.ML_DATA,
        )

        max_training_samples = kwargs.get("max_training_samples")
        if max_training_samples is not None:
            logger.debug(
                "Limit training data to {} training samples.".format(
                    max_training_samples
                )
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
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[State]:
        """Transforms tracker to states for prediction.

        Args:
            tracker: The tracker to be featurized.
            domain: The Domain.
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.

        Returns:
            A list of states.
        """
        return self.featurizer.prediction_states(
            [tracker],
            domain,
            use_text_for_last_user_input=use_text_for_last_user_input,
            ignore_rule_only_turns=self.supported_data() == SupportedData.ML_DATA,
            rule_only_data=rule_only_data,
            ignore_action_unlikely_intent=self.supported_data()
            == SupportedData.ML_DATA,
        )[0]

    def _featurize_for_prediction(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization],
        rule_only_data: Optional[Dict[Text, Any]],
        use_text_for_last_user_input: bool = False,
    ) -> List[List[Dict[Text, List[Features]]]]:
        """Transforms training tracker into a vector representation.

        The trackers, consisting of multiple turns, will be transformed
        into a float vector which can be used by a ML model.

        Args:
            tracker: The tracker to be featurized.
            domain: The Domain.
            precomputations: Contains precomputed features and attributes.
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.

        Returns:
            A list (corresponds to the list of trackers)
            of lists (corresponds to all dialogue turns)
            of dictionaries of state type (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
            ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
            turns in all trackers.
        """
        return self.featurizer.create_state_features(
            [tracker],
            domain,
            precomputations=precomputations,
            use_text_for_last_user_input=use_text_for_last_user_input,
            ignore_rule_only_turns=self.supported_data() == SupportedData.ML_DATA,
            rule_only_data=rule_only_data,
            ignore_action_unlikely_intent=self.supported_data()
            == SupportedData.ML_DATA,
        )

    @abc.abstractmethod
    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:
        """Trains a policy.

        Args:
            training_trackers: The story and rules trackers from the training data.
            domain: The model's domain.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to train itself.

        Returns:
            A policy must return its resource locator so that potential children nodes
            can load the policy from the resource.
        """
        raise NotImplementedError("Policy must have the capacity to train.")

    @abc.abstractmethod
    async def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.

        Backwards compatibility: this method can also be implemented without
        `async`. In that case, the method will be run in a synchronous way.

        Args:
            tracker: The tracker containing the conversation history up to now.
            domain: The model's domain.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to make predictions.

        Returns:
             The prediction.
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
    ) -> PolicyPrediction:
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

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> Policy:
        """Loads a trained policy (see parent class for full docstring)."""
        from rasa.core.featurizers.tracker_featurizers import (
            TrackerFeaturizer,
            FEATURIZER_FILE,
        )

        featurizer = None

        try:
            with model_storage.read_from(resource) as path:
                if (Path(path) / FEATURIZER_FILE).is_file():
                    featurizer = TrackerFeaturizer.load(path)

                config.update(kwargs)

        except (ValueError, FileNotFoundError, FileIOException):
            logger.debug(
                f"Couldn't load metadata for policy '{cls.__name__}' as the persisted "
                f"metadata couldn't be loaded."
            )

        return cls(
            config, model_storage, resource, execution_context, featurizer=featurizer
        )

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
                            state_messages.append(f"user text: {state[USER][TEXT]!s}")
                        if INTENT in state[USER]:
                            state_messages.append(
                                f"user intent: {state[USER][INTENT]!s}"
                            )
                        if ENTITIES in state[USER]:
                            state_messages.append(
                                f"user entities: {state[USER][ENTITIES]!s}"
                            )
                    if PREVIOUS_ACTION in state:
                        if ACTION_NAME in state[PREVIOUS_ACTION]:
                            state_messages.append(
                                f"previous action name: "
                                f"{state[PREVIOUS_ACTION][ACTION_NAME]!s}"
                            )
                        if ACTION_TEXT in state[PREVIOUS_ACTION]:
                            state_messages.append(
                                f"previous action text: "
                                f"{state[PREVIOUS_ACTION][ACTION_TEXT]!s}"
                            )
                    if ACTIVE_LOOP in state:
                        state_messages.append(f"active loop: {state[ACTIVE_LOOP]!s}")
                    if SLOTS in state:
                        state_messages.append(f"slots: {state[SLOTS]!s}")
                    state_message_formatted = " | ".join(state_messages)
                    state_formatted = f"[state {index!s}] {state_message_formatted}"
                    formatted_states.append(state_formatted)

        return "\n".join(formatted_states)

    def __repr__(self) -> Text:
        """Returns text representation of object."""
        return f"{self.__class__.__name__}@{id(self)}"


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
        is_no_user_prediction: bool = False,
        diagnostic_data: Optional[Dict[Text, Any]] = None,
        hide_rule_turn: bool = False,
        action_metadata: Optional[Dict[Text, Any]] = None,
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
            is_no_user_prediction: `True` if the prediction uses neither the text
                of the user message nor the intent. This is for the example the case
                for happy loop paths.
            diagnostic_data: Intermediate results or other information that is not
                necessary for Rasa to function, but intended for debugging and
                fine-tuning purposes.
            hide_rule_turn: `True` if the prediction was made by the rules which
                do not appear in the stories
            action_metadata: Specifies additional metadata that can be passed
                by policies.
        """
        self.probabilities = probabilities
        self.policy_name = policy_name
        self.policy_priority = policy_priority
        self.events = events or []
        self.optional_events = optional_events or []
        self.is_end_to_end_prediction = is_end_to_end_prediction
        self.is_no_user_prediction = is_no_user_prediction
        self.diagnostic_data = diagnostic_data or {}
        self.hide_rule_turn = hide_rule_turn
        self.action_metadata = action_metadata

    @staticmethod
    def for_action_name(
        domain: Domain,
        action_name: Text,
        policy_name: Optional[Text] = None,
        confidence: float = 1.0,
        action_metadata: Optional[Dict[Text, Any]] = None,
    ) -> "PolicyPrediction":
        """Create a prediction for a given action.

        Args:
            domain: The current model domain
            action_name: The action which should be predicted.
            policy_name: The policy which did the prediction.
            confidence: The prediction confidence.
            action_metadata: Additional metadata to be attached with the prediction.

        Returns:
            The prediction.
        """
        probabilities = confidence_scores_for(action_name, confidence, domain)

        return PolicyPrediction(
            probabilities, policy_name, action_metadata=action_metadata
        )

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
            and self.optional_events == other.optional_events
            and self.is_end_to_end_prediction == other.is_end_to_end_prediction
            and self.is_no_user_prediction == other.is_no_user_prediction
            and self.hide_rule_turn == other.hide_rule_turn
            and self.action_metadata == other.action_metadata
            # We do not compare `diagnostic_data`, because it has no effect on the
            # action prediction.
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
        """Gets the highest predicted confidence.

        Returns:
            The highest predicted confidence.
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


class InvalidPolicyConfig(RasaException):
    """Exception that can be raised when policy config is not valid."""


def _get_featurizer_from_config(
    config: List[Dict[Text, Any]], policy_name: Text, lookup_path: Text
) -> Callable[..., "TrackerFeaturizer"]:
    """Gets the featurizer initializer and its arguments from a policy config."""
    # Only 1 featurizer is allowed
    if len(config) > 1:
        featurizer_names = [
            featurizer_config.get("name") for featurizer_config in config
        ]
        raise InvalidPolicyConfig(
            f"Every policy can only have 1 featurizer but '{policy_name}' "
            f"uses {len(config)} featurizers ('{', '.join(featurizer_names)}')."
        )

    featurizer_config = config[0]
    featurizer_name = featurizer_config.pop("name")
    featurizer_func = rasa.shared.utils.common.class_from_module_path(
        featurizer_name, lookup_path=lookup_path
    )

    return featurizer_func
