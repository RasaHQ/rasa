from os import stat_result
import jsonpickle
import logging
import copy

from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Text, Any
import numpy as np

from rasa.architecture_prototype.interfaces import ComponentPersistorInterface
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.shared.core.domain import State, Domain, SubState
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    is_prev_action_listen_in_state,
)
from rasa.shared.core.constants import PREVIOUS_ACTION, USER
from rasa.shared.nlu.constants import TEXT, INTENT, ENTITIES, ACTION_NAME, ACTION_TEXT
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message

FEATURIZER_FILE = "featurizer.json"

logger = logging.getLogger(__name__)


def copy_state(
    state: State,
    sub_state_type_attribute_combinations: Optional[Dict[Text, List[Text]]] = None,
) -> State:
    """Creates a copy that only contains certain substate types and attributes.
    """
    if sub_state_type_attribute_combinations is None:
        return copy.deepcopy(state)
    partial_state = dict()
    for sub_state_type, attributes in sub_state_type_attribute_combinations.items():
        sub_state = dict()
        if sub_state_type in state and attributes:
            attributes_left = set(sub_state.keys()).intersection(attributes)
            for attribute in attributes_left:
                sub_state[attribute] = copy.deepcpoy(state[sub_state_type][attribute])
            partial_state[sub_state] = sub_state
    return partial_state


class InvalidStory(RasaException):
    """Exception that can be raised if story cannot be featurized."""

    def __init__(self, message: Text) -> None:
        self.message = message
        super(InvalidStory, self).__init__()

    def __str__(self) -> Text:
        return self.message


class TrackerFeaturizer:
    """ ...
    """

    def __init__(
        self,
        state_featurizer: Optional[SingleStateFeaturizer] = None,
        max_history: Optional[int] = None,
        remove_duplicates: bool = True,
        persistor: Optional[ComponentPersistorInterface] = None,
        bilout_tagging: bool = False,
    ) -> None:
        """Initialize the tracker featurizer.

        Args:
            state_featurizer: The state featurizer used to encode the states.
        """
        self.state_featurizer = state_featurizer
        self._persistor = persistor
        self.max_history = max_history
        self.remove_duplicates = remove_duplicates
        self.bilou_tagging = False

    @staticmethod
    def slice_state_history(
        states: List[State], slice_length: Optional[int]
    ) -> List[State]:
        """At most the last slice_length states from the given list.

        Args:
            states: The states
            slice_length: the slice length or None

        Returns:
            sub-sequence of the given list of states if slice_length is defined;
            otherwise just the given list of states
        """
        if not slice_length:
            return states

        return states[-slice_length:]

    @staticmethod
    def _hash_example(states: List[State]) -> int:
        """Hash states for efficient deduplication."""
        # FIXME: these hashes change with a new session - is that a problem?
        frozen_states = tuple(
            s if s is None else DialogueStateTracker.freeze_current_state(s)
            for s in states
        )
        return hash((frozen_states,))

    @staticmethod
    def _remove_user_text_if_intent(states: List[State]) -> None:
        for state in states:
            # remove text features to only use intent
            if state.get(USER, {}).get(INTENT) and state.get(USER, {}).get(TEXT):
                del state[USER][TEXT]

    def extract_states_from_trackers_for_training(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
    ) -> List[List[State]]:
        """Transforms list of trackers to lists of states.

        Note that these states contain entity and action information.

        Args:
            trackers: The trackers to transform
            domain: The domain
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            A tuple of list of states, list of actions and list of entity data.
        """
        list_of_state_sequences: List[List[State]] = []

        if self.remove_duplicates:
            # from multiple states that create equal featurizations
            # we only need to keep one.
            hashed_examples = set()

        logger.debug(
            "Creating states and action examples from "
            "collected trackers (by {}({}))..."
            "".format(type(self).__name__, type(self.state_featurizer).__name__)
        )
        pbar = tqdm(
            trackers,
            desc="Processed trackers",
            disable=rasa.shared.utils.io.is_logging_disabled(),
        )
        for tracker in pbar:

            states = tracker.past_states(
                domain,
                omit_unset_slots=omit_unset_slots,
                ignore_rule_only_turns=False,
                rule_only_data=None,
            )

            num_action_executed_events = 0
            for current_event in tracker.applied_events():
                if not isinstance(current_event, ActionExecuted):
                    continue
                num_action_executed_events += 1

                # use only actions which can be predicted at a stories start
                # TODO: There must be a better way to filter the past_states output
                # (cf. `_mark_first_action_in_story_steps_as_unpredictable` in
                # `core/generator`)
                if current_event.unpredictable:
                    continue

                sliced_states = self.slice_state_history(
                    states[: (num_action_executed_events + 1)],
                    self.max_history
                    if (self.max_history is None)
                    else (self.max_history + 1),
                )
                if self.remove_duplicates:
                    # only continue with tracker_states that created a
                    # hashed_featurization we haven't observed
                    hashed = self._hash_example(sliced_states)
                    if hashed not in hashed_examples:
                        hashed_examples.add(hashed)
                        list_of_state_sequences.append(sliced_states)
                else:
                    list_of_state_sequences.append(sliced_states)

        # NOTE: the remove text if intent part needs to be moved to encode_state in
        # the single state featurizer because we work with the parsed message here
        # that still has the TEXT features even though we remove something from the
        # state here...

        return list_of_state_sequences

    def unfeaturized_trackers_for_training(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool,
    ) -> Tuple[List[List[State]], List[List[Text]]]:
        """Creates unfeaturized training data for e.g. rule/memoization policy.

        """
        list_of_state_sequences = self.extract_states_from_trackers_for_training(
            trackers, domain, omit_unset_slots=omit_unset_slots,
        )
        inputs = []
        outputs = []
        for state_sequence in list_of_state_sequences:
            inputs.append(state_sequence[:-1])
            last_action = state_sequence[-1][PREVIOUS_ACTION]
            last_action_name_or_text = (
                last_action.get(ACTION_NAME, None) or last_action[ACTION_TEXT]
            )
            outputs.append([last_action_name_or_text])
        return inputs, outputs

    def featurize_trackers_for_training(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        bilou_tagging: bool = False,
        e2e_features: Optional[Dict[Text, Message]] = None,
        targets_include_actions: bool = True,
        targets_include_entities: bool = True,
    ) -> Tuple[
        List[List[Dict[Text, List[Features]]]],
        np.ndarray,
        List[List[Dict[Text, List[Features]]]],
    ]:
        """Creates training data that contains `Features` from the NLU pipeline.

        Args:
            trackers: list of training trackers
            domain: the domain
            bilou_tagging: indicates whether BILOU tagging should be used or not

        Returns:
            - a dictionary of state types (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
              ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
              turns in all training trackers
            - the label ids (e.g. action ids) for every dialogue turn in all training
              trackers
            - A dictionary of entity type (ENTITY_TAGS) to a list of features
              containing entity tag ids for text user inputs otherwise empty dict
              for all dialogue turns in all training trackers
        """
        if self.state_featurizer is None:
            raise ValueError(
                f"Instance variable 'state_featurizer' is not set. "
                f"During initialization set 'state_featurizer' to an instance of "
                f"'{SingleStateFeaturizer.__class__.__name__}' class "
                f"to get numerical features for trackers."
            )

        self.state_featurizer.setup(domain, bilou_tagging)

        list_of_state_sequences: List[
            List[State]
        ] = self.extract_states_from_trackers_for_training(trackers, domain)

        inputs = []
        entities = []
        actions = []
        for state_sequence in list_of_state_sequences:
            inputs.append([])
            entities.append([])
            actions_for_state_sequence = np.zeros(len(state_sequence) - 1)
            for idx, state in enumerate(state_sequence):
                state_encoding = self.state_featurizer.encode_state(
                    state,
                    domain,
                    self.bilou_tagging,
                    e2e_features,
                    targets_include_actions=targets_include_actions,
                    targets_include_entities=targets_include_entities,
                )
                if idx < len(state_sequence) - 1:
                    inputs[-1].append(state_encoding["input"])
                if idx > 0:
                    if targets_include_entities:
                        entities[-1].append(state_encoding["target_entity"])
                    if targets_include_actions:
                        actions_for_state_sequence[idx - 1] = state_encoding[
                            "target_action"
                        ]
            if targets_include_actions:
                actions.append(np.array(actions_for_state_sequence))
        actions = np.array(actions)

        return (inputs, actions, entities)

    def _prepare_for_prediction(
        self, states: List[State], use_text_for_last_user_input: bool
    ) -> None:
        """Prepares a sequence of states that was extracted for prediction.

        """
        last_state = states[-1]

        # (1) only update the state of the real user utterance
        if not is_prev_action_listen_in_state(last_state):
            return

        # (2) keep either intent+entities or text only for the *last state*
        #     depending on the given flag
        if use_text_for_last_user_input:
            # remove intent features to only use text
            if last_state.get(USER, {}).get(INTENT):
                del last_state[USER][INTENT]
            # don't add entities if text is used for featurization
            if last_state.get(USER, {}).get(ENTITIES):
                del last_state[USER][ENTITIES]
        else:
            # remove text features to only use intent
            if last_state.get(USER, {}).get(TEXT):
                del last_state[USER][TEXT]

        # (3) for all other states, *always* remove the text if an intent is present

        # FIXME: if those messages passed through NLU pipeline before the policy
        # prediction pipeline, then we'll always have intents here? (unless it's for
        # an e2e policy)
        self._remove_user_text_if_intent(states)

    def extract_states_from_tracker_for_prediction(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[List[State]]:
        """Transforms a tracker to a lists of states for prediction.

        Args:
            tracker: The tracker to transform.
            domain: The domain.
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.

        Returns:
            A list of states.
        """
        states = tracker.past_states(
            domain,
            omit_unset_slots=False,
            ignore_rule_only_turns=ignore_rule_only_turns,
            rule_only_data=rule_only_data,
        )
        states = self.slice_state_history(states, self.max_history)
        # FIXME/TODO: why does this happen after slicing?
        self._prepare_for_prediction(states, use_text_for_last_user_input)
        return states

    def featurize_trackers_for_prediction(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        e2e_features: Dict[Text, Message],
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[List[Dict[Text, List[Features]]]]:
        """Create state features for prediction.

        Args:
            trackers: A list of state trackers
            domain: The domain
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.

        Returns:
            A list (corresponds to the list of trackers)
            of lists (corresponds to all dialogue turns)
            of dictionaries of state type (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
            ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
            turns in all trackers.
        """
        output = []
        for tracker in trackers:
            states = self.extract_states_from_tracker_for_prediction(
                tracker,
                domain,
                use_text_for_last_user_input,
                ignore_rule_only_turns,
                rule_only_data,
            )
            list_of_state_encodings = [
                self.state_featurizer.encode_state(
                    state,
                    domain,
                    self.bilou_tagging,
                    e2e_features,
                    targets_include_actions=False,
                    targets_include_entities=False,
                )
                for state in states
            ]
            output.append(
                [state_encoding["input"] for state_encoding in list_of_state_encodings]
            )
        return output

    def persist(self) -> None:
        """Persist the tracker featurizer to the given path.

        Args:
            path: The path to persist the tracker featurizer to.
        """
        featurizer_file = self._persistor.file_for(FEATURIZER_FILE)

        # entity tags are persisted in TED policy, they are not needed for prediction
        if self.state_featurizer is not None:
            self.state_featurizer.entity_tag_specs = None

        # noinspection PyTypeChecker
        rasa.shared.utils.io.write_text_file(
            str(jsonpickle.encode(self)), featurizer_file
        )

    @staticmethod
    def load(
        persistor: ComponentPersistorInterface, resource_name: Text
    ) -> Optional["TrackerFeaturizer"]:
        """Load the featurizer from file.

        Args:
            path: The path to load the tracker featurizer from.

        Returns:
            The loaded tracker featurizer.
        """
        featurizer_file = persistor.get_resource(resource_name, FEATURIZER_FILE)
        return jsonpickle.decode(rasa.shared.utils.io.read_file(featurizer_file))


# NOTE: FullDialogueFeaturizer wasn't used anymore and *Intent* version can be
# replaced by adding options to the above
