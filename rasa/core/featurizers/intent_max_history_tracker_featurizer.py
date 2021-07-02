from typing import List, Text, Tuple, Dict, Any, Iterator, Optional
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import logging

from rasa.core.featurizers.tracker_featurizers import (
    MaxHistoryTrackerFeaturizer,
    is_prev_action_listen_in_state,
)
from rasa.shared.core.domain import State, Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.tensorflow.constants import LABEL_PAD_ID
from rasa.shared.core.events import ActionExecuted, UserUttered
import rasa.shared.utils.io

logger = logging.getLogger(__name__)


class IntentMaxHistoryTrackerFeaturizer(MaxHistoryTrackerFeaturizer):
    """Truncates the tracker history into `max_history` long sequences.

    Creates training data from trackers where intents are the output prediction
    labels. Tracker state sequences which represent policy input are truncated
    to not excede `max_history` states.
    """

    LABEL_NAME = "intent"

    @classmethod
    def _convert_labels_to_ids(
        cls, trackers_as_intents: List[List[Text]], domain: Domain
    ) -> np.ndarray:
        """Converts a list of labels to a matrix of label ids.

        The number of rows is equal to `len(trackers_as_intents)`. The number of
        columns is equal to the maximum number of positive labels that any training
        example is associated with. Rows are padded with `LABEL_PAD_ID` if not all rows
        have the same number of labels.

        Args:
            trackers_as_intents: Positive example label ids
                associated with each training example.
            domain: The domain of the training data.

        Returns:
           A matrix of label ids.
        """
        # store labels in numpy arrays so that it corresponds to np arrays
        # of input features
        label_ids = [
            [domain.intents.index(intent) for intent in tracker_intents]
            for tracker_intents in trackers_as_intents
        ]

        return np.array(cls._pad_label_ids(label_ids))

    @staticmethod
    def _pad_label_ids(label_ids: List[List[int]]) -> List[List[int]]:
        """Pads label ids so that all are of the same length.

        The pad value `LABEL_PAD_ID` is set in `rasa.utils.tensorflow.constants`.

        Args:
            label_ids: Label ids of varying lengths

        Returns:
            Label ids padded to be of uniform length.
        """
        # If `label_ids` is an empty list, no padding needs to be added.
        if not label_ids:
            return label_ids

        # Add `LABEL_PAD_ID` padding to labels array so that
        # each example has equal number of labels
        multiple_labels_count = [len(a) for a in label_ids]
        max_labels_count = max(multiple_labels_count)
        num_padding_needed = [max_labels_count - len(a) for a in label_ids]

        padded_label_ids = []
        for ids, num_pads in zip(label_ids, num_padding_needed):
            padded_row = list(ids) + [LABEL_PAD_ID] * num_pads
            padded_label_ids.append(padded_row)
        return padded_label_ids

    def training_states_labels_and_entities(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
        ignore_action_unlikely_intent: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]:
        """Transforms trackers to states, intent labels, and entity data.

        Args:
            trackers: The trackers to transform.
            domain: The domain.
            omit_unset_slots: If `True` do not include the initial values of slots.
            ignore_action_unlikely_intent: Whether to remove `action_unlikely_intent`
                from training states.

        Returns:
            Trackers as states, labels, and entity data.
        """
        example_states = []
        example_labels = []
        example_entities = []

        # Store of example hashes for removing duplicate training examples.
        hashed_examples = set()
        # Mapping of example state hash to list of positive labels associated with
        # the state. Note that each individual 'label' instance is a list of ints.
        state_hash_to_label_list_instances: defaultdict[
            int, List[List[int]]
        ] = defaultdict(list)

        logger.debug(
            f"Creating states and {self.LABEL_NAME} label examples from "
            f"collected trackers "
            f"(by {type(self).__name__}({type(self.state_featurizer).__name__}))..."
        )
        pbar = tqdm(
            trackers,
            desc="Processed trackers",
            disable=rasa.shared.utils.io.is_logging_disabled(),
        )
        for tracker in pbar:

            for states, label, entities in self._extract_examples(
                tracker,
                domain,
                omit_unset_slots=omit_unset_slots,
                ignore_action_unlikely_intent=ignore_action_unlikely_intent,
            ):

                if self.remove_duplicates:
                    hashed = self._hash_example(tracker, states, label)
                    if hashed in hashed_examples:
                        continue
                    hashed_examples.add(hashed)

                # Store all positive labels associated with a training state.
                state_hash = self._hash_example(tracker, states)
                state_hash_to_label_list_instances[state_hash].append(label)

                example_states.append(states)
                example_labels.append(label)
                example_entities.append(entities)

                pbar.set_postfix({f"# {self.LABEL_NAME}": f"{len(example_labels):d}"})

        self._remove_user_text_if_intent(example_states)

        # Collect all positive intent labels for a given state hash and add
        # the set back to the singleton labels.
        for positive_label_list in state_hash_to_label_list_instances.values():
            # Get the set of positive labels associated with each state hash.
            positive_label_set = set([labels[0] for labels in positive_label_list])

            for labels in positive_label_list:
                # Extend the singletone `labels` with the `positive_label_set`.
                # Note that we need to filter out the redundant label `labels[0]`
                # from `positive_label_set` so as not to add it twice.
                filtered_label_set = filter(
                    lambda label: label != labels[0], positive_label_set
                )
                labels.extend(filtered_label_set)

        logger.debug(f"Created {len(example_states)} {self.LABEL_NAME} examples.")

        return example_states, example_labels, example_entities

    def _extract_examples(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        omit_unset_slots: bool = False,
        ignore_action_unlikely_intent: bool = False,
    ) -> Iterator[Tuple[List[State], List[Text], List[Dict[Text, Any]]]]:
        """Creates an iterator over training examples from a tracker.

        Args:
            trackers: The tracker from which to extract training examples.
            domain: The domain of the training data.
            omit_unset_slots: If `True` do not include the initial values of slots.
            ignore_action_unlikely_intent: Whether to remove `action_unlikely_intent`
                from training states.

        Returns:
            An iterator over example states, labels, and entity data.
        """
        tracker_states = self._create_states(
            tracker, domain, omit_unset_slots=omit_unset_slots
        )
        events = tracker.applied_events()

        if ignore_action_unlikely_intent:
            tracker_states = self._remove_action_unlikely_intent_from_states(
                tracker_states
            )
            events = self._remove_action_unlikely_intent_from_events(events)

        label_index = 0
        for event in events:

            if isinstance(event, ActionExecuted):
                label_index += 1

            elif isinstance(event, UserUttered):

                sliced_states = self.slice_state_history(
                    tracker_states[:label_index], self.max_history
                )
                label = [event.intent_name or event.text]
                entities = [{}]

                yield sliced_states, label, entities

    @staticmethod
    def _cleanup_last_user_state_with_action_listen(
        trackers_as_states: List[List[State]],
    ) -> List[List[State]]:
        """Removes the last state in a tracker if the previous action is `action_listen`.

        States with the previous action equal to `action_listen` correspond to states with
        a new user intent. This information is what `UnexpecTEDIntentPolicy` is trying to
        predict so it needs to be removed before obtaining a prediction.

        Args:
            trackers_as_states: Trackers converted to states

        Returns:
            Filtered states with last `action_listen` removed.
        """
        for states in trackers_as_states:
            if not states:
                continue
            last_state = states[-1]
            if is_prev_action_listen_in_state(last_state):
                del states[-1]

        return trackers_as_states

    def prediction_states(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        ignore_action_unlikely_intent: bool = False,
    ) -> List[List[State]]:
        """Transforms trackers to states for prediction.

        Args:
            trackers: The trackers to transform.
            domain: The domain.
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.
            ignore_action_unlikely_intent: Whether to remove any states containing
                `action_unlikely_intent` from prediction states.

        Returns:
            Trackers as states for prediction.
        """
        trackers_as_states = [
            self._create_states(
                tracker,
                domain,
                ignore_rule_only_turns=ignore_rule_only_turns,
                rule_only_data=rule_only_data,
            )
            for tracker in trackers
        ]

        # Remove `action_unlikely_intent` from `trackers_as_states`.
        # This must be done before state history slicing to ensure the
        # max history of the sliced states matches training time.
        if ignore_action_unlikely_intent:
            trackers_as_states = [
                self._remove_action_unlikely_intent_from_states(states)
                for states in trackers_as_states
            ]

        self._choose_last_user_input(trackers_as_states, use_text_for_last_user_input)

        # `tracker_as_states` contain a state with intent = last intent
        # and previous action = action_listen. This state needs to be
        # removed as it was not present during training as well because
        # predicting the last intent is what the policies using this
        # featurizer do. This is specifically done before state history
        # is sliced so that the number of past states is same as `max_history`.
        self._cleanup_last_user_state_with_action_listen(trackers_as_states)

        trackers_as_states = [
            self.slice_state_history(states, self.max_history)
            for states in trackers_as_states
        ]

        return trackers_as_states
