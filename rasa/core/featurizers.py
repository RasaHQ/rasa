import io
import jsonpickle
import logging
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Text, Any

import rasa.utils.io
from rasa.core import utils
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.domain import PREV_PREFIX, Domain
from rasa.core.events import ActionExecuted
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.data import DialogueTrainingData
from rasa.utils.common import is_logging_disabled

logger = logging.getLogger(__name__)


class SingleStateFeaturizer(object):
    """Base class for mechanisms to transform the conversations state
    into machine learning formats.

    Subclasses of SingleStateFeaturizer decide how the bot will transform
    the conversation state to a format which a classifier can read:
    feature vector."""

    def __init__(self):
        """Declares instant variables."""
        self.user_feature_len = None
        self.slot_feature_len = None

    def prepare_from_domain(self, domain: Domain) -> None:
        """Helper method to init based on domain"""
        pass

    def encode(self, state: Dict[Text, float]) -> np.ndarray:
        raise NotImplementedError(
            "SingleStateFeaturizer must have "
            "the capacity to "
            "encode states to a feature vector"
        )

    @staticmethod
    def action_as_one_hot(action: Text, domain: Domain) -> np.ndarray:
        if action is None:
            return np.ones(domain.num_actions, dtype=int) * -1

        y = np.zeros(domain.num_actions, dtype=int)
        y[domain.index_for_action(action)] = 1
        return y

    def create_encoded_all_actions(self, domain: Domain) -> np.ndarray:
        """Create matrix with all actions from domain
            encoded in rows."""
        pass


class BinarySingleStateFeaturizer(SingleStateFeaturizer):
    """Assumes all features are binary.

    All features should be either on or off, denoting them with 1 or 0."""

    def __init__(self):
        """Declares instant variables."""
        super(BinarySingleStateFeaturizer, self).__init__()

        self.num_features = None
        self.input_state_map = None

    def prepare_from_domain(self, domain: Domain) -> None:
        self.num_features = domain.num_states
        self.input_state_map = domain.input_state_map

        self.user_feature_len = len(domain.intent_states) + len(domain.entity_states)
        self.slot_feature_len = len(domain.slot_states)

    def encode(self, state: Dict[Text, float]) -> np.ndarray:
        """Returns a binary vector indicating which features are active.

            Given a dictionary of states (e.g. 'intent_greet',
            'prev_action_listen',...) return a binary vector indicating which
            features of `self.input_features` are in the bag. NB it's a
            regular double precision float array type.

            For example with two active features out of five possible features
            this would return a vector like `[0 0 1 0 1]`

            If intent features are given with a probability, for example
            with two active features and two uncertain intents out
            of five possible features this would return a vector
            like `[0.3, 0.7, 1.0, 0, 1.0]`.

            If this is just a padding vector we set all values to `-1`.
            padding vectors are specified by a `None` or `[None]`
            value for states.
        """

        if not self.num_features:
            raise Exception(
                "BinarySingleStateFeaturizer was not prepared before encoding."
            )

        if state is None or None in state:
            return np.ones(self.num_features, dtype=np.int32) * -1

        # we are going to use floats and convert to int later if possible
        used_features = np.zeros(self.num_features, dtype=np.float)
        using_only_ints = True
        for state_name, prob in state.items():
            if state_name in self.input_state_map:
                idx = self.input_state_map[state_name]
                used_features[idx] = prob
                using_only_ints = using_only_ints and utils.is_int(prob)
            else:
                logger.debug(
                    "Feature '{}' (value: '{}') could not be found in "
                    "feature map. Make sure you added all intents and "
                    "entities to the domain".format(state_name, prob)
                )

        if using_only_ints:
            # this is an optimization - saves us a bit of memory
            return used_features.astype(np.int32)
        else:
            return used_features

    def create_encoded_all_actions(self, domain: Domain) -> np.ndarray:
        """Create matrix with all actions from domain
            encoded in rows as bag of words."""
        return np.eye(domain.num_actions)


class LabelTokenizerSingleStateFeaturizer(SingleStateFeaturizer):
    """SingleStateFeaturizer that splits user intents and
    bot action names into tokens and uses these tokens to
    create bag-of-words feature vectors.

    Args:
        split_symbol: The symbol that separates words in
            intets and action names.

        use_shared_vocab: The flag that specifies if to create
            the same vocabulary for user intents and bot actions.
    """

    def __init__(
        self, use_shared_vocab: bool = False, split_symbol: Text = "_"
    ) -> None:
        """inits vocabulary for label bag of words representation"""
        super(LabelTokenizerSingleStateFeaturizer, self).__init__()

        self.use_shared_vocab = use_shared_vocab
        self.split_symbol = split_symbol

        self.num_features = None
        self.user_labels = []
        self.slot_labels = []
        self.bot_labels = []

        self.bot_vocab = None
        self.user_vocab = None

    @staticmethod
    def _create_label_token_dict(labels, split_symbol="_"):
        """Splits labels into tokens by using provided symbol.
        Creates the lookup dictionary for this tokens.
        Values in this dict are used for featurization."""

        distinct_tokens = set(
            [token for label in labels for token in label.split(split_symbol)]
        )
        return {token: idx for idx, token in enumerate(sorted(distinct_tokens))}

    def prepare_from_domain(self, domain: Domain) -> None:
        """Creates internal vocabularies for user intents
        and bot actions to use for featurization"""
        self.user_labels = domain.intent_states + domain.entity_states
        self.slot_labels = domain.slot_states
        self.bot_labels = domain.action_names

        if self.use_shared_vocab:
            self.bot_vocab = self._create_label_token_dict(
                self.bot_labels + self.user_labels, self.split_symbol
            )
            self.user_vocab = self.bot_vocab
        else:
            self.bot_vocab = self._create_label_token_dict(
                self.bot_labels, self.split_symbol
            )
            self.user_vocab = self._create_label_token_dict(
                self.user_labels, self.split_symbol
            )

        self.num_features = (
            len(self.user_vocab) + len(self.slot_labels) + len(self.bot_vocab)
        )

        self.user_feature_len = len(self.user_vocab)
        self.slot_feature_len = len(self.slot_labels)

    def encode(self, state: Dict[Text, float]) -> np.ndarray:
        if not self.num_features:
            raise Exception(
                "LabelTokenizerSingleStateFeaturizer "
                "was not prepared before encoding."
            )

        if state is None or None in state:
            return np.ones(self.num_features, dtype=np.int32) * -1

        # we are going to use floats and convert to int later if possible
        used_features = np.zeros(self.num_features, dtype=np.float)
        using_only_ints = True
        for state_name, prob in state.items():
            using_only_ints = using_only_ints and utils.is_int(prob)
            if state_name in self.user_labels:
                if PREV_PREFIX + ACTION_LISTEN_NAME in state:
                    # else we predict next action from bot action and memory
                    for t in state_name.split(self.split_symbol):
                        used_features[self.user_vocab[t]] += prob

            elif state_name in self.slot_labels:
                offset = len(self.user_vocab)
                idx = self.slot_labels.index(state_name)
                used_features[offset + idx] += prob

            elif state_name[len(PREV_PREFIX) :] in self.bot_labels:
                action_name = state_name[len(PREV_PREFIX) :]
                for t in action_name.split(self.split_symbol):
                    offset = len(self.user_vocab) + len(self.slot_labels)
                    idx = self.bot_vocab[t]
                    used_features[offset + idx] += prob

            else:
                logger.warning(
                    "Feature '{}' could not be found in "
                    "feature map.".format(state_name)
                )

        if using_only_ints:
            # this is an optimization - saves us a bit of memory
            return used_features.astype(np.int32)
        else:
            return used_features

    def create_encoded_all_actions(self, domain: Domain) -> np.ndarray:
        """Create matrix with all actions from domain
            encoded in rows as bag of words."""
        encoded_all_actions = np.zeros(
            (domain.num_actions, len(self.bot_vocab)), dtype=int
        )
        for idx, name in enumerate(domain.action_names):
            for t in name.split(self.split_symbol):
                encoded_all_actions[idx, self.bot_vocab[t]] = 1
        return encoded_all_actions


class TrackerFeaturizer(object):
    """Base class for actual tracker featurizers"""

    def __init__(
        self,
        state_featurizer: Optional[SingleStateFeaturizer] = None,
        use_intent_probabilities: bool = False,
    ) -> None:

        self.state_featurizer = state_featurizer or SingleStateFeaturizer()
        self.use_intent_probabilities = use_intent_probabilities

    def _create_states(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        is_binary_training: bool = False,
    ) -> List[Dict[Text, float]]:
        """Create states: a list of dictionaries.
            If use_intent_probabilities is False (default behaviour),
            pick the most probable intent out of all provided ones and
            set its probability to 1.0, while all the others to 0.0."""
        states = tracker.past_states(domain)

        # during training we encounter only 1 or 0
        if not self.use_intent_probabilities and not is_binary_training:
            bin_states = []
            for state in states:
                # copy state dict to preserve internal order of keys
                bin_state = dict(state)
                best_intent = None
                best_intent_prob = -1.0
                for state_name, prob in state:
                    if state_name.startswith("intent_"):
                        if prob > best_intent_prob:
                            # finding the maximum confidence intent
                            if best_intent is not None:
                                # delete previous best intent
                                del bin_state[best_intent]
                            best_intent = state_name
                            best_intent_prob = prob
                        else:
                            # delete other intents
                            del bin_state[state_name]

                if best_intent is not None:
                    # set the confidence of best intent to 1.0
                    bin_state[best_intent] = 1.0

                bin_states.append(bin_state)
            return bin_states
        else:
            return [dict(state) for state in states]

    def _pad_states(self, states: List[Any]) -> List[Any]:
        return states

    def _featurize_states(
        self, trackers_as_states: List[List[Dict[Text, float]]]
    ) -> Tuple[np.ndarray, List[int]]:
        """Create X"""
        features = []
        true_lengths = []

        for tracker_states in trackers_as_states:
            dialogue_len = len(tracker_states)

            # len(trackers_as_states) = 1 means
            # it is called during prediction or we have
            # only one story, so no padding is needed

            if len(trackers_as_states) > 1:
                tracker_states = self._pad_states(tracker_states)

            story_features = [
                self.state_featurizer.encode(state) for state in tracker_states
            ]

            features.append(story_features)
            true_lengths.append(dialogue_len)

        # noinspection PyPep8Naming
        X = np.array(features)

        return X, true_lengths

    def _featurize_labels(
        self, trackers_as_actions: List[List[Text]], domain: Domain
    ) -> np.ndarray:
        """Create y"""

        labels = []
        for tracker_actions in trackers_as_actions:

            if len(trackers_as_actions) > 1:
                tracker_actions = self._pad_states(tracker_actions)

            story_labels = [
                self.state_featurizer.action_as_one_hot(action, domain)
                for action in tracker_actions
            ]

            labels.append(story_labels)

        # if it is MaxHistoryFeaturizer, squeeze out time axis
        y = np.array(labels).squeeze()

        return y

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[Dict]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and actions"""
        raise NotImplementedError(
            "Featurizer must have the capacity to encode trackers to feature vectors"
        )

    def featurize_trackers(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> DialogueTrainingData:
        """Create training data"""
        self.state_featurizer.prepare_from_domain(domain)

        (trackers_as_states, trackers_as_actions) = self.training_states_and_actions(
            trackers, domain
        )

        # noinspection PyPep8Naming
        X, true_lengths = self._featurize_states(trackers_as_states)
        y = self._featurize_labels(trackers_as_actions, domain)

        return DialogueTrainingData(X, y, true_lengths)

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[Dict[Text, float]]]:
        """Transforms list of trackers to lists of states for prediction"""
        raise NotImplementedError(
            "Featurizer must have the capacity to create feature vector"
        )

    # noinspection PyPep8Naming
    def create_X(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> np.ndarray:
        """Create X for prediction"""

        trackers_as_states = self.prediction_states(trackers, domain)
        X, _ = self._featurize_states(trackers_as_states)
        return X

    def persist(self, path):
        featurizer_file = os.path.join(path, "featurizer.json")
        utils.create_dir_for_file(featurizer_file)
        with open(featurizer_file, "w", encoding="utf-8") as f:
            # noinspection PyTypeChecker
            f.write(str(jsonpickle.encode(self)))

    @staticmethod
    def load(path):
        featurizer_file = os.path.join(path, "featurizer.json")
        if os.path.isfile(featurizer_file):
            return jsonpickle.decode(rasa.utils.io.read_file(featurizer_file))
        else:
            logger.error(
                "Couldn't load featurizer for policy. "
                "File '{}' doesn't exist.".format(featurizer_file)
            )
            return None


class FullDialogueTrackerFeaturizer(TrackerFeaturizer):
    """Tracker featurizer that takes the trackers
    and creates full dialogue training data for
    time distributed rnn.
    Training data is padded up to the length of the longest
    dialogue with -1"""

    def __init__(
        self,
        state_featurizer: SingleStateFeaturizer,
        use_intent_probabilities: bool = False,
    ) -> None:
        super(FullDialogueTrackerFeaturizer, self).__init__(
            state_featurizer, use_intent_probabilities
        )
        self.max_len = None

    @staticmethod
    def _calculate_max_len(trackers_as_actions):
        if trackers_as_actions:
            return max([len(states) for states in trackers_as_actions])
        else:
            return None

    def _pad_states(self, states: List[Any]) -> List[Any]:
        """Pads states up to max_len"""

        if len(states) < self.max_len:
            states += [None] * (self.max_len - len(states))

        return states

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[Dict]], List[List[Text]]]:

        trackers_as_states = []
        trackers_as_actions = []

        logger.debug(
            "Creating states and action examples from "
            "collected trackers (by {}({}))..."
            "".format(type(self).__name__, type(self.state_featurizer).__name__)
        )
        pbar = tqdm(trackers, desc="Processed trackers", disable=is_logging_disabled())
        for tracker in pbar:
            states = self._create_states(tracker, domain, is_binary_training=True)

            delete_first_state = False
            actions = []
            for event in tracker.applied_events():
                if isinstance(event, ActionExecuted):
                    if not event.unpredictable:
                        # only actions which can be
                        # predicted at a stories start
                        actions.append(event.action_name)
                    else:
                        # unpredictable actions can be
                        # only the first in the story
                        if delete_first_state:
                            raise Exception(
                                "Found two unpredictable "
                                "actions in one story."
                                "Check your story files."
                            )
                        else:
                            delete_first_state = True

            if delete_first_state:
                states = states[1:]

            trackers_as_states.append(states[:-1])
            trackers_as_actions.append(actions)

        self.max_len = self._calculate_max_len(trackers_as_actions)
        logger.debug("The longest dialogue has {} actions.".format(self.max_len))

        return trackers_as_states, trackers_as_actions

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[Dict[Text, float]]]:

        trackers_as_states = [
            self._create_states(tracker, domain) for tracker in trackers
        ]

        return trackers_as_states


class MaxHistoryTrackerFeaturizer(TrackerFeaturizer):
    """Tracker featurizer that takes the trackers,
    slices them into max_history batches and
    creates  training data for rnn that uses last output
    for prediction.
    Training data is padded up to the max_history with -1"""

    MAX_HISTORY_DEFAULT = 5

    def __init__(
        self,
        state_featurizer: Optional[SingleStateFeaturizer] = None,
        max_history: int = None,
        remove_duplicates: bool = True,
        use_intent_probabilities: bool = False,
    ) -> None:
        super(MaxHistoryTrackerFeaturizer, self).__init__(
            state_featurizer, use_intent_probabilities
        )
        self.max_history = max_history or self.MAX_HISTORY_DEFAULT
        self.remove_duplicates = remove_duplicates

    @staticmethod
    def slice_state_history(
        states: List[Dict[Text, float]], slice_length: int
    ) -> List[Optional[Dict[Text, float]]]:
        """Slices states from the trackers history.

        If the slice is at the array borders, padding will be added to ensure
        the slice length."""

        slice_end = len(states)
        slice_start = max(0, slice_end - slice_length)
        padding = [None] * max(0, slice_length - slice_end)
        # noinspection PyTypeChecker
        state_features = padding + states[slice_start:]
        return state_features

    @staticmethod
    def _hash_example(states, action):
        frozen_states = tuple(
            (s if s is None else frozenset(s.items()) for s in states)
        )
        frozen_actions = (action,)
        return hash((frozen_states, frozen_actions))

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[Dict]], List[List[Text]]]:

        trackers_as_states = []
        trackers_as_actions = []

        # from multiple states that create equal featurizations
        # we only need to keep one.
        hashed_examples = set()

        logger.debug(
            "Creating states and action examples from "
            "collected trackers (by {}({}))..."
            "".format(type(self).__name__, type(self.state_featurizer).__name__)
        )
        pbar = tqdm(trackers, desc="Processed trackers", disable=is_logging_disabled())
        for tracker in pbar:
            states = self._create_states(tracker, domain, True)

            idx = 0
            for event in tracker.applied_events():
                if isinstance(event, ActionExecuted):
                    if not event.unpredictable:
                        # only actions which can be
                        # predicted at a stories start
                        sliced_states = self.slice_state_history(
                            states[: idx + 1], self.max_history
                        )

                        if self.remove_duplicates:
                            hashed = self._hash_example(
                                sliced_states, event.action_name
                            )

                            # only continue with tracker_states that created a
                            # hashed_featurization we haven't observed
                            if hashed not in hashed_examples:
                                hashed_examples.add(hashed)
                                trackers_as_states.append(sliced_states)
                                trackers_as_actions.append([event.action_name])
                        else:
                            trackers_as_states.append(sliced_states)
                            trackers_as_actions.append([event.action_name])

                        pbar.set_postfix(
                            {"# actions": "{:d}".format(len(trackers_as_actions))}
                        )
                    idx += 1

        logger.debug("Created {} action examples.".format(len(trackers_as_actions)))

        return trackers_as_states, trackers_as_actions

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[Dict[Text, float]]]:

        trackers_as_states = [
            self._create_states(tracker, domain) for tracker in trackers
        ]
        trackers_as_states = [
            self.slice_state_history(states, self.max_history)
            for states in trackers_as_states
        ]

        return trackers_as_states
