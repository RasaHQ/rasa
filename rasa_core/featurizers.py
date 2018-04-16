from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os
import typing
import json

import jsonpickle
import numpy as np
from typing import Tuple, List, Optional, Dict, Text
from builtins import str

from rasa_core import utils
from rasa_core.events import ActionExecuted
from rasa_core.training.data import DialogueTrainingData

from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.domain import PREV_PREFIX

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.domain import Domain


class SingleStateFeaturizer(object):
    """Base class for mechanisms to transform the conversations state
    into machine learning formats.

    Subclasses of SingleStateFeaturizer decide how the bot will transform
    the conversation state to a format which a classifier can read:
    feature vector."""

    def prepare_from_domain(self, domain):
        # type: (Domain) -> None
        """Helper method to init based on domain"""
        pass

    def encode(self, states):
        # type: (Optional[Text, float]) -> np.ndarray
        raise NotImplementedError("SingleStateFeaturizer must have "
                                  "the capacity to "
                                  "encode states to a feature vector")

    def encode_action(self, action, domain):
        # type: (Optional[Text, float], Domain) -> np.ndarray
        if action is None:
            return np.ones(domain.num_actions, dtype=int) * -1

        y = np.zeros(domain.num_actions, dtype=int)
        y[domain.index_for_action(action)] = 1
        return y


class BinarySingleStateFeaturizer(SingleStateFeaturizer):
    """Assumes all features are binary.

    All features should be either on or off, denoting them with 1 or 0."""

    def __init__(self):
        """Declares instant variables."""
        self.num_features = None
        self.input_state_map = None

    def prepare_from_domain(self, domain):
        # type: (Domain) -> None
        self.num_features = domain.num_states
        self.input_state_map = domain.input_state_map

    def encode(self, states):
        # type: (Optional[Text, float]) -> np.ndarray
        """Returns a binary vector indicating which features are active.

        Given a dictionary of states (e.g. 'intent_greet',
        'prev_action_listen',...) return a binary vector indicating which
        features of `self.input_features` are in the bag. NB it's a
        regular double precision float array type.

        For example with two active features out of five possible features
        this would return a vector like `[0 0 1 0 1]`

        If this is just a padding vector we set all values to `-1`.
        padding vectors are specified by a `None` or `[None]`
        value for states."""

        if not self.num_features:
            raise Exception("BinarySingleStateFeaturizer "
                            "was not prepared "
                            "before encoding.")

        if states is None or None in states:
            return np.ones(self.num_features, dtype=np.int32) * -1
        else:
            # we are going to use floats and convert to int later if possible
            used_features = np.zeros(self.num_features, dtype=float)
            using_only_ints = True
            best_intent = None
            best_intent_prob = 0.0

            for state_name, prob in states.items():
                if state_name.startswith('intent_'):
                    if prob >= best_intent_prob:
                        best_intent = state_name
                        best_intent_prob = prob
                elif state_name in self.input_state_map:
                    if prob != 0.0:
                        idx = self.input_state_map[state_name]
                        used_features[idx] = prob
                        using_only_ints = using_only_ints and utils.is_int(prob)
                else:
                    logger.debug(
                            "Feature '{}' (value: '{}') could not be found in "
                            "feature map. Make sure you added all intents and "
                            "entities to the domain".format(state_name, prob))

            if best_intent is not None:
                # finding the maximum confidence intent and
                # appending it to the states val
                index_in_feature_list = self.input_state_map.get(best_intent)
                if index_in_feature_list is not None:
                    used_features[index_in_feature_list] = 1
                else:
                    logger.warning(
                            "Couldn't set most probable feature '{}', "
                            "it wasn't found in the feature list of the domain."
                            " Make sure you added all intents and "
                            "entities to the domain.".format(best_intent))

            if using_only_ints:
                # this is an optimization - saves us a bit of memory
                return used_features.astype(np.int32)
            else:
                return used_features


class ProbabilisticSingleStateFeaturizer(BinarySingleStateFeaturizer):
    """Uses intent probabilities of the NLU and feeds them into the model."""

    def encode(self, states):
        # type: (Optional[Text, float]) -> np.ndarray
        """Returns a binary vector indicating active features,
        but with intent features given with a probability.

        Given a dictionary of states (e.g. 'intent_greet',
        'prev_action_listen',...) and intent probabilities
        from rasa_nlu, will be a binary vector indicating which features
        of `self.input_features` are active.

        For example with two active features and two uncertain intents out
        of five possible features this would return a vector
        like `[0.3, 0.7, 1, 0, 1]`.

        If this is just a padding vector we set all values to `-1`.
        padding vectors are specified by a `None` or `[None]`
        value for states."""

        if not self.num_features:
            raise Exception("ProbabilisticSingleStateFeaturizer "
                            "was not prepared "
                            "before encoding.")

        if states is None or None in states:
            return np.ones(self.num_features, dtype=np.int32) * -1
        else:

            used_features = np.zeros(self.num_features, dtype=np.float)
            for state, value in states.items():
                if state in self.input_state_map:
                    idx = self.input_state_map[state]
                    used_features[idx] = value
                else:
                    logger.debug(
                            "Found feature not in feature map. "
                            "Name: {} Value: {}".format(state, value))
            return used_features


class LabelTokenizerSingleStateFeaturizer(SingleStateFeaturizer):
    """SingleStateFeaturizer that splits user intents and
    bot action names into tokens and uses these tokens to
    create bag-of-words feature vectors.

    :param Text split_symbol:
      The symbol that separates words in intets and action names.

    :param bool share_vocab:
      The flag that specifies if to create the same vocabulary for
      user intents and bot actions."""

    def __init__(self, split_symbol='_', share_vocab=False):
        # type: (Text, bool) -> None
        """inits vocabulary for label bag of words representation"""

        self.share_vocab = share_vocab
        self.split_symbol = split_symbol

        self.num_features = None
        self.user_labels = []
        self.bot_labels = []
        self.other_labels = []

        self.bot_vocab = None
        self.user_vocab = None

    @staticmethod
    def _create_label_token_dict(labels, split_symbol='_'):
        """Splits labels into tokens by using provided symbol.
        Creates the lookup dictionary for this tokens.
        Values in this dict are used for featurization."""

        distinct_tokens = set([token
                               for label in labels
                               for token in label.split(split_symbol)])
        return {token: idx
                for idx, token in enumerate(sorted(distinct_tokens))}

    def prepare_from_domain(self, domain):
        # type: (Domain) -> None
        """Creates internal vocabularies for user intents
        and bot actions to use for featurization"""
        self.user_labels = domain.intent_states + domain.entity_states
        self.bot_labels = domain.action_names
        self.other_labels = domain.slot_states

        if self.share_vocab:
            self.bot_vocab = self._create_label_token_dict(self.bot_labels +
                                                           self.user_labels,
                                                           self.split_symbol)
            self.user_vocab = self.bot_vocab
        else:
            self.bot_vocab = self._create_label_token_dict(self.bot_labels,
                                                           self.split_symbol)
            self.user_vocab = self._create_label_token_dict(self.user_labels,
                                                            self.split_symbol)

        self.num_features = (len(self.user_vocab) +
                             len(self.bot_vocab) +
                             len(self.other_labels))

    def encode(self, states):
        # type: (Optional[Text, float]) -> np.ndarray
        if not self.num_features:
            raise Exception("LabelTokenizerSingleStateFeaturizer "
                            "was not prepared "
                            "before encoding.")

        if states is None or None in states:
            return np.ones(self.num_features, dtype=int) * -1

        used_features = np.zeros(self.num_features, dtype=int)
        for state_name, prob in states.items():

            if state_name in self.user_labels:
                if PREV_PREFIX + ACTION_LISTEN_NAME not in states:
                    # we predict next action from bot action
                    # TODO do we need state_name = 'intent_listen' ?
                    used_features[:len(self.user_vocab)] = 0
                else:
                    for t in state_name.split(self.split_symbol):
                        used_features[self.user_vocab[t]] += 1

            elif state_name[len(PREV_PREFIX):] in self.bot_labels:
                for t in state_name[len(PREV_PREFIX):].split(self.split_symbol):
                    used_features[len(self.user_vocab) +
                                  self.bot_vocab[t]] += 1

            elif state_name in self.other_labels:
                idx = (len(self.user_vocab) +
                       len(self.bot_vocab) +
                       self.other_labels.index(state_name))
                used_features[idx] += 1
            else:
                logger.warning(
                    "Feature '{}' could not be found in "
                    "feature map.".format(state_name))

        return used_features


class TrackerFeaturizer(object):
    """Base class for actual tracker featurizers"""
    def __init__(self, state_featurizer=None):
        # type: (Optional[SingleStateFeaturizer]) -> None

        self.state_featurizer = state_featurizer or SingleStateFeaturizer()

    def _pad_states(self, states):
        return states

    def _featurize_states(self, trackers_as_states):
        """Create X"""
        features = []
        true_lengths = []

        for tracker_states in trackers_as_states:
            if len(trackers_as_states) > 1:
                tracker_states = self._pad_states(tracker_states)
            dialogue_len = len(tracker_states)

            story_features = [self.state_featurizer.encode(state)
                              for state in tracker_states]

            features.append(story_features)
            true_lengths.append(dialogue_len)

        X = np.array(features)

        return X, true_lengths

    def _featurize_labels(self, trackers_as_actions, domain):
        """Create y"""

        labels = []
        for tracker_actions in trackers_as_actions:

            if len(trackers_as_actions) > 1:
                tracker_actions = self._pad_states(tracker_actions)

            story_labels = [self.state_featurizer.encode_action(action,
                                                                domain)
                            for action in tracker_actions]

            labels.append(story_labels)

        # if it is MaxHistoryFeaturizer, squeeze out time axis
        y = np.array(labels).squeeze()

        return y

    def training_states_and_actions(
            self,
            trackers,  # type: List[DialogueStateTracker]
            domain  # type: Domain
    ):
        # type: (...) -> Tuple[List[List[Dict]], List[List[Dict]], Dict]
        """Transforms list of trackers to lists of states and actions"""
        raise NotImplementedError("Featurizer must have the capacity to "
                                  "encode trackers to feature vectors")

    def featurize_trackers(self,
                           trackers,  # type: List[DialogueStateTracker]
                           domain  # type: Domain
                           ):
        # type: (...) -> DialogueTrainingData
        """Create training data"""
        self.state_featurizer.prepare_from_domain(domain)

        (trackers_as_states,
         trackers_as_actions) = self.training_states_and_actions(trackers, domain)

        X, true_lengths = self._featurize_states(trackers_as_states)
        y = self._featurize_labels(trackers_as_actions, domain)

        return DialogueTrainingData(X, y, true_lengths)

    def prediction_states(self,
                          trackers,  # type: List[DialogueStateTracker]
                          domain  # type: Domain
                          ):
        # type: (...) -> List[List[Dict[Text, float]]]
        """Transforms list of trackers to lists of states for prediction"""
        raise NotImplementedError("Featurizer must have the capacity to "
                                  "create feature vector")

    def create_X(self,
                 trackers,  # type: List[DialogueStateTracker]
                 domain  # type: Domain
                 ):
        # type: (...) -> Tuple[np.ndarray, List[int]]
        """Create X for prediction"""

        trackers_as_states = self.prediction_states(trackers, domain)
        X, true_lengths = self._featurize_states(trackers_as_states)
        return X, true_lengths

    def persist(self, path):
        featurizer_file = os.path.join(path, "featurizer.json")
        utils.create_dir_for_file(featurizer_file)
        with io.open(featurizer_file, 'w') as f:
            f.write(str(jsonpickle.encode(self)))

    @staticmethod
    def load(path):
        featurizer_file = os.path.join(path, "featurizer.json")
        if os.path.isfile(featurizer_file):
            with io.open(featurizer_file, 'r') as f:
                _json = f.read()
            return jsonpickle.decode(_json)
        else:
            logger.info("Couldn't load featurizer for policy. "
                        "File '{}' doesn't exist.".format(featurizer_file))
            return None


class FullDialogueTrackerFeaturizer(TrackerFeaturizer):
    """Tracker featurizer that takes the trackers
    and creates full dialogue training data for
    time distributed rnn.
    Training data is padded up to the length of the longest
    dialogue with -1"""

    def __init__(self, state_featurizer):
        # type: (SingleStateFeaturizer) -> None
        super(FullDialogueTrackerFeaturizer, self).__init__(state_featurizer)

        self.max_len = None

    @staticmethod
    def _calculate_max_len(trackers_as_actions):
        if trackers_as_actions:
            return max([len(states) for states in trackers_as_actions])
        else:
            return None

    def _pad_states(self, states):
        # pad up to max_len
        if len(states) < self.max_len:
            states += [None] * (self.max_len - len(states))

        return states

    def training_states_and_actions(
            self,
            trackers,  # type: List[DialogueStateTracker]
            domain  # type: Domain
    ):
        # type: (...) -> Tuple[List[List[Dict]], List[List[Dict]]]

        trackers_as_states = []
        trackers_as_actions = []

        for tracker in trackers:
            states = domain.states_for_tracker_history(tracker)

            delete_first_state = False
            actions = []
            for event in tracker._applied_events():
                if isinstance(event, ActionExecuted):
                    if not event.unpredictable:
                        # only actions which can be predicted at a stories start
                        actions.append(event.action_name)
                    else:
                        # unpredictable actions can be only the first in the story
                        if delete_first_state:
                            raise Exception("Found two unpredictable "
                                            "actions in one story."
                                            "Check your story files.")
                        else:
                            delete_first_state = True

            if delete_first_state:
                states = states[1:]

            trackers_as_states.append(states[:-1])
            trackers_as_actions.append(actions)

        self.max_len = self._calculate_max_len(trackers_as_actions)
        logger.info("The longest dialogue has {} actions."
                    "".format(self.max_len))

        return trackers_as_states, trackers_as_actions

    def prediction_states(self,
                          trackers,  # type: List[DialogueStateTracker]
                          domain  # type: Domain
                          ):
        # type: (...) -> List[List[Dict[Text, float]]]

        trackers_as_states = [domain.states_for_tracker_history(tracker)
                              for tracker in trackers]

        return trackers_as_states


class MaxHistoryTrackerFeaturizer(TrackerFeaturizer):
    """Tracker featurizer that takes the trackers,
    slices them into max_history batches and
    creates  training data for rnn that uses last output
    for prediction.
    Training data is padded up to the max_history with -1"""

    def __init__(self, state_featurizer=None,
                 max_history=5, remove_duplicates=True):
        # type: (Optional(SingleStateFeaturizer), int, bool) -> None
        super(MaxHistoryTrackerFeaturizer, self).__init__(state_featurizer)

        self.max_history = max_history

        self.remove_duplicates = remove_duplicates

    @staticmethod
    def slice_state_history(
            states,  # type: List[Dict[Text, float]]
            slice_length  # type: int
    ):
        # type: (...) -> List[Optional[Dict[Text, float]]]
        """Slices states from the trackers history.

        If the slice is at the array borders, padding will be added to ensure
        the slice length."""

        slice_end = len(states)
        slice_start = max(0, slice_end - slice_length)
        padding = [None] * max(0, slice_length - slice_end)
        state_features = padding + states[slice_start:]
        return state_features

    def training_states_and_actions(
            self,
            trackers,  # type: List[DialogueStateTracker]
            domain  # type: Domain
    ):
        # type: (...) -> Tuple[List[List[Dict]], List[List[Dict]]]

        trackers_as_states = []
        trackers_as_actions = []

        for tracker in trackers:
            states = domain.states_for_tracker_history(tracker)
            idx = 0
            for event in tracker._applied_events():
                if isinstance(event, ActionExecuted):
                    if not event.unpredictable:
                        # only actions which can be predicted at a stories start
                        # TODO unite with padding
                        sliced_states = self.slice_state_history(
                            states[:idx + 1], self.max_history)
                        trackers_as_states.append(sliced_states)
                        trackers_as_actions.append([event.action_name])
                    idx += 1

        if self.remove_duplicates:
            logger.debug("Got {} action examples."
                         "".format(len(trackers_as_actions)))
            (trackers_as_states,
             trackers_as_actions) = self._remove_duplicate_states(
                                        trackers_as_states,
                                        trackers_as_actions)
            logger.debug("Deduplicated to {} unique action examples."
                         "".format(len(trackers_as_actions)))

        return trackers_as_states, trackers_as_actions

    def prediction_states(self,
                          trackers,  # type: List[DialogueStateTracker]
                          domain  # type: Domain
                          ):
        # type: (...) -> List[List[Dict[Text, float]]]

        trackers_as_states = [domain.states_for_tracker_history(tracker)
                              for tracker in trackers]
        trackers_as_states = [self.slice_state_history(states,
                                                       self.max_history)
                              for states in trackers_as_states]

        return trackers_as_states

    @staticmethod
    def _remove_duplicate_states(
            trackers_as_states,  # type: List[List[Dict[Text, float]]]
            trackers_as_actions,  # type: List[List[Dict[Text, float]]]
    ):
        # type: (...) -> Tuple[List[List[Dict]], List[List[Dict]]]
        """Removes states that create equal featurizations.

        From multiple states that create equal featurizations
        we only need to keep one."""

        hashed_featurizations = set()

        # collected trackers_as_states that created different featurizations
        unique_trackers_as_states = []
        unique_trackers_as_actions = []

        for (tracker_states,
             tracker_actions) in zip(trackers_as_states,
                                     trackers_as_actions):

            states = tracker_states + [tracker_actions]
            hashed = json.dumps(states, sort_keys=True)

            # only continue with tracker_states that created a
            # hashed_featurization we haven't observed
            if hashed not in hashed_featurizations:
                hashed_featurizations.add(hashed)
                unique_trackers_as_states.append(tracker_states)
                unique_trackers_as_actions.append(tracker_actions)

        return unique_trackers_as_states, unique_trackers_as_actions
