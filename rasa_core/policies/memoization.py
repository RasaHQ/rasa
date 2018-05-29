from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import base64
import io
import json
import logging
import os
import zlib
import typing
from tqdm import tqdm

from builtins import bytes
from typing import Optional, Any, Dict, List, Text

from rasa_core.policies.policy import Policy
from rasa_core import utils
from rasa_core.featurizers import \
    TrackerFeaturizer, MaxHistoryTrackerFeaturizer

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.domain import Domain


class MemoizationPolicy(Policy):
    """The policy that remembers exact examples of
        `max_history` turns from training stories.

        Since `slots` that are set some time in the past are
        preserved in all future feature vectors until they are set
        to None, this policy implicitly remembers and most importantly
        recalls examples in the context of the current dialogue
        longer than `max_history`.

        If it is needed to recall turns from training dialogues where
        some slots might not be set during prediction time, and there are
        training stories for this, use AugmentedMemoizationPolicy.
    """

    ENABLE_FEATURE_STRING_COMPRESSION = True

    SUPPORTS_ONLINE_TRAINING = True

    USE_NLU_CONFIDENCE_AS_SCORE = False

    @classmethod
    def _standard_featurizer(cls, max_history=None):
        max_history = max_history or cls.MAX_HISTORY_DEFAULT
        # Memoization policy always uses MaxHistoryTrackerFeaturizer
        # without state_featurizer
        return MaxHistoryTrackerFeaturizer(state_featurizer=None,
                                           max_history=max_history,
                                           use_intent_probabilities=False)

    def __init__(self,
                 featurizer=None,  # type: Optional[TrackerFeaturizer]
                 max_history=None,  # type: Optional[int]
                 lookup=None  # type: Optional[Dict]
                 ):
        # type: (...) -> None

        if not featurizer:
            featurizer = self._standard_featurizer(max_history)

        super(MemoizationPolicy, self).__init__(featurizer)

        self.max_history = self.featurizer.max_history
        self.lookup = lookup if lookup is not None else {}
        self.is_enabled = True

    def toggle(self, activate):
        # type: (bool) -> None
        self.is_enabled = activate

    def _preprocess_states(self, states):
        # type: (List[Dict[Text, float]]) -> List[List[Dict[Text, float]]]
        """Helper method to preprocess tracker's states.
            E.g., to a create list of states with deleted history
            for augmented Memoization"""
        return [states]

    def _add(self, trackers_as_states, trackers_as_actions,
             domain, online=False):

        if not trackers_as_states:
            return

        assert len(trackers_as_states[0]) == self.max_history, \
            ("Trying to mem featurized data with {} historic turns. Expected: "
             "{}".format(len(trackers_as_states[0]), self.max_history))

        assert len(trackers_as_actions[0]) == 1, \
            ("The second dimension of trackers_as_action should be 1, "
             "instead of {}".format(len(trackers_as_actions[0])))

        ambiguous_feature_keys = set()

        pbar = tqdm(zip(trackers_as_states, trackers_as_actions),
                    desc="Processed actions", disable=online)
        for states, actions in pbar:
            action = actions[0]
            for i, states_aug in enumerate(self._preprocess_states(states)):
                feature_key = self._create_feature_key(states_aug)
                feature_item = domain.index_for_action(action)

                if feature_key not in ambiguous_feature_keys:
                    if feature_key in self.lookup.keys():
                        if self.lookup[feature_key] != feature_item:
                            if online and i == 0:
                                logger.info("Original stories are "
                                            "different for {} -- {}\n"
                                            "Memorized the new ones for "
                                            "now. Delete contradicting "
                                            "examples after exporting "
                                            "the new stories."
                                            "".format(states_aug, action))
                                self.lookup[feature_key] = feature_item
                            else:
                                # delete contradicting example created by
                                # partial history augmentation from memory
                                ambiguous_feature_keys.add(feature_key)
                                del self.lookup[feature_key]
                    else:
                        self.lookup[feature_key] = feature_item
                pbar.set_postfix({"# examples": "{:d}".format(
                                                    len(self.lookup))})

    def _create_feature_key(self, states):
        feature_str = json.dumps(states, sort_keys=True).replace("\"", "")
        if self.ENABLE_FEATURE_STRING_COMPRESSION:
            compressed = zlib.compress(bytes(feature_str, "utf-8"))
            return base64.b64encode(compressed).decode("utf-8")
        else:
            return feature_str

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: **Any
              ):
        # type: (...) -> None
        """Trains the policy on given training trackers."""
        self.lookup = {}

        (trackers_as_states,
         trackers_as_actions) = self.featurizer.training_states_and_actions(
                                    training_trackers, domain)
        self._add(trackers_as_states, trackers_as_actions, domain)
        logger.info("Memorized {} unique action examples."
                    "".format(len(self.lookup)))

    def continue_training(self, training_trackers, domain, **kwargs):
        # type: (List[DialogueStateTracker], Domain, **Any) -> None

        # add only the last tracker, because it is the only new one
        (trackers_as_states,
         trackers_as_actions) = self.featurizer.training_states_and_actions(
                                    training_trackers[-1:], domain)
        self._add(trackers_as_states, trackers_as_actions,
                  domain, online=True)

    def _recall_states(self, states):
        # type: (List[Dict[Text, float]]) -> Optional[int]

        for states_aug in self._preprocess_states(states):
            memorised = self.lookup.get(
                self._create_feature_key(states_aug))
            if memorised is not None:
                return memorised
        return None

    def recall(self,
               states,  # type: List[Dict[Text, float]]
               tracker,  # type: DialogueStateTracker
               domain  # type: Domain
               ):
        # type: (...) -> Optional[int]

        return self._recall_states(states)

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts the next action the bot should take
            after seeing the tracker.

            Returns the list of probabilities for the next actions.
            If memorized action was found returns 1.0 for its index,
            else returns 0.0 for all actions."""
        result = [0.0] * domain.num_actions

        if not self.is_enabled:
            return result

        tracker_as_states = self.featurizer.prediction_states(
                                [tracker], domain)
        states = tracker_as_states[0]
        logger.debug("Current tracker state {}".format(states))
        recalled = self.recall(states, tracker, domain)
        if recalled is not None:
            logger.debug("There is a memorised next action '{}'"
                         "".format(recalled))

            if self.USE_NLU_CONFIDENCE_AS_SCORE:
                # the memoization will use the confidence of NLU on the latest
                # user message to set the confidence of the action
                score = tracker.latest_message.intent.get("confidence", 1.0)
            else:
                score = 1.0

            result[recalled] = score
        else:
            logger.debug("There is no memorised next action")

        return result

    def persist(self, path):
        # type: (Text) -> None

        self.featurizer.persist(path)

        memorized_file = os.path.join(path, 'memorized_turns.json')
        data = {
            "max_history": self.max_history,
            "lookup": self.lookup
        }
        utils.create_dir_for_file(memorized_file)
        utils.dump_obj_as_json_to_file(memorized_file, data)

    @classmethod
    def load(cls, path):
        # type: (Text) -> MemoizationPolicy

        featurizer = TrackerFeaturizer.load(path)
        memorized_file = os.path.join(path, 'memorized_turns.json')
        if os.path.isfile(memorized_file):
            with io.open(memorized_file) as f:
                data = json.loads(f.read())
            return cls(featurizer=featurizer, lookup=data["lookup"])
        else:
            logger.info("Couldn't load memoization for policy. "
                        "File '{}' doesn't exist. Falling back to empty "
                        "turn memory.".format(memorized_file))
            return cls()
