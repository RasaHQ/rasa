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

ENABLE_FEATURE_STRING_COMPRESSION = True


class MemoizationPolicy(Policy):
    SUPPORTS_ONLINE_TRAINING = True

    @classmethod
    def _standard_featurizer(cls, max_history=None):
        max_history = max_history or cls.MAX_HISTORY_DEFAULT
        # Memoization policy always uses MaxHistoryTrackerFeaturizer
        # without state_featurizer
        return MaxHistoryTrackerFeaturizer(None, max_history)

    def __init__(self, max_history=None, lookup=None):
        # type: (int, Optional[Dict]) -> None

        featurizer = self._standard_featurizer(max_history)
        super(MemoizationPolicy, self).__init__(featurizer)

        self.max_history = self.featurizer.max_history
        self.lookup = lookup if lookup is not None else {}
        self.is_enabled = True

    def toggle(self, activate):
        # type: (bool) -> None
        self.is_enabled = activate

    def _memorise(self, trackers_as_states, trackers_as_actions, domain):
        self.lookup = {}
        self._add(trackers_as_states, trackers_as_actions, domain)

    def _create_partial_histories(self, states):
        augmented = [list(states)]
        augmented_states = list(states)
        for i in range(self.max_history - 1):
            augmented_states[i] = None
            augmented.append(list(augmented_states))
        return augmented

    def _add(self, trackers_as_states, trackers_as_actions,
             domain, online=False):

        assert len(trackers_as_states[0]) == self.max_history, \
            ("Trying to mem featurized data with {} historic turns. Expected: "
             "{}".format(len(trackers_as_states[0]), self.max_history))

        ambiguous_feature_keys = set()

        pbar = tqdm(zip(trackers_as_states, trackers_as_actions),
                    desc="Processed actions", disable=online)
        for states, actions in pbar:
            action = actions[0]
            for i, states_augmented in enumerate(
                    self._create_partial_histories(states)):
                feature_key = self._create_feature_key(states_augmented)
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
                                            "".format(states_augmented,
                                                      action))
                                self.lookup[feature_key] = feature_item
                            else:
                                # delete contradicting example created by
                                # partial history augmentation from memory
                                ambiguous_feature_keys.add(feature_key)
                                del self.lookup[feature_key]
                    else:
                        self.lookup[feature_key] = feature_item

                pbar.set_postfix({
                    "# examples": len(self.lookup)})

    @staticmethod
    def _create_feature_key(states):
        feature_str = json.dumps(states, sort_keys=True).replace("\"", "")
        if ENABLE_FEATURE_STRING_COMPRESSION:
            compressed = zlib.compress(bytes(feature_str, "utf-8"))
            return base64.b64encode(compressed).decode("utf-8")
        else:
            return feature_str

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: **Any
              ):
        # type: (...) -> Dict[Text: Any]
        """Trains the policy on given training trackers."""
        (trackers_as_states,
         trackers_as_actions) = self.featurizer.training_states_and_actions(
                                    training_trackers, domain)

        self._memorise(trackers_as_states,
                       trackers_as_actions,
                       domain)
        logger.info("Memorized {} unique augmented examples."
                    "".format(len(self.lookup)))

    def continue_training(self, trackers, domain, **kwargs):
        # type: (List[DialogueStateTracker], Domain, **Any) -> None

        # add only the last tracker, because it is the only new one
        (trackers_as_states,
         trackers_as_actions,
         _) = self.featurizer.training_states_and_actions(trackers[-1:],
                                                          domain)
        # fit to one extra example
        self._add(trackers_as_states, trackers_as_actions,
                  domain, online=True)

    def _recall(self, states):
        return self.lookup.get(self._create_feature_key(states))

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts the next action the bot should take
        after seeing the tracker.

        Returns the list of probabilities for the next actions.
        If memorized action was found returns 1.0 for its index,
        else returns 0.0 for all actions."""

        tracker_as_states = self.featurizer.prediction_states([tracker], domain)
        states = tracker_as_states[0]
        logger.debug('Current tracker state {}'.format(states))

        result = [0.0] * domain.num_actions

        memorised = self._recall(states)
        if memorised is not None and self.is_enabled:
            logger.debug("Used memorised next action '{}'".format(memorised))
            result[memorised] = 1.0
        return result

    def persist(self, path):
        # type: (Text) -> None

        super(MemoizationPolicy, self).persist(path)

        memorized_file = os.path.join(path, 'memorized_turns.json')
        data = {
            "lookup": self.lookup
        }
        utils.create_dir_for_file(memorized_file)
        utils.dump_obj_as_json_to_file(memorized_file, data)

    @classmethod
    def load(cls, path):
        # type: (Text) -> MemoizationPolicy

        featurizer = TrackerFeaturizer.load(path)
        assert isinstance(featurizer, MaxHistoryTrackerFeaturizer), \
            ("Loaded featurizer of type {}, should be "
             "MaxHistoryTrackerFeaturizer.".format(type(featurizer).__name__))

        memorized_file = os.path.join(path, 'memorized_turns.json')
        if os.path.isfile(memorized_file):
            with io.open(memorized_file) as f:
                data = json.loads(f.read())
            return cls(featurizer.max_history, data["lookup"])
        else:
            logger.info("Couldn't load memoization for policy. "
                        "File '{}' doesn't exist. Falling back to empty "
                        "turn memory.".format(memorized_file))
            return MemoizationPolicy()
